from __future__ import annotations

"""Lossless, timeline-aligned track recording and full-file analysis.

There is deliberately no global analysis index.  A deterministic filename,
derived from the complete now-playing identity (including duration), is the
only lookup key. Derived visual analysis is returned in memory and is never
saved.

The PortAudio callback only copies blocks into a bounded queue.  Disk I/O is
performed by a writer thread, while the heavier whole-file FFT pass runs in a
separate process so neither capture nor the Qt render loop can be stalled.
"""

import concurrent.futures
from collections import deque
import ctypes
import hashlib
import json
import math
import multiprocessing
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from settings import (
    AUDIO_FULL_PROFILE_FPS,
    AUDIO_MP3_BITRATE_KBPS,
    AUDIO_RECORDING_DIR,
    AUDIO_RECORDING_MIN_COVERAGE,
)


WAV_HEADER_BYTES = 44
_COVERAGE_RES_SEC = 0.01
# Layer-III synthesis can overshoot its input peak by several dB on tonal
# material. Four dB of codec headroom keeps decoded playback below full scale.
_MP3_TARGET_PEAK_DBFS = -4.0
_MAX_NORMALIZATION_GAIN_DB = 48.0
_RECORDING_FORMAT_VERSION = "2"


def _set_recorder_thread_priority() -> None:
    """Ask macOS for user-initiated QoS without making startup depend on it."""
    if sys.platform != "darwin":
        return
    try:
        # pthread/qos.h: QOS_CLASS_USER_INITIATED
        fn = ctypes.CDLL(None).pthread_set_qos_class_self_np
        fn.argtypes = (ctypes.c_uint, ctypes.c_int)
        fn.restype = ctypes.c_int
        fn(0x19, 0)
    except Exception:
        pass


def _lower_analysis_process_priority() -> None:
    """Keep whole-file feature extraction behind capture and UI work."""
    try:
        os.nice(10)
    except Exception:
        pass


@dataclass(frozen=True)
class TrackIdentity:
    sig: str
    title: str
    artist: str
    album: str
    duration: float
    genre: str = ""
    track_number: Optional[int] = None
    total_track_count: Optional[int] = None
    source_app: str = ""
    content_identifier: str = ""


@dataclass(frozen=True)
class TrackPaths:
    base: Path
    audio: Path

    def partial_audio(self, samplerate: int) -> Path:
        return self.base.parent / ".work" / f"{self.base.name}.{samplerate}Hz.partial.wav"

def make_audio_signature(
    title: str, artist: str, album: str, duration_sec: float
) -> str:
    """Exact audio identity made only from the current now-playing fields."""
    duration_ms = int(round(max(0.0, float(duration_sec)) * 1000.0))
    return "|||".join(
        ((title or "").strip(), (artist or "").strip(), (album or "").strip(), f"{duration_ms}ms")
    )


def _safe_component(value: str, limit: int) -> str:
    value = unicodedata.normalize("NFKC", value or "").strip()
    value = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "-", value)
    value = re.sub(r"\s+", " ", value).strip(" .-")
    return (value or "Unknown")[:limit].rstrip(" .-")


def paths_for_track(track: TrackIdentity, root: Optional[str] = None) -> TrackPaths:
    """Return a readable but collision-safe filename for a media identity."""
    root = AUDIO_RECORDING_DIR if root is None else root
    root_path = Path(root).expanduser().resolve()
    # If the user plays one of our finished files to verify it, Music may use
    # its basename as now-playing title and omit artist/album. Reuse that exact
    # MP3 instead of recursively recording the recording.
    direct_name = Path(track.title or "").name
    if direct_name == (track.title or "") and not track.artist and not track.album:
        candidate = root_path / (direct_name if direct_name.lower().endswith(".mp3") else f"{direct_name}.mp3")
        if candidate.exists():
            return TrackPaths(base=candidate.with_suffix(""), audio=candidate)
    digest = hashlib.sha256(track.sig.encode("utf-8", "replace")).hexdigest()[:12]
    artist = _safe_component(track.artist, 48)
    title = _safe_component(track.title, 64)
    album = _safe_component(track.album, 40) if track.album else ""
    human = f"{artist} - {title}"
    if album:
        human += f" - {album}"
    duration_label = f"{track.duration:.3f}".rstrip("0").rstrip(".")
    human += f" [{duration_label}s] [{digest}]"
    base = root_path / human
    return TrackPaths(
        base=base,
        audio=Path(f"{base}.mp3"),
    )


def _is_working_recording_title(title: str) -> bool:
    value = (title or "").lower()
    return ".partial" in value or bool(re.search(r"\.\d+hz\.partial$", value))


class FullTrackProfile:
    """Dense, read-only visual values indexed by the media-reported time."""

    def __init__(
        self,
        *,
        sig: str,
        duration: float,
        fps: float,
        bpm: float,
        confidence: float,
        beat_offset: float,
        values: np.ndarray,
        audio_path: str,
    ) -> None:
        self.sig = sig
        self.duration = float(duration)
        self.fps = float(fps)
        self.bpm = float(bpm)
        self.confidence = float(confidence)
        self.beat_offset = float(beat_offset)
        self.audio_path = audio_path
        # columns: bass, mid, high, rms, pulse, beat, vocal presence,
        # brightness, spectral flux, stereo width, section, section change,
        # then 24 normalized log-spaced spectrum bins and a signed 64-point
        # audio trace for real waveform geometry, followed by relative track
        # intensity, buildup, anticipation, drop impact, and calmness.
        self._values = np.asarray(values, dtype=np.float32)

    def sample(self, position_sec: float) -> Optional[Tuple[float, ...]]:
        if self.fps <= 0 or len(self._values) == 0:
            return None
        x = max(0.0, float(position_sec)) * self.fps
        if x > len(self._values) - 1:
            if position_sec > self.duration + 0.5:
                return None
            x = float(len(self._values) - 1)
        i0 = int(math.floor(x))
        i1 = min(len(self._values) - 1, i0 + 1)
        frac = x - i0
        row = self._values[i0] * (1.0 - frac) + self._values[i1] * frac
        return tuple(float(v) for v in row)  # type: ignore[return-value]


def _float_wav_header(samplerate: int, channels: int, frames: int) -> bytes:
    """Canonical RIFF/WAVE header for interleaved IEEE float32 samples."""
    data_bytes = int(frames) * int(channels) * 4
    if data_bytes > 0xFFFFFFFF - 36:
        raise ValueError("recording is too long for a standard WAV file")
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        3,  # WAVE_FORMAT_IEEE_FLOAT
        channels,
        samplerate,
        samplerate * channels * 4,
        channels * 4,
        32,
        b"data",
        data_bytes,
    )


def _encode_mp3(
    wav_path: Path,
    mp3_path: Path,
    track: TrackIdentity,
    artwork_bytes: Optional[bytes],
    source_peak: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to create the finished MP3")
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{mp3_path.name}.", suffix=".mp3", dir=str(mp3_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    artwork_path: Optional[Path] = None
    try:
        peak = max(0.0, float(source_peak))
        target_peak = 10.0 ** (_MP3_TARGET_PEAK_DBFS / 20.0)
        if peak > 1e-7:
            gain_db = min(
                _MAX_NORMALIZATION_GAIN_DB,
                20.0 * math.log10(target_peak / peak),
            )
        else:
            gain_db = 0.0
        gain = 10.0 ** (gain_db / 20.0)

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-threads", "1",
            "-i", str(wav_path),
        ]
        if artwork_bytes:
            suffix = ".png" if artwork_bytes.startswith(b"\x89PNG") else ".jpg"
            afd, artwork_name = tempfile.mkstemp(prefix=".cover.", suffix=suffix, dir=str(mp3_path.parent))
            with os.fdopen(afd, "wb") as artwork_file:
                artwork_file.write(artwork_bytes)
            artwork_path = Path(artwork_name)
            cmd += ["-i", str(artwork_path), "-map", "0:a:0", "-map", "1:v:0"]
        else:
            cmd += ["-map", "0:a:0"]

        now_playing = {
            "title": track.title,
            "artist": track.artist,
            "album": track.album,
            "genre": track.genre,
            "track_number": track.track_number,
            "total_track_count": track.total_track_count,
            "duration_seconds": track.duration,
            "source_app": track.source_app,
            "content_identifier": track.content_identifier,
            "signature": track.sig,
        }
        cmd += [
            "-af", f"volume={gain:.12g}:precision=double",
            "-c:a", "libmp3lame",
            "-b:a", f"{int(AUDIO_MP3_BITRATE_KBPS)}k",
            "-ar", "48000",
            "-ac", "2",
            "-id3v2_version", "3",
            "-map_metadata", "-1",
            "-metadata", f"title={track.title}",
            "-metadata", f"artist={track.artist}",
            "-metadata", f"album={track.album}",
            "-metadata", f"genre={track.genre}",
            "-metadata", f"comment={json.dumps(now_playing, ensure_ascii=False, separators=(',', ':'))}",
            "-metadata", f"now_playing_duration_ms={int(round(track.duration * 1000.0))}",
            "-metadata", f"now_playing_source_app={track.source_app}",
            "-metadata", f"now_playing_content_identifier={track.content_identifier}",
            "-metadata", f"recording_source_peak_dbfs={20.0 * math.log10(max(peak, 1e-12)):.3f}",
            "-metadata", f"recording_normalization_gain_db={gain_db:.3f}",
            "-metadata", f"recording_format_version={_RECORDING_FORMAT_VERSION}",
        ]
        if track.track_number is not None:
            track_value = str(track.track_number)
            if track.total_track_count:
                track_value += f"/{track.total_track_count}"
            cmd += ["-metadata", f"track={track_value}"]
        if artwork_path is not None:
            cmd += [
                "-c:v", "copy",
                "-disposition:v", "attached_pic",
                "-metadata:s:v", "title=Album cover",
                "-metadata:s:v", "comment=Cover (front)",
            ]
        cmd.append(str(tmp_path))
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"MP3 encoding failed: {result.stderr.strip()}")
        os.replace(tmp_path, mp3_path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        if artwork_path is not None:
            try:
                artwork_path.unlink()
            except OSError:
                pass


class _WriteSession:
    def __init__(
        self,
        track: TrackIdentity,
        paths: TrackPaths,
        samplerate: int,
        channels: int,
        coverage: Optional[np.ndarray] = None,
    ) -> None:
        self.track = track
        self.paths = paths
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.frames = int(math.ceil(track.duration * samplerate))
        self.partial_audio = paths.partial_audio(self.samplerate)
        coverage_size = max(1, int(math.ceil(track.duration / _COVERAGE_RES_SEC)))
        self._coverage = (
            np.asarray(coverage, dtype=np.bool_).copy()
            if coverage is not None and len(coverage) == coverage_size
            else np.zeros(coverage_size, dtype=np.bool_)
        )
        self._last_end_pos: Optional[float] = None
        self._last_packet_mono: Optional[float] = None
        self._fh = None

        paths.base.parent.mkdir(parents=True, exist_ok=True)
        self.partial_audio.parent.mkdir(parents=True, exist_ok=True)
        if coverage is not None and self.partial_audio.exists():
            self._fh = self.partial_audio.open("r+b", buffering=1024 * 1024)
        else:
            self._fh = self.partial_audio.open("w+b", buffering=1024 * 1024)
            self._fh.write(_float_wav_header(self.samplerate, self.channels, self.frames))
            self._fh.truncate(WAV_HEADER_BYTES + self.frames * self.channels * 4)

    @property
    def coverage(self) -> float:
        return float(np.mean(self._coverage)) if len(self._coverage) else 0.0

    def _complete_enough(self) -> bool:
        covered = np.flatnonzero(self._coverage)
        if len(covered) == 0 or self.coverage < AUDIO_RECORDING_MIN_COVERAGE:
            return False
        first_t = float(covered[0]) * _COVERAGE_RES_SEC
        last_t = float(covered[-1] + 1) * _COVERAGE_RES_SEC
        return first_t <= 0.50 and last_t >= self.track.duration - 0.50

    def write(self, start_pos: float, samples: np.ndarray, packet_mono: float) -> bool:
        if self._fh is None or len(samples) == 0:
            return False
        block_sec = len(samples) / self.samplerate
        # Between adjacent callbacks, use sample continuity instead of the
        # slightly jittery media timestamp.  A pause/seek re-anchors it.
        if (
            self._last_end_pos is not None
            and self._last_packet_mono is not None
            and packet_mono - self._last_packet_mono < max(0.50, block_sec * 8.0)
            and abs(start_pos - self._last_end_pos) < 0.50
        ):
            start_pos = self._last_end_pos

        frame0_raw = int(round(start_pos * self.samplerate))
        src0 = max(0, -frame0_raw)
        frame0 = max(0, frame0_raw)
        count = min(len(samples) - src0, self.frames - frame0)
        if count <= 0:
            return False

        block = np.ascontiguousarray(samples[src0:src0 + count, : self.channels], dtype="<f4")
        self._fh.seek(WAV_HEADER_BYTES + frame0 * self.channels * 4)
        self._fh.write(block.tobytes(order="C"))

        actual_start = frame0 / self.samplerate
        actual_end = (frame0 + count) / self.samplerate
        c0 = max(0, int(actual_start / _COVERAGE_RES_SEC))
        c1 = min(len(self._coverage), int(math.ceil(actual_end / _COVERAGE_RES_SEC)))
        self._coverage[c0:c1] = True
        self._last_end_pos = actual_end
        self._last_packet_mono = packet_mono

        return self._complete_enough()

    def coverage_bits(self) -> np.ndarray:
        return self._coverage.copy()

    def close_partial(self) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self._fh = None

    def finalize(self, artwork_bytes: Optional[bytes] = None) -> Path:
        if self._fh is None:
            return self.paths.audio
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self._fh = None
        source_peak = 0.0
        with self.partial_audio.open("rb") as source:
            source.seek(WAV_HEADER_BYTES)
            while True:
                raw = source.read(4 * 1024 * 1024)
                if not raw:
                    break
                usable = len(raw) - (len(raw) % 4)
                if usable <= 0:
                    continue
                values = np.frombuffer(raw[:usable], dtype="<f4")
                if len(values):
                    source_peak = max(source_peak, float(np.max(np.abs(values))))
        _encode_mp3(
            self.partial_audio,
            self.paths.audio,
            self.track,
            artwork_bytes,
            source_peak,
        )
        try:
            self.partial_audio.unlink()
        except OSError:
            pass
        try:
            self.partial_audio.parent.rmdir()
        except OSError:
            pass
        return self.paths.audio


def _smooth_series(values: np.ndarray, fps: float, tau_up: float, tau_down: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    cur = 0.0
    dt = 1.0 / max(1.0, fps)
    for i, target in enumerate(values):
        tau = tau_up if target > cur else tau_down
        cur += (float(target) - cur) * (1.0 - math.exp(-dt / tau))
        out[i] = cur
    return out


def _robust_norm(values: np.ndarray, exponent: float = 0.72) -> np.ndarray:
    positive = values[values > 1e-12]
    if len(positive) == 0:
        return np.zeros_like(values, dtype=np.float64)
    scale = float(np.percentile(positive, 99.0)) + 1e-12
    return np.clip((values / scale) ** exponent, 0.0, 1.0)


def _tempo_and_phase(onsets: np.ndarray, fps: float) -> Tuple[float, float, float]:
    """Estimate one stable tempo/grid using the complete onset envelope."""
    if len(onsets) < int(fps * 8) or float(np.max(onsets, initial=0.0)) < 1e-5:
        return 0.0, 0.0, 0.0
    centered = onsets - float(np.mean(onsets))
    lo = max(2, int(round(fps * 60.0 / 190.0)))
    hi = min(len(centered) // 3, int(round(fps * 60.0 / 60.0)))
    if hi <= lo:
        return 0.0, 0.0, 0.0
    lags = np.arange(lo, hi + 1, dtype=np.int64)
    ac = np.array([
        float(np.dot(centered[:-lag], centered[lag:])) / max(1, len(centered) - lag)
        for lag in lags
    ])
    ac = np.maximum(0.0, ac)
    if float(np.max(ac, initial=0.0)) <= 0:
        return 0.0, 0.0, 0.0
    bpms = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)
    support = ac.copy()
    for i, lag in enumerate(lags):
        for multiple in (0.5, 2.0):
            target = int(round(lag * multiple))
            j = target - lo
            if 0 <= j < len(ac):
                support[i] += 0.45 * ac[j]
    score = support * prior
    best = int(np.argmax(score))
    period = float(lags[best]) / fps
    bpm = 60.0 / period
    while bpm < 84.0 and bpm * 2.0 <= 190.0:
        bpm *= 2.0
    while bpm > 168.0 and bpm / 2.0 >= 60.0:
        bpm /= 2.0
    period = 60.0 / bpm
    sharpness = float(ac[best] / (float(np.mean(ac)) + 1e-12))
    confidence = float(np.clip((sharpness - 1.0) / 4.0, 0.0, 1.0))

    times = np.arange(len(onsets), dtype=np.float64) / fps
    phases = np.linspace(0.0, period, max(32, int(round(period * fps * 4))), endpoint=False)
    weights = onsets ** 1.35
    phase_scores = np.empty(len(phases), dtype=np.float64)
    sigma = max(0.025, min(0.08, period * 0.08))
    for i, phase in enumerate(phases):
        wrapped = np.mod(times - phase + period * 0.5, period) - period * 0.5
        phase_scores[i] = float(np.sum(weights * np.exp(-0.5 * (wrapped / sigma) ** 2)))
    beat_offset = float(phases[int(np.argmax(phase_scores))])
    return bpm, confidence, beat_offset


def analyze_audio_file(
    audio_path: str,
    signature: str,
    duration: float,
    fps: float,
) -> FullTrackProfile:
    """Process-pool entry point: decode the MP3 and return only memory data."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to analyze recorded MP3 audio")
    samplerate = 48000
    frames = int(math.ceil(duration * samplerate))
    fps = float(max(10.0, min(100.0, fps)))
    hop = max(1, int(round(samplerate / fps)))
    window_n = max(hop, 256, int(round(samplerate * 0.085)))
    nfft = 1 << int(math.ceil(math.log2(window_n)))
    window = np.hanning(window_n).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / samplerate)
    masks = (
        (freqs >= 25.0) & (freqs < 130.0),
        (freqs >= 130.0) & (freqs < 2000.0),
        (freqs >= 2000.0) & (freqs < min(12000.0, samplerate * 0.49)),
    )
    fine_edges = np.geomspace(35.0, min(12000.0, samplerate * 0.49), 25)
    fine_masks = [
        (freqs >= fine_edges[i]) & (freqs < fine_edges[i + 1])
        for i in range(24)
    ]
    flux_mask = freqs < min(8000.0, samplerate * 0.49)
    count = max(1, int(math.ceil(frames / hop)))
    raw = np.zeros((count, 4), dtype=np.float64)
    raw_fine = np.zeros((count, 24), dtype=np.float64)
    raw_wave = np.zeros((count, 64), dtype=np.float64)
    # centroid, flatness, vocal-band share, stereo width
    descriptors = np.zeros((count, 4), dtype=np.float64)
    flux = np.zeros(count, dtype=np.float64)
    prev_mag: Optional[np.ndarray] = None
    proc = subprocess.Popen(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", audio_path,
            "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "2", "-ar", str(samplerate), "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None

    def read_samples(sample_count: int) -> np.ndarray:
        need = sample_count * 2 * 4
        chunks = bytearray()
        while len(chunks) < need:
            chunk = proc.stdout.read(need - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        decoded = np.frombuffer(bytes(chunks), dtype="<f4")
        decoded = decoded[: (len(decoded) // 2) * 2]
        return decoded.reshape((-1, 2))

    try:
        stereo = read_samples(window_n)
        if len(stereo) < window_n:
            stereo = np.pad(stereo, ((0, window_n - len(stereo)), (0, 0)))
        for i in range(count):
            mono = np.mean(stereo, axis=1)
            wave_idx = np.linspace(0, max(0, len(mono) - 1), 64).astype(np.int64)
            raw_wave[i] = mono[wave_idx]
            raw[i, 3] = float(np.sqrt(np.mean(mono * mono)) + 1e-12)
            mag = np.abs(np.fft.rfft(mono * window, n=nfft))
            power = mag * mag
            for band_i, mask in enumerate(masks):
                raw[i, band_i] = float(np.sqrt(np.mean(power[mask]))) if np.any(mask) else 0.0
            for band_i, mask in enumerate(fine_masks):
                raw_fine[i, band_i] = float(np.sqrt(np.mean(power[mask]))) if np.any(mask) else 0.0
            log_mag = np.log1p(mag[flux_mask] * 4.0)
            if prev_mag is not None:
                flux[i] = float(np.sum(np.maximum(0.0, log_mag - prev_mag)))
            prev_mag = log_mag

            audible = (freqs >= 25.0) & (freqs < min(12000.0, samplerate * 0.49))
            vocal_band = (freqs >= 160.0) & (freqs < 4200.0)
            total_power = float(np.sum(power[audible])) + 1e-18
            descriptors[i, 0] = float(np.sum(freqs[audible] * power[audible]) / total_power / 12000.0)
            audible_power = power[audible] + 1e-18
            descriptors[i, 1] = float(
                np.exp(np.mean(np.log(audible_power))) / (np.mean(audible_power) + 1e-18)
            )
            descriptors[i, 2] = float(np.sum(power[vocal_band]) / total_power)
            mid_signal = (stereo[:, 0] + stereo[:, 1]) * 0.5
            side_signal = (stereo[:, 0] - stereo[:, 1]) * 0.5
            mid_rms = float(np.sqrt(np.mean(mid_signal * mid_signal))) + 1e-12
            side_rms = float(np.sqrt(np.mean(side_signal * side_signal)))
            descriptors[i, 3] = float(np.clip(side_rms / mid_rms, 0.0, 1.0))
            if i + 1 < count:
                new = read_samples(hop)
                if len(new) < hop:
                    new = np.pad(new, ((0, hop - len(new)), (0, 0)))
                stereo = np.concatenate((stereo[hop:], new), axis=0)
        proc.stdout.read()  # drain encoder padding so ffmpeg exits cleanly
        proc.stdout.close()
        if proc.wait(timeout=10.0) != 0:
            raise RuntimeError("ffmpeg could not decode the recorded MP3")
    except Exception:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.kill()
        proc.wait()
        raise

    # Whole-track normalization is stable and repeatable; it does not chase a
    # moving live AGC peak, which is the key advantage of the complete sample.
    band_scale = float(np.percentile(raw[:, :3][raw[:, :3] > 1e-12], 99.0)) \
        if np.any(raw[:, :3] > 1e-12) else 1.0
    bands = np.clip((raw[:, :3] / max(1e-12, band_scale)) ** 0.78, 0.0, 1.0)
    rms = _robust_norm(raw[:, 3], exponent=0.70)
    for bi in range(3):
        bands[:, bi] = _smooth_series(bands[:, bi], fps, 0.045, 0.30)
    rms = _smooth_series(rms, fps, 0.045, 0.35)
    fine_scale = np.percentile(np.maximum(raw_fine, 1e-12), 99.0, axis=0)
    fine_bands = np.clip((raw_fine / np.maximum(1e-12, fine_scale[None, :])) ** 0.76, 0.0, 1.0)
    for bi in range(fine_bands.shape[1]):
        fine_bands[:, bi] = _smooth_series(fine_bands[:, bi], fps, 0.04, 0.26)
    wave_scale = float(np.percentile(np.abs(raw_wave), 99.0)) if raw_wave.size else 1.0
    waveform = np.clip(raw_wave / max(1e-8, wave_scale), -1.0, 1.0)

    brightness = np.clip(descriptors[:, 0] * 2.4, 0.0, 1.0)
    brightness = _smooth_series(brightness, fps, 0.10, 0.48)
    stereo_width = _smooth_series(np.clip(descriptors[:, 3], 0.0, 1.0), fps, 0.12, 0.55)
    harmonicity = np.clip(1.0 - descriptors[:, 1], 0.0, 1.0)
    vocal_share = np.clip((descriptors[:, 2] - 0.16) / 0.68, 0.0, 1.0)
    # This is intentionally a vocal-presence cue, not a claim of stem
    # separation. Stable centered harmonic energy in the speech/singing band
    # produces the strongest value and is excellent animation material.
    vocal = np.clip(vocal_share * (0.28 + 0.72 * harmonicity) * np.clip(rms * 1.55, 0.0, 1.0), 0.0, 1.0)
    vocal = _smooth_series(vocal, fps, 0.09, 0.46)
    spectral_flux = _smooth_series(_robust_norm(flux, exponent=0.72), fps, 0.035, 0.26)

    local_n = max(3, int(round(fps * 0.35)))
    local = np.convolve(flux, np.ones(local_n) / local_n, mode="same")
    onset = np.maximum(0.0, flux - local * 0.82)
    onset = _robust_norm(onset, exponent=0.72)
    impulses = np.zeros_like(onset)
    if len(onset) >= 3:
        peak = (onset[1:-1] >= onset[:-2]) & (onset[1:-1] > onset[2:]) & (onset[1:-1] > 0.06)
        impulses[1:-1][peak] = onset[1:-1][peak]
    pulse = np.zeros_like(onset)
    decay = math.exp(-1.0 / (fps * 0.16))
    for i in range(len(pulse)):
        pulse[i] = max(float(impulses[i]), (float(pulse[i - 1]) * decay) if i else 0.0)

    bpm, confidence, beat_offset = _tempo_and_phase(onset, fps)
    beat = pulse.copy()
    if bpm > 0 and confidence > 0:
        period = 60.0 / bpm
        timeline = np.arange(count, dtype=np.float64) / fps
        since = np.mod(timeline - beat_offset, period)
        gate = float(np.clip((confidence - 0.12) / 0.55, 0.0, 1.0))
        grid = np.exp(-since / 0.11) * gate
        beat = np.maximum(pulse * (1.0 - 0.35 * gate), grid)

    # Detect large, sustained changes in the complete recording. A two-second
    # comparison is long enough to ignore individual notes but reacts to
    # verse/chorus, breakdown, instrumentation, and mix-width transitions.
    features = np.column_stack((bands, rms, vocal, brightness, stereo_width, spectral_flux))
    smooth_n = min(count, max(1, int(round(fps * 0.8))))
    kernel = np.ones(smooth_n, dtype=np.float64) / smooth_n
    smooth_features = np.column_stack(
        [np.convolve(features[:, j], kernel, mode="same") for j in range(features.shape[1])]
    )
    lag = max(1, int(round(fps * 2.0)))
    novelty = np.zeros(count, dtype=np.float64)
    if count > lag:
        delta = smooth_features[lag:] - smooth_features[:-lag]
        novelty[lag:] = np.sqrt(np.mean(delta * delta, axis=1))
    novelty = _robust_norm(novelty, exponent=0.78)

    boundaries = np.zeros(count, dtype=np.float64)
    if count >= 3 and duration >= 5.0:
        threshold = max(0.24, float(np.percentile(novelty, 78.0)))
        candidates = np.flatnonzero(
            (novelty[1:-1] >= novelty[:-2])
            & (novelty[1:-1] > novelty[2:])
            & (novelty[1:-1] >= threshold)
        ) + 1
        min_gap = max(1, int(round(fps * 3.5)))
        last = -min_gap
        for idx in candidates:
            if int(idx) - last >= min_gap:
                boundaries[idx] = novelty[idx]
                last = int(idx)

    section_change = np.zeros(count, dtype=np.float64)
    section_decay = math.exp(-1.0 / max(1.0, fps * 0.85))
    section_ids = np.zeros(count, dtype=np.float64)
    section_no = 0
    for i in range(count):
        if boundaries[i] > 0:
            section_no += 1
        section_ids[i] = float(section_no % 9) / 8.0
        section_change[i] = max(
            float(boundaries[i]),
            float(section_change[i - 1]) * section_decay if i else 0.0,
        )

    # Whole-song structure. These signals intentionally operate over seconds,
    # not individual beats: intensity marks slow versus dense passages,
    # buildup compares the musical picture before/after each timestamp,
    # anticipation ramps into detected jumps, and drop_impact lands on them.
    activity = np.clip(
        0.26 * rms + 0.15 * bands[:, 0] + 0.17 * bands[:, 1]
        + 0.10 * bands[:, 2] + 0.14 * spectral_flux + 0.12 * vocal
        + 0.06 * stereo_width,
        0.0,
        1.0,
    )
    activity = _smooth_series(activity, fps, 0.28, 1.05)
    low_activity = float(np.percentile(activity, 12.0))
    high_activity = float(np.percentile(activity, 94.0))
    track_intensity = np.clip(
        (activity - low_activity) / max(1e-6, high_activity - low_activity),
        0.0,
        1.0,
    )
    track_intensity = _smooth_series(track_intensity, fps, 0.30, 0.85)

    prefix = np.concatenate(([0.0], np.cumsum(track_intensity, dtype=np.float64)))
    timeline_i = np.arange(count, dtype=np.int64)

    def interval_mean(start_sec: float, end_sec: float) -> np.ndarray:
        starts = np.clip(timeline_i + int(round(start_sec * fps)), 0, count)
        ends = np.clip(timeline_i + int(round(end_sec * fps)), 0, count)
        ends = np.maximum(ends, starts + 1)
        ends = np.minimum(ends, count)
        starts = np.minimum(starts, ends - 1)
        return (prefix[ends] - prefix[starts]) / np.maximum(1, ends - starts)

    past_picture = interval_mean(-2.2, -0.25)
    future_picture = interval_mean(0.25, 2.2)
    buildup = np.clip((future_picture - past_picture) * 2.4, -1.0, 1.0)
    buildup = _smooth_series(buildup, fps, 0.38, 0.72)

    before_jump = interval_mean(-1.35, -0.18)
    after_jump = interval_mean(0.0, 0.72)
    jump = np.maximum(0.0, after_jump - before_jump)
    drop_score = np.clip(
        jump * 2.9
        + section_change * (0.34 + 0.34 * track_intensity)
        + spectral_flux * track_intensity * 0.16,
        0.0,
        1.0,
    )
    drop_impulses = np.zeros(count, dtype=np.float64)
    if count >= 3 and duration >= 4.0:
        positive = drop_score[drop_score > 0.05]
        threshold = max(
            0.19,
            float(np.percentile(positive, 74.0)) if len(positive) else 1.0,
        )
        candidates = np.flatnonzero(
            (drop_score[1:-1] >= drop_score[:-2])
            & (drop_score[1:-1] > drop_score[2:])
            & (drop_score[1:-1] >= threshold)
        ) + 1
        min_drop_gap = max(1, int(round(fps * 3.2)))
        last_drop = -min_drop_gap
        for idx in candidates:
            if int(idx) - last_drop >= min_drop_gap:
                drop_impulses[idx] = drop_score[idx]
                last_drop = int(idx)

    anticipation = np.zeros(count, dtype=np.float64)
    anticipation_frames = max(1, int(round(fps * 3.0)))
    for idx in np.flatnonzero(drop_impulses > 0):
        start = max(0, int(idx) - anticipation_frames)
        rise = np.linspace(0.0, 1.0, int(idx) - start + 1)
        rise = rise * rise * (3.0 - 2.0 * rise)
        anticipation[start:int(idx) + 1] = np.maximum(
            anticipation[start:int(idx) + 1],
            rise * drop_impulses[idx],
        )
    buildup = np.clip(np.maximum(buildup, anticipation * 0.72), -1.0, 1.0)

    drop_impact = np.zeros(count, dtype=np.float64)
    drop_decay = math.exp(-1.0 / max(1.0, fps * 0.78))
    for i in range(count):
        drop_impact[i] = max(
            float(drop_impulses[i]),
            float(drop_impact[i - 1]) * drop_decay if i else 0.0,
        )
    calmness = _smooth_series(1.0 - track_intensity, fps, 0.55, 0.85)

    values = np.column_stack(
        (
            bands, rms, pulse, beat, vocal, brightness, spectral_flux,
            stereo_width, section_ids, section_change, fine_bands, waveform,
            track_intensity, buildup, anticipation, drop_impact, calmness,
        )
    ).astype(np.float32)
    return FullTrackProfile(
        sig=signature,
        duration=duration,
        fps=fps,
        bpm=bpm,
        confidence=confidence,
        beat_offset=beat_offset,
        values=values,
        audio_path=str(Path(audio_path).resolve()),
    )


@dataclass
class _Packet:
    track: TrackIdentity
    samplerate: int
    channels: int
    start_pos: float
    packet_mono: float
    samples: np.ndarray


@dataclass
class _RawPacket:
    samplerate: int
    channels: int
    packet_mono: float
    samples: np.ndarray


class TrackFileManager:
    """Coordinates callback-safe capture and asynchronous full-file reads."""

    def __init__(self, on_profile: Callable[[str, FullTrackProfile], None]) -> None:
        self._on_profile = on_profile
        self._lock = threading.Lock()
        self._current: Optional[TrackIdentity] = None
        self._current_paths: Optional[TrackPaths] = None
        self._capture_allowed = True
        self._current_has_audio = False
        self._position: Optional[Tuple[float, float, bool]] = None
        self._needs_preroll = False
        self._preroll: "deque[_RawPacket]" = deque()
        self._status: Dict[str, object] = {"mode": "idle", "coverage": 0.0, "dropped": 0}
        self._queue: "queue.Queue[Tuple[str, object]]" = queue.Queue(maxsize=4096)
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self._pending_analysis: set[str] = set()
        self._artwork_by_sig: Dict[str, bytes] = {}
        ctx = multiprocessing.get_context("spawn")
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
            initializer=_lower_analysis_process_priority,
        )

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._writer_run, name="audio-recorder", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop = True
        with self._lock:
            self._current = None
            self._current_paths = None
            self._current_has_audio = False
            self._position = None
        try:
            self._queue.put(("stop", None), timeout=1.0)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def set_track(self, track: TrackIdentity) -> None:
        paths = paths_for_track(track)
        with self._lock:
            if self._current == track:
                return
            self._current = track
            self._current_paths = paths
            self._capture_allowed = not _is_working_recording_title(track.title)
            self._current_has_audio = paths.audio.exists()
            self._position = None
            self._needs_preroll = True
            if not self._capture_allowed:
                mode = "working-file-ignored"
            elif self._current_has_audio:
                mode = "analyzing-file"
            else:
                mode = "waiting-for-position"
            self._status = {"mode": mode, "coverage": 0.0, "dropped": 0}
            try:
                self._queue.put_nowait(("switch", track))
            except queue.Full:
                pass
        if self._current_has_audio:
            self._schedule_profile(track, paths)

    def note_artwork(self, artwork_bytes: Optional[bytes]) -> None:
        if not artwork_bytes:
            return
        with self._lock:
            if self._current is None:
                return
            self._artwork_by_sig[self._current.sig] = bytes(artwork_bytes)
            if len(self._artwork_by_sig) > 16:
                oldest = next(iter(self._artwork_by_sig))
                self._artwork_by_sig.pop(oldest, None)

    def note_position(self, position_sec: float, playing: bool) -> None:
        now = time.monotonic()
        with self._lock:
            if self._current is not None:
                pos = float(position_sec)
                self._position = (pos, now, bool(playing))
                if self._capture_allowed and not self._current_has_audio:
                    current_mode = str(self._status.get("mode", ""))
                    if playing and current_mode in (
                        "waiting-for-position", "waiting-for-audio", "paused",
                    ):
                        self._status["mode"] = "waiting-for-audio"
                    elif not playing and current_mode not in (
                        "encoding-mp3", "analyzing-file", "file-ready",
                    ):
                        self._status["mode"] = "paused"
                # media-control can report a new identity a fraction of a
                # second after its audio has started. Reassign just that
                # matching pre-roll to the new timeline so sample zero is not
                # lost to polling latency.
                if self._needs_preroll and playing and self._capture_allowed:
                    paths = self._current_paths
                    if paths is not None and not self._current_has_audio:
                        for raw in self._preroll:
                            end_pos = pos - max(0.0, now - raw.packet_mono)
                            start_pos = end_pos - len(raw.samples) / max(1, raw.samplerate)
                            if end_pos <= -0.02:
                                continue
                            packet = _Packet(
                                track=self._current,
                                samplerate=raw.samplerate,
                                channels=raw.channels,
                                start_pos=start_pos,
                                packet_mono=raw.packet_mono,
                                samples=raw.samples,
                            )
                            try:
                                self._queue.put_nowait(("audio", packet))
                            except queue.Full:
                                self._status["dropped"] = int(self._status.get("dropped", 0)) + 1
                                break
                    self._needs_preroll = False

    def push_audio(self, indata: np.ndarray, samplerate: float, packet_mono: float) -> None:
        self._push_audio(indata, samplerate, packet_mono, owned=False)

    def push_audio_owned(self, indata: np.ndarray, samplerate: float, packet_mono: float) -> None:
        """Queue a callback copy whose lifetime is already owned by us."""
        self._push_audio(indata, samplerate, packet_mono, owned=True)

    def _push_audio(
        self,
        indata: np.ndarray,
        samplerate: float,
        packet_mono: float,
        *,
        owned: bool,
    ) -> None:
        arr = np.asarray(indata)
        if arr.ndim == 1:
            arr = arr[:, None]
        frames = len(arr)
        if frames <= 0:
            return
        sr = int(round(samplerate))
        channels = min(2, int(arr.shape[1]))
        view = arr[:, :channels]
        copied = (
            view
            if owned and view.dtype == np.float32 and view.flags.c_contiguous
            else np.ascontiguousarray(view, dtype=np.float32)
        )
        with self._lock:
            self._preroll.append(_RawPacket(sr, channels, packet_mono, copied))
            while self._preroll and packet_mono - self._preroll[0].packet_mono > 2.0:
                self._preroll.popleft()
            track, paths, pair = self._current, self._current_paths, self._position
            if track is None or paths is None:
                return
            if not self._capture_allowed or self._current_has_audio:
                return
            if pair is None:
                self._status["mode"] = "waiting-for-position"
                return
            pos, stamp, playing = pair
            if not playing:
                self._status["mode"] = "paused"
                return
            if packet_mono - stamp > 1.0:
                self._status["mode"] = "waiting-for-position"
                return
            # A PortAudio block can legitimately end just before the newest
            # now-playing poll because its input-buffer latency is known.
            age = max(-0.25, min(0.25, packet_mono - stamp))
            end_pos = pos + age
            start_pos = end_pos - frames / max(1, sr)
            packet = _Packet(
                track=track,
                samplerate=sr,
                channels=channels,
                start_pos=start_pos,
                packet_mono=packet_mono,
                samples=copied,
            )
            try:
                self._queue.put_nowait(("audio", packet))
            except queue.Full:
                self._status["dropped"] = int(self._status.get("dropped", 0)) + 1

    def status(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._status)

    def _set_status(self, **values: object) -> None:
        with self._lock:
            self._status.update(values)

    def _schedule_profile(self, track: TrackIdentity, paths: TrackPaths) -> None:
        key = str(paths.audio)
        with self._lock:
            if key in self._pending_analysis:
                return
            self._pending_analysis.add(key)
            if self._current == track:
                self._status["mode"] = "analyzing-file"
                self._status["coverage"] = 1.0
                self._status["bitrate"] = int(AUDIO_MP3_BITRATE_KBPS) * 1000
                self._status["audio_file"] = str(paths.audio)
        future = self._executor.submit(
            analyze_audio_file,
            str(paths.audio),
            track.sig,
            track.duration,
            float(AUDIO_FULL_PROFILE_FPS),
        )

        def done(fut: concurrent.futures.Future) -> None:
            with self._lock:
                self._pending_analysis.discard(key)
            try:
                profile = fut.result()
                if not isinstance(profile, FullTrackProfile) or profile.sig != track.sig:
                    raise ValueError("background analysis identity mismatch")
            except Exception as exc:
                if self._current == track:
                    self._set_status(mode="analysis-error", error=str(exc))
                return
            if self._current == track:
                self._set_status(mode="file-ready", coverage=1.0, audio_file=str(paths.audio))
            self._on_profile(track.sig, profile)

        future.add_done_callback(done)

    def _writer_run(self) -> None:
        _set_recorder_thread_priority()
        session: Optional[_WriteSession] = None
        coverage_cache: Dict[Tuple[str, int, int], np.ndarray] = {}

        def session_key(value: _WriteSession) -> Tuple[str, int, int]:
            return value.track.sig, value.samplerate, value.channels

        def close_and_remember(value: _WriteSession) -> None:
            coverage_cache[session_key(value)] = value.coverage_bits()
            value.close_partial()

        while True:
            try:
                kind, payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop:
                    break
                continue
            if kind == "stop":
                break
            if kind == "switch":
                if session is not None:
                    close_and_remember(session)
                    session = None
                continue
            if kind != "audio" or not isinstance(payload, _Packet):
                continue
            packet = payload
            paths = paths_for_track(packet.track)
            if paths.audio.exists():
                if session is not None:
                    close_and_remember(session)
                    session = None
                self._schedule_profile(packet.track, paths)
                continue
            if (
                session is None
                or session.track != packet.track
                or session.samplerate != packet.samplerate
                or session.channels != packet.channels
            ):
                if session is not None:
                    close_and_remember(session)
                key = (packet.track.sig, packet.samplerate, packet.channels)
                try:
                    session = _WriteSession(
                        packet.track,
                        paths,
                        packet.samplerate,
                        packet.channels,
                        coverage=coverage_cache.get(key),
                    )
                except Exception as exc:
                    session = None
                    self._set_status(mode="recording-error", error=str(exc))
                    continue
            try:
                complete = session.write(packet.start_pos, packet.samples, packet.packet_mono)
                self._set_status(
                    mode="recording",
                    coverage=session.coverage,
                    samplerate=session.samplerate,
                    channels=session.channels,
                    bitrate=session.samplerate * session.channels * 32,
                )
                if complete:
                    self._set_status(mode="encoding-mp3", coverage=1.0)
                    with self._lock:
                        artwork = self._artwork_by_sig.get(packet.track.sig)
                    session.finalize(artwork)
                    coverage_cache.pop(session_key(session), None)
                    with self._lock:
                        if self._current == packet.track:
                            self._current_has_audio = True
                    self._set_status(mode="analyzing-file", coverage=1.0)
                    self._schedule_profile(packet.track, paths)
                    session = None
            except Exception as exc:
                try:
                    session.close_partial()
                except Exception:
                    pass
                session = None
                self._set_status(mode="recording-error", error=str(exc))
        if session is not None:
            try:
                close_and_remember(session)
            except Exception:
                pass
