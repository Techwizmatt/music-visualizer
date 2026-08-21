from __future__ import annotations

"""
Real-time beat tracking and prediction.

Pipeline (all causal, cheap enough for a background thread):
  1. Spectral flux onset envelope: 1024-sample FFT frames, hop 512
     (~93 frames/sec @ 48k), half-wave-rectified per-bin magnitude rises,
     log-compressed, high-passed against a local average.
  2. Tempo: autocorrelation of the last ~6s of the envelope over the
     60-190 BPM lag range, weighted by a log-Gaussian prior around 120 BPM
     plus harmonic support (half/double lags). Hysteresis so the tempo only
     switches after repeated evidence; octave errors folded into 84-168.
  3. Phase: a 4-period comb over the recent envelope finds where beats sit;
     a PLL-style nudge keeps the anchor locked without jitter.
  4. Prediction: the beat grid (anchor + period) extrapolates forward, so
     callers evaluate the clock slightly in the FUTURE to cancel the
     capture->analysis->render latency. That anticipation is what makes the
     visuals feel live instead of trailing the speakers.

The tracker works in "stream time" (seconds of audio consumed); the caller
maintains the stream->monotonic mapping.

AnalysisCache persists {track sig: bpm} so a song's second play locks
instantly (the tempo prior is seeded before the first beat even lands).
"""

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from settings import (
    ANALYSIS_CACHE_MAX_ENTRIES,
    ANALYSIS_CACHE_PATH,
    BEAT_MAX_BPM,
    BEAT_MIN_BPM,
)

FFT_N = 1024
HOP = 512


@dataclass
class BeatQuery:
    bpm: float
    period: float          # seconds
    conf: float            # 0..1
    phase: float           # 0..1 within the current beat (0 = on the beat)
    time_since_beat: float # seconds
    beat_index: int        # running count (mod 4 gives a "bar" position)


class BeatTracker:
    ENV_FPS_NOMINAL = 48000 / HOP

    def __init__(self, samplerate: float = 48000.0) -> None:
        self._lock = Lock()
        self.reset(samplerate)

    def reset(self, samplerate: float) -> None:
        with self._lock:
            self._sr = float(samplerate)
            self._env_fps = self._sr / HOP
            self._window = np.hanning(FFT_N).astype(np.float32)
            self._prev_mag: Optional[np.ndarray] = None
            self._env: List[float] = []
            self._env_bass: List[float] = []         # <250Hz flux, for PHASE
            self._bass_hi_bin = max(2, int(250.0 * FFT_N / samplerate))
            self._env_max = int(self._env_fps * 8)   # keep ~8s
            self._frames = 0                          # hops processed
            self._period: float = 0.0                 # seconds; 0 = no lock
            self._bpm: float = 0.0
            self._conf: float = 0.0
            self._anchor: float = 0.0                 # stream-time of a beat
            self._dissent_period: float = 0.0
            self._dissent_votes: int = 0
            self._prior_bpm: float = 0.0
            self._last_energy: float = 0.0

    # ------------------------------------------------------------ input

    def set_prior_bpm(self, bpm: float) -> None:
        """Seed tempo from the analysis cache: the grid starts at this period
        immediately (low confidence) and the comb only has to find phase."""
        with self._lock:
            if bpm and BEAT_MIN_BPM <= bpm <= BEAT_MAX_BPM * 1.05:
                self._prior_bpm = float(bpm)
                if self._period <= 0.0:
                    self._period = 60.0 / float(bpm)
                    self._bpm = float(bpm)
                    self._conf = 0.20

    def seed_grid(self, period: float, anchor_stream_t: float, conf: float = 0.35) -> None:
        """Install a full predicted grid (tempo AND phase) from a cached track
        profile — beats land correctly before live analysis has warmed up."""
        with self._lock:
            if period <= 0:
                return
            self._period = float(period)
            self._bpm = 60.0 / float(period)
            self._anchor = float(anchor_stream_t)
            self._conf = max(self._conf, float(conf))
            self._prior_bpm = self._bpm

    def process_hop(self, frame: np.ndarray) -> None:
        """`frame` is the latest FFT_N mono samples (50% overlap with the
        previous call). Appends one onset-envelope value."""
        mag = np.abs(np.fft.rfft(frame * self._window))
        mag = np.log1p(mag * 4.0)
        # Ignore >8kHz: cymbals wash out the flux up there.
        hi_bin = int(8000.0 * FFT_N / self._sr)
        mag = mag[:hi_bin]

        with self._lock:
            if self._prev_mag is not None and len(self._prev_mag) == len(mag):
                rise = np.maximum(0.0, mag - self._prev_mag)
                flux = float(np.sum(rise))
                # Bass-only flux: kicks live down here, hi-hats don't. Tempo
                # uses the full band; PHASE prefers this one so the grid
                # locks to kicks instead of off-beat hats.
                flux_bass = float(np.sum(rise[: self._bass_hi_bin]))
            else:
                flux = 0.0
                flux_bass = 0.0
            self._prev_mag = mag
            self._env.append(flux)
            self._env_bass.append(flux_bass)
            if len(self._env) > self._env_max:
                del self._env[: len(self._env) - self._env_max]
            if len(self._env_bass) > self._env_max:
                del self._env_bass[: len(self._env_bass) - self._env_max]
            self._frames += 1
            self._last_energy = float(np.mean(mag))

    # ------------------------------------------------------------ estimation

    def _lag_range(self) -> Tuple[int, int]:
        lo = max(2, int(round(self._env_fps * 60.0 / BEAT_MAX_BPM)))
        hi = int(round(self._env_fps * 60.0 / BEAT_MIN_BPM))
        return lo, hi

    def estimate(self) -> None:
        """Run tempo + phase estimation over the recent envelope. Call every
        ~0.5s from the analysis thread."""
        with self._lock:
            env = np.array(self._env, dtype=np.float64)
            env_bass = np.array(self._env_bass, dtype=np.float64)
            fps = self._env_fps
            frames = self._frames
        need = int(fps * 3.0)
        if len(env) < need:
            return

        def detrend(e: np.ndarray) -> np.ndarray:
            kernel = max(3, int(fps * 0.09))
            pad = np.concatenate([np.full(kernel, e[0] if len(e) else 0.0), e])
            local = np.convolve(pad, np.ones(kernel) / kernel, mode="same")[kernel:]
            return np.maximum(0.0, e - local)

        oss = detrend(env)

        if float(np.mean(oss)) < 1e-6:
            with self._lock:
                self._conf *= 0.85
            return

        lo, hi = self._lag_range()
        n = len(oss)
        hi = min(hi, n // 2)
        if hi <= lo + 2:
            return

        o = oss - np.mean(oss)
        # Autocorrelation over the candidate lag range.
        ac = np.array([float(np.dot(o[: n - lag], o[lag:])) / (n - lag) for lag in range(lo, hi + 1)])
        ac = np.maximum(0.0, ac)
        if float(ac.max(initial=0.0)) <= 0.0:
            return

        lags = np.arange(lo, hi + 1, dtype=np.float64)
        bpms = 60.0 * fps / lags
        center = self._prior_bpm if self._prior_bpm else 120.0
        prior = np.exp(-0.5 * ((np.log2(bpms / center)) / 0.9) ** 2)

        # Harmonic support: a true beat lag also scores at half/double.
        support = ac.copy()
        for i, lag in enumerate(lags):
            for mult in (0.5, 2.0):
                j = int(round(lag * mult)) - lo
                if 0 <= j < len(ac):
                    support[i] += 0.5 * ac[j]
        score = support * prior

        best_i = int(np.argmax(score))
        # Parabolic refinement of the lag peak.
        if 0 < best_i < len(score) - 1:
            y0, y1, y2 = score[best_i - 1], score[best_i], score[best_i + 1]
            denom = (y0 - 2 * y1 + y2)
            shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            shift = float(np.clip(shift, -0.5, 0.5))
        else:
            shift = 0.0
        best_lag = (lo + best_i + shift)
        cand_period = best_lag / fps
        cand_bpm = 60.0 / cand_period

        # Fold octave errors into a comfortable range.
        while cand_bpm < 84.0 and cand_bpm * 2.0 <= BEAT_MAX_BPM:
            cand_bpm *= 2.0
        while cand_bpm > 168.0 and cand_bpm / 2.0 >= BEAT_MIN_BPM:
            cand_bpm /= 2.0
        cand_period = 60.0 / cand_bpm

        # Peak sharpness -> confidence.
        mean_ac = float(np.mean(ac)) + 1e-12
        cand_conf = float(np.clip((ac[best_i] / mean_ac - 1.0) / 4.0, 0.0, 1.0))

        # Phase: comb over the bass envelope when the track has real low-end
        # (locks to kicks, not off-beat hats); otherwise the full envelope.
        oss_bass = detrend(env_bass) if len(env_bass) == len(env) else oss
        use_bass = float(np.sum(oss_bass)) > 0.04 * float(np.sum(oss))
        p_frames = cand_period * fps
        cand_anchor = self._comb_phase(oss_bass if use_bass else oss, p_frames, frames, fps)

        with self._lock:
            self._apply_estimate(cand_period, cand_bpm, cand_conf, cand_anchor, frames, fps)

    def _comb_phase(self, oss: np.ndarray, p_frames: float, frames_total: int, fps: float) -> float:
        n = len(oss)
        k_weights = (1.0, 0.85, 0.7, 0.55)
        steps = int(max(8, round(p_frames)))
        best_off, best_s = 0.0, -1.0
        for s in range(steps):
            off = s * p_frames / steps
            total = 0.0
            for k, wk in enumerate(k_weights):
                idx = n - 1 - off - k * p_frames
                i0 = int(idx)
                if i0 < 0:
                    break
                frac = idx - i0
                v = oss[i0] * (1 - frac) + (oss[min(n - 1, i0 + 1)] * frac)
                total += wk * v
            if total > best_s:
                best_s, best_off = total, off
        # Stream time of that most recent beat.
        end_stream_t = frames_total * HOP / self._sr
        return end_stream_t - (best_off / fps)

    def _apply_estimate(
        self, cand_period: float, cand_bpm: float, cand_conf: float,
        cand_anchor: float, frames_total: int, fps: float,
    ) -> None:
        now_stream = frames_total * HOP / self._sr

        if self._period <= 0.0:
            self._period, self._bpm = cand_period, cand_bpm
            self._anchor = cand_anchor
            self._conf = cand_conf * 0.7
            return

        rel = abs(cand_period - self._period) / self._period
        if rel < 0.045:
            # Agreement: converge period and PLL-nudge the phase.
            self._period += (cand_period - self._period) * 0.35
            self._bpm = 60.0 / self._period
            err = cand_anchor - self._anchor
            err -= round(err / self._period) * self._period  # shortest wrap
            self._anchor += err * (0.30 if cand_conf > 0.3 else 0.12)
            self._conf += (cand_conf - self._conf) * 0.30
            self._dissent_votes = 0
        else:
            # Disagreement: require sustained evidence before switching.
            if self._dissent_period and abs(cand_period - self._dissent_period) / self._dissent_period < 0.05:
                self._dissent_votes += 1
            else:
                self._dissent_period = cand_period
                self._dissent_votes = 1
            self._conf *= 0.92
            if self._dissent_votes >= 3 and cand_conf > 0.25:
                self._period, self._bpm = cand_period, cand_bpm
                self._anchor = cand_anchor
                self._conf = cand_conf * 0.6
                self._dissent_votes = 0
                self._dissent_period = 0.0

        # Keep the anchor near "now" so float error never accumulates.
        if self._period > 0:
            n = math.floor((now_stream - self._anchor) / self._period)
            self._anchor += n * self._period

    # ------------------------------------------------------------ queries

    def query(self, stream_t: float) -> Optional[BeatQuery]:
        with self._lock:
            period, conf, anchor, bpm = self._period, self._conf, self._anchor, self._bpm
        if period <= 0.0:
            return None
        n = math.floor((stream_t - anchor) / period)
        tsb = stream_t - (anchor + n * period)
        return BeatQuery(
            bpm=bpm, period=period, conf=conf,
            phase=float(tsb / period), time_since_beat=float(tsb),
            beat_index=int(n),
        )

    def debug_envelope(self, seconds: float = 3.0) -> Tuple[List[float], float]:
        """(recent onset envelope, env fps) for the debug panel."""
        with self._lock:
            k = int(self._env_fps * seconds)
            return list(self._env[-k:]), self._env_fps

    @property
    def stream_time(self) -> float:
        with self._lock:
            return self._frames * HOP / self._sr


# ---------------------------------------------------------------- profile


class TrackProfile:
    """
    Per-track analysis 'signature' — derived data only, never audio:
      * tempo (bpm) + confidence
      * beat grid phase expressed in TRACK time (`beat_offset`), so a replay
        can place beats correctly from the current playback position alone
      * 1s-resolution energy profile (bass/mid/high/rms, quantized to 0..255)
        so visuals can anticipate loud sections the next time the song plays
    """

    BANDS = ("bass", "mid", "high", "rms")
    RES_SEC = 1.0

    def __init__(self, sig: str, duration_sec: int) -> None:
        self.sig = sig
        self.duration = max(1, int(duration_sec))
        n = self.duration + 2
        self._acc = np.zeros((4, n), dtype=np.float64)
        self._cnt = np.zeros(n, dtype=np.int64)

    def add_sample(self, pos_sec: float, bass: float, mid: float, high: float, rms: float) -> None:
        i = int(pos_sec / self.RES_SEC)
        if 0 <= i < self._acc.shape[1]:
            self._acc[:, i] += (bass, mid, high, rms)
            self._cnt[i] += 1

    @property
    def coverage(self) -> float:
        return float(np.count_nonzero(self._cnt[: self.duration])) / max(1, self.duration)

    @property
    def seconds_recorded(self) -> int:
        return int(np.count_nonzero(self._cnt))

    def bands_entry(self) -> Dict[str, List[int]]:
        cnt = np.maximum(1, self._cnt)
        avg = self._acc / cnt
        out = {}
        for bi, name in enumerate(self.BANDS):
            q = np.clip(np.round(avg[bi] * 255.0), 0, 255).astype(int)
            q[self._cnt == 0] = 0
            out[name] = q[: self.duration].tolist()
        return out


class LoadedProfile:
    """Read-only view of a cached entry for playback-time anticipation."""

    def __init__(self, entry: Dict[str, Any]) -> None:
        self.bpm = float(entry.get("bpm") or 0.0)
        self.conf = float(entry.get("conf") or 0.0)
        self.beat_offset = entry.get("beat_offset")  # track-time of a beat, sec
        self.duration = int(entry.get("duration") or 0)
        self._bands: Dict[str, np.ndarray] = {}
        bands = entry.get("bands")
        if isinstance(bands, dict):
            for name in TrackProfile.BANDS:
                arr = bands.get(name)
                if isinstance(arr, list) and arr:
                    self._bands[name] = np.array(arr, dtype=np.float64) / 255.0

    @property
    def has_energy(self) -> bool:
        return bool(self._bands)

    def energy_at(self, pos_sec: float, band: str = "rms") -> Optional[float]:
        arr = self._bands.get(band)
        if arr is None or len(arr) == 0:
            return None
        i = int(pos_sec / TrackProfile.RES_SEC)
        if i < 0 or i >= len(arr):
            return None
        v = float(arr[i])
        return v if v > 0.0 else None


# ---------------------------------------------------------------- cache


class AnalysisCache:
    """Persistent map: track signature -> TrackProfile data. Lets a repeat
    play lock tempo AND beat phase before a single beat has hit."""

    def __init__(self, path: str = ANALYSIS_CACHE_PATH) -> None:
        self._path = path
        self._lock = Lock()

    def _read(self) -> Dict[str, Any]:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def load(self, sig: str) -> Optional[LoadedProfile]:
        entry = self._read().get(sig)
        if isinstance(entry, dict) and float(entry.get("bpm") or 0.0) > 0:
            return LoadedProfile(entry)
        return None

    def save(
        self,
        sig: str,
        bpm: float,
        conf: float,
        beat_offset: Optional[float] = None,
        profile: Optional[TrackProfile] = None,
        duration: int = 0,
    ) -> None:
        if not sig or bpm <= 0:
            return
        with self._lock:
            data = self._read()
            old = data.get(sig) if isinstance(data.get(sig), dict) else {}
            entry: Dict[str, Any] = dict(old)
            # Keep the higher-confidence tempo measurement.
            if conf >= float(old.get("conf", 0.0)) - 0.05:
                entry["bpm"] = round(float(bpm), 2)
                entry["conf"] = round(float(conf), 3)
                if beat_offset is not None:
                    entry["beat_offset"] = round(float(beat_offset), 4)
            if profile is not None and profile.seconds_recorded >= 15:
                old_cov = float(old.get("coverage", 0.0))
                if profile.coverage >= old_cov - 0.02:
                    entry["bands"] = profile.bands_entry()
                    entry["coverage"] = round(profile.coverage, 3)
            entry["duration"] = int(duration or old.get("duration") or 0)
            entry["saved_at"] = time.time()
            entry["plays"] = int(old.get("plays", 0)) + (0 if old else 1)
            data[sig] = entry

            if len(data) > ANALYSIS_CACHE_MAX_ENTRIES:
                items = sorted(data.items(), key=lambda kv: float((kv[1] or {}).get("saved_at", 0)))
                for k, _ in items[: len(data) - ANALYSIS_CACHE_MAX_ENTRIES]:
                    data.pop(k, None)
            try:
                d = os.path.dirname(os.path.abspath(self._path)) or "."
                fd, tmp = tempfile.mkstemp(prefix=".analysis_", dir=d)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
                os.replace(tmp, self._path)
            except Exception:
                pass
