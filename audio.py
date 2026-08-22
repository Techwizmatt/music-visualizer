from __future__ import annotations

"""
Live audio capture + analysis from a loopback input device (BlackHole).

Two background threads:
  * device manager — owns the sounddevice InputStream, reconnects if the
    device vanishes. The PortAudio callback writes mono samples into a ring
    buffer and queues the untouched stereo float stream for recording.
  * beat thread — consumes the ring in fixed 512-sample hops, feeds the
    BeatTracker (onset envelope -> tempo -> phase).

The now-playing timestamp also maps every captured block into a deterministic
per-track WAV.  On a later exact metadata+duration match, a full-file profile
is loaded/generated in a background process.  Live analysis remains active
until that profile is ready, then snapshots are read from it at track time.

`snapshot()` (UI thread, every frame) returns smoothed band levels plus the
PREDICTED beat clock evaluated slightly in the future
(BEAT_PREDICT_LOOKAHEAD_SEC), so visuals pulse with the speakers instead of
trailing them. `levels.beat` is the blend: predicted pulse when the tracker
is confident, reactive pulse otherwise.
"""

import math
import ctypes
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from beat import HOP, BeatTracker
from recording import FullTrackProfile, TrackFileManager, TrackIdentity
from settings import (
    AUDIO_BLOCK_SIZE,
    AUDIO_CAPTURE_QUEUE_BLOCKS,
    AUDIO_DEVICE_SUBSTRING,
    AUDIO_ENABLED,
    AUDIO_FFT_SIZE,
    AUDIO_PREFERRED_SAMPLERATE,
    AUDIO_RECONNECT_SEC,
    AUDIO_SILENCE_HINT_SEC,
    BEAT_ENABLED,
    BEAT_PREDICT_LOOKAHEAD_SEC,
)

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - missing portaudio etc.
    sd = None


def _set_audio_thread_priority(qos_class: int) -> None:
    """Apply a Darwin QoS class when available; remain portable otherwise."""
    if sys.platform != "darwin":
        return
    try:
        fn = ctypes.CDLL(None).pthread_set_qos_class_self_np
        fn.argtypes = (ctypes.c_uint, ctypes.c_int)
        fn.restype = ctypes.c_int
        fn(int(qos_class), 0)
    except Exception:
        pass


@dataclass
class AudioLevels:
    """Everything the visualizer needs, all values already smoothed to [0..1]."""
    ok: bool = False                 # stream currently running
    silent: bool = True              # no meaningful signal
    rms: float = 0.0                 # overall loudness
    bass: float = 0.0                # ~25-130 Hz
    mid: float = 0.0                 # ~130-2000 Hz
    high: float = 0.0                # ~2-9 kHz
    pulse: float = 0.0               # reactive bass transient (no prediction)
    bands: List[float] = field(default_factory=list)  # 24 log-spaced bins 0..1
    waveform: List[float] = field(default_factory=list)  # signed 64-point audio trace
    waveform_timeline: List[List[float]] = field(default_factory=list)
    waveform_timeline_center: int = 0
    status: str = "off"              # off | no-device | starting | ok | silent | file
    source: str = "live"             # live | full-file
    # --- predicted beat clock (evaluated with lookahead) ---
    bpm: float = 0.0
    beat_conf: float = 0.0
    beat_phase: float = 0.0          # 0 on the (predicted) beat
    beat: float = 0.0                # blended pulse: use THIS to drive visuals
    energy_ahead: float = 0.0        # full-file profile loudness ~1.5s ahead
    # --- complete-file-only detail, indexed from the now-playing clock ---
    vocal: float = 0.0               # voice-like harmonic energy, not source separation
    brightness: float = 0.0          # normalized spectral centroid
    spectral_flux: float = 0.0       # dense note/transient activity
    stereo_width: float = 0.0        # side energy relative to centered energy
    section: float = 0.0             # stable normalized section identifier
    section_change: float = 0.0      # decaying pulse at detected boundaries
    music_motion: float = 0.0        # sustained full-spectrum activity
    energy_flow: float = 0.0         # -1 release/implode .. +1 build/explode
    spectral_shift: float = 0.0      # -1 darker/downward .. +1 brighter/upward
    climax: float = 0.0              # combined high-point/drop strength
    track_intensity: float = 0.0     # relative slow passage .. dense passage
    buildup: float = 0.0             # sustained structural rise (-1 release)
    anticipation: float = 0.0        # smooth ramp into a detected large jump
    drop: float = 0.0                # chorus/drop impact with a soft decay
    calmness: float = 0.0            # relative slow-section strength


def _band_edges(n_bands: int, f_lo: float, f_hi: float) -> np.ndarray:
    return np.geomspace(f_lo, f_hi, n_bands + 1)


class AudioAnalyzer:
    N_BANDS = 24
    RING_SIZE = 1 << 17     # ~2.7s @ 48k — headroom for the hop reader

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ring = np.zeros(self.RING_SIZE, dtype=np.float32)
        self._write_idx = 0
        self._total_written = 0
        self._samplerate: float = float(AUDIO_PREFERRED_SAMPLERATE)
        self._device_name: str = ""

        self._stream = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_queue: "queue.Queue[Optional[Tuple[np.ndarray, float, float]]]" = queue.Queue(
            maxsize=max(64, int(AUDIO_CAPTURE_QUEUE_BLOCKS))
        )
        self._capture_ready = threading.Event()
        self._capture_queue_drops = 0
        self._input_overflows = 0
        self._capture_peak = 0.0
        self._status = "off"

        # Analysis state (only touched from the snapshot() caller's thread)
        self._window = np.hanning(AUDIO_FFT_SIZE).astype(np.float32)
        self._last_t = time.monotonic()
        self._sm_rms = 0.0
        self._sm_bass = 0.0
        self._sm_mid = 0.0
        self._sm_high = 0.0
        self._sm_bands = np.zeros(self.N_BANDS)
        self._bass_slow = 0.0
        self._pulse = 0.0
        self._agc_peak = 1e-6        # running peak for auto-gain
        self._last_signal_t = 0.0

        # --- beat engine ---
        self._tracker = BeatTracker(self._samplerate)
        self._beat_thread: Optional[threading.Thread] = None
        self._total_read = 0
        self._prev_tail = np.zeros(HOP, dtype=np.float32)
        self._stream_mono_off: Optional[float] = None   # mono_t - stream_t
        self._last_estimate_t = 0.0

        self._track_lock = threading.Lock()
        self._sig = ""
        self._full_profile: Optional[FullTrackProfile] = None
        self._pos_pair: Optional[Tuple[float, float, bool]] = None
        self._files = TrackFileManager(self._on_full_profile)

    # ---------- lifecycle ----------

    def start(self) -> None:
        if not AUDIO_ENABLED or sd is None:
            self._status = "off"
            return
        if self._thread is not None:
            return
        self._files.start()
        self._capture_thread = threading.Thread(
            target=self._capture_run,
            name="audio-capture-priority",
            daemon=True,
        )
        self._capture_thread.start()
        self._thread = threading.Thread(target=self._run, name="audio-analyzer", daemon=True)
        self._thread.start()
        if BEAT_ENABLED:
            self._beat_thread = threading.Thread(target=self._beat_run, name="beat-tracker", daemon=True)
            self._beat_thread.start()

    def stop(self) -> None:
        self._stop = True
        self._close_stream()
        try:
            self._capture_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=3.0)
        self._files.stop()

    # ---------- track context (called from the UI thread) ----------

    def set_track(
        self,
        sig: str,
        duration_sec: float,
        title: str = "",
        artist: str = "",
        album: str = "",
        genre: str = "",
        track_number: Optional[int] = None,
        total_track_count: Optional[int] = None,
        source_app: str = "",
        content_identifier: str = "",
    ) -> None:
        """Close the old file boundary and select the exact new media file."""
        with self._track_lock:
            if sig == self._sig:
                return
            duration = float(max(0.0, duration_sec or 0.0))
            self._sig = sig
            self._full_profile = None
            self._pos_pair = None
        self._files.set_track(
            TrackIdentity(
                sig=sig,
                title=title,
                artist=artist,
                album=album,
                duration=duration,
                genre=genre,
                track_number=track_number,
                total_track_count=total_track_count,
                source_app=source_app,
                content_identifier=content_identifier,
            )
        )

    def note_artwork(self, artwork_bytes: Optional[bytes]) -> None:
        self._files.note_artwork(artwork_bytes)

    def _on_full_profile(self, sig: str, profile: FullTrackProfile) -> None:
        """Called off the UI thread after a saved file has been analyzed."""
        with self._track_lock:
            if sig == self._sig:
                self._full_profile = profile

    def note_position(self, pos_sec: float, playing: bool = True) -> None:
        """Feed exact media time/state to both recording and file playback."""
        now = time.monotonic()
        with self._track_lock:
            self._pos_pair = (float(pos_sec), now, bool(playing))
        self._files.note_position(pos_sec, playing)

    # ---------- device management ----------

    def _find_device(self) -> Optional[int]:
        try:
            devices = sd.query_devices()
        except Exception:
            return None
        needle = AUDIO_DEVICE_SUBSTRING.lower()
        for i, dev in enumerate(devices):
            try:
                if needle in str(dev["name"]).lower() and int(dev["max_input_channels"]) >= 1:
                    return i
            except Exception:
                continue
        return None

    def _close_stream(self) -> None:
        st = self._stream
        self._stream = None
        if st is not None:
            try:
                st.stop()
                st.close()
            except Exception:
                pass

    def _open_stream(self, device_idx: int) -> bool:
        channels = 2
        try:
            info = sd.query_devices(device_idx)
            channels = max(1, min(2, int(info["max_input_channels"])))
            default_sr = float(info.get("default_samplerate") or AUDIO_PREFERRED_SAMPLERATE)
            self._device_name = str(info.get("name") or "")
        except Exception:
            default_sr = float(AUDIO_PREFERRED_SAMPLERATE)

        # MP3 supports at most 48 kHz. Capture float32 stereo at that rate so
        # the encoder receives its highest useful quality without the 4x
        # callback/disk load of a 192 kHz BlackHole configuration.
        sample_rates = tuple(dict.fromkeys((float(AUDIO_PREFERRED_SAMPLERATE), default_sr)))
        for sr in sample_rates:
            try:
                # Reset the hop reader and tracker for the new stream clock.
                with self._lock:
                    self._total_written = 0
                    self._total_read = 0
                self._prev_tail = np.zeros(HOP, dtype=np.float32)
                self._stream_mono_off = None
                self._tracker.reset(sr)

                stream = sd.InputStream(
                    device=device_idx,
                    channels=channels,
                    samplerate=sr,
                    blocksize=AUDIO_BLOCK_SIZE,
                    dtype="float32",
                    callback=self._callback,
                )
                # Set before start(): CoreAudio may invoke the first callback
                # synchronously as the stream transitions to active.
                self._samplerate = sr
                self._capture_ready.clear()
                stream.start()
                self._stream = stream
                return True
            except Exception:
                continue
        return False

    def _run(self) -> None:
        # Device discovery/reconnect is intentionally lower priority than the
        # worker that drains already-arrived CoreAudio blocks.
        _set_audio_thread_priority(0x11)  # QOS_CLASS_UTILITY
        while not self._stop:
            if self._stream is not None:
                try:
                    active = bool(self._stream.active)
                except Exception:
                    active = False
                if not active:
                    self._close_stream()
                    self._status = "no-device"
                time.sleep(0.5)
                continue

            idx = self._find_device()
            if idx is None:
                self._status = "no-device"
                time.sleep(AUDIO_RECONNECT_SEC)
                continue

            self._status = "starting"
            if self._open_stream(idx):
                # "ok" means BlackHole is delivering frames, not merely that
                # CoreAudio accepted the open request.
                self._capture_ready.wait(timeout=1.0)
                self._status = "ok"
            else:
                self._status = "no-device"
                time.sleep(AUDIO_RECONNECT_SEC)

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        """Real-time CoreAudio callback: one owned copy and a nonblocking put.

        No FFT, downmix, timeline lookup, disk access, or recorder locking is
        allowed here. Those operations run on the priority capture worker.
        """
        try:
            callback_mono = time.monotonic()
            packet_end_mono = callback_mono
            # PortAudio reports when the first input sample reached the ADC.
            # Convert its clock to monotonic at callback time, then use the
            # block end for exact placement on the now-playing timeline.
            try:
                pa_now = float(getattr(time_info, "currentTime"))
                pa_start = float(getattr(time_info, "inputBufferAdcTime"))
                block_end = pa_start + float(frames) / max(1.0, self._samplerate)
                input_delay = pa_now - block_end
                if pa_now > 0.0 and pa_start > 0.0 and -0.05 <= input_delay <= 2.0:
                    packet_end_mono = callback_mono - max(0.0, input_delay)
            except (AttributeError, TypeError, ValueError):
                pass
            if status and bool(getattr(status, "input_overflow", False)):
                self._input_overflows += 1
            owned = np.array(indata, dtype=np.float32, order="C", copy=True)
            try:
                self._capture_queue.put_nowait((owned, self._samplerate, packet_end_mono))
            except queue.Full:
                self._capture_queue_drops += 1
        except Exception:
            pass

    def _capture_run(self) -> None:
        """High-priority bridge from CoreAudio to analysis and file writing."""
        _set_audio_thread_priority(0x21)  # QOS_CLASS_USER_INTERACTIVE
        while True:
            try:
                item = self._capture_queue.get(timeout=0.25)
            except queue.Empty:
                if self._stop:
                    break
                continue
            if item is None:
                break
            samples, samplerate, callback_mono = item
            try:
                mono = (
                    np.mean(samples, axis=1, dtype=np.float32)
                    if samples.ndim == 2
                    else samples
                )
                n = len(mono)
                if n:
                    peak = float(np.max(np.abs(samples)))
                    self._capture_peak = max(peak, self._capture_peak * 0.998)
                    self._capture_ready.set()
                with self._lock:
                    ring = self._ring
                    size = len(ring)
                    i = self._write_idx
                    first = min(n, size - i)
                    ring[i:i + first] = mono[:first]
                    rest = n - first
                    if rest > 0:
                        ring[:rest] = mono[first:]
                    self._write_idx = (i + n) % size
                    self._total_written += n
                self._files.push_audio_owned(samples, samplerate, callback_mono)
            except Exception:
                # A bad analysis block must never stop future capture blocks.
                continue

    # ---------- beat thread ----------

    def _beat_run(self) -> None:
        _set_audio_thread_priority(0x19)  # QOS_CLASS_USER_INITIATED
        while not self._stop:
            if self._stream is None:
                time.sleep(0.25)
                continue

            processed = False
            while True:
                with self._lock:
                    avail = self._total_written - self._total_read
                    size = len(self._ring)
                    if avail > size - HOP * 4:
                        # Fell behind (system stall): jump close to the head,
                        # keeping hop alignment.
                        skip = (avail - HOP * 8) // HOP * HOP
                        if skip > 0:
                            self._total_read += skip
                            avail -= skip
                    if avail < HOP:
                        break
                    start = self._total_read % size
                    if start + HOP <= size:
                        chunk = self._ring[start:start + HOP].copy()
                    else:
                        k = size - start
                        chunk = np.concatenate((self._ring[start:], self._ring[:HOP - k]))
                    self._total_read += HOP
                frame = np.concatenate((self._prev_tail, chunk))
                self._prev_tail = chunk
                try:
                    self._tracker.process_hop(frame)
                except Exception:
                    pass
                processed = True

            now = time.monotonic()
            if processed:
                # Map tracker stream-time onto the monotonic clock (EMA damps
                # scheduling jitter; a constant residual is absorbed by the
                # user-facing lookahead setting).
                off = now - self._tracker.stream_time
                if self._stream_mono_off is None:
                    self._stream_mono_off = off
                else:
                    self._stream_mono_off += (off - self._stream_mono_off) * 0.05

            if now - self._last_estimate_t >= 0.5:
                self._last_estimate_t = now
                try:
                    self._tracker.estimate()
                except Exception:
                    pass
            time.sleep(0.03)

    # ---------- analysis ----------

    def _latest_window(self) -> np.ndarray:
        with self._lock:
            i = self._write_idx
            ring = self._ring
            n = AUDIO_FFT_SIZE
            if i >= n:
                return ring[i - n:i].copy()
            return np.concatenate((ring[-(n - i):], ring[:i])).copy()

    @staticmethod
    def _smooth(cur: float, target: float, dt: float, tau_up: float, tau_down: float) -> float:
        tau = tau_up if target > cur else tau_down
        if tau <= 0:
            return target
        a = 1.0 - math.exp(-dt / tau)
        return cur + (target - cur) * a

    def _beat_fields(self, now_mono: float, gate_ok: bool) -> Tuple[float, float, float, float, float]:
        """(bpm, conf, phase, blended_beat, energy_ahead)"""
        beat = self._pulse
        bpm = conf = phase = e_ahead = 0.0
        if BEAT_ENABLED and self._stream_mono_off is not None:
            stream_now = now_mono - self._stream_mono_off
            q = self._tracker.query(stream_now + BEAT_PREDICT_LOOKAHEAD_SEC)
            if q is not None:
                bpm, conf, phase = q.bpm, q.conf, q.phase
                cg = max(0.0, min(1.0, (conf - 0.25) / 0.45))
                if gate_ok and cg > 0.0:
                    predicted = math.exp(-q.time_since_beat / 0.11) * cg
                    beat = max(self._pulse * (1.0 - 0.55 * cg), predicted)
        return bpm, conf, phase, beat, e_ahead

    def _full_file_snapshot(self) -> Optional[AudioLevels]:
        with self._track_lock:
            profile = self._full_profile
            pair = self._pos_pair
        if profile is None or pair is None:
            return None
        pos, stamp, playing = pair
        age = time.monotonic() - stamp
        if age > 2.0:
            return None
        if playing:
            pos += min(age, 0.1)
        if not playing:
            return AudioLevels(
                ok=True,
                silent=True,
                bands=[0.0] * self.N_BANDS,
                status="file",
                source="full-file",
                bpm=profile.bpm,
                beat_conf=profile.confidence,
            )

        # The file is indexed in track time, so a seek or a GarageBand loop
        # lands on the matching pre-analyzed frame immediately.
        values = profile.sample(pos + BEAT_PREDICT_LOOKAHEAD_SEC)
        if values is None:
            return None
        core = values[:12]
        (
            bass, mid, high, rms, pulse, beat,
            vocal, brightness, spectral_flux, stereo_width,
            section, section_change,
        ) = core
        ahead = profile.sample(pos + 1.25)
        behind = profile.sample(max(0.0, pos - 0.85))
        energy_ahead = ahead[3] if ahead is not None else 0.0
        if ahead is not None and behind is not None:
            energy_flow = float(np.clip((ahead[3] - behind[3]) * 2.4, -1.0, 1.0))
            spectral_shift = float(np.clip(
                ((ahead[2] - ahead[0]) - (behind[2] - behind[0])) * 1.7,
                -1.0,
                1.0,
            ))
        else:
            energy_flow = spectral_shift = 0.0
        music_motion = float(np.clip(
            0.18 * rms + 0.14 * bass + 0.16 * mid + 0.10 * high
            + 0.20 * spectral_flux + 0.16 * vocal + 0.10 * stereo_width,
            0.0,
            1.0,
        ))
        climax = float(np.clip(
            0.36 * rms + 0.22 * spectral_flux + 0.34 * section_change
            + 0.20 * pulse + 0.10 * high,
            0.0,
            1.0,
        ))
        phase = 0.0
        if profile.bpm > 0:
            period = 60.0 / profile.bpm
            phase = ((pos + BEAT_PREDICT_LOOKAHEAD_SEC - profile.beat_offset) % period) / period
        fine = values[12:36]
        bands = list(fine) if len(fine) == 24 else ([bass] * 5 + [mid] * 12 + [high] * 7)
        waveform = list(values[36:100])
        # Recorded playback can show both sides of the playhead. Twenty-five
        # traces span six seconds: earlier music, current time, upcoming music.
        waveform_timeline: List[List[float]] = []
        timeline_offsets = np.linspace(-3.0, 3.0, 25)
        for offset in timeline_offsets:
            sample_time = pos + float(offset)
            timeline_row = (
                profile.sample(sample_time)
                if 0.0 <= sample_time <= profile.duration
                else None
            )
            if timeline_row is not None and len(timeline_row) >= 100:
                waveform_timeline.append(list(timeline_row[36:100]))
            else:
                waveform_timeline.append([0.0] * 64)
        if len(values) >= 105:
            track_intensity, buildup, anticipation, drop, calmness = values[100:105]
        else:
            track_intensity = music_motion
            buildup = energy_flow
            anticipation = drop = 0.0
            calmness = 1.0 - track_intensity
        energy_flow = float(np.clip(0.48 * energy_flow + 0.52 * buildup, -1.0, 1.0))
        music_motion = float(np.clip(
            0.30 * music_motion + 0.50 * track_intensity
            + 0.14 * anticipation + 0.18 * drop,
            0.0,
            1.0,
        ))
        climax = max(climax, float(drop))
        return AudioLevels(
            ok=True,
            silent=rms < 0.004,
            rms=rms,
            bass=bass,
            mid=mid,
            high=high,
            pulse=pulse,
            bands=bands,
            waveform=waveform,
            waveform_timeline=waveform_timeline,
            waveform_timeline_center=len(waveform_timeline) // 2,
            status="file",
            source="full-file",
            bpm=profile.bpm,
            beat_conf=profile.confidence,
            beat_phase=phase,
            beat=beat,
            energy_ahead=energy_ahead,
            vocal=vocal,
            brightness=brightness,
            spectral_flux=spectral_flux,
            stereo_width=stereo_width,
            section=section,
            section_change=section_change,
            music_motion=music_motion,
            energy_flow=energy_flow,
            spectral_shift=spectral_shift,
            climax=climax,
            track_intensity=track_intensity,
            buildup=buildup,
            anticipation=anticipation,
            drop=drop,
            calmness=calmness,
        )

    def snapshot(self) -> AudioLevels:
        now = time.monotonic()
        dt = max(1e-3, min(0.1, now - self._last_t))
        self._last_t = now

        full = self._full_file_snapshot()
        if full is not None:
            return full

        if self._stream is None or self._status not in ("ok", "silent"):
            self._sm_rms = self._smooth(self._sm_rms, 0.0, dt, 0.05, 0.6)
            self._pulse = self._smooth(self._pulse, 0.0, dt, 0.05, 0.25)
            return AudioLevels(ok=False, silent=True, status=self._status,
                               bands=[0.0] * self.N_BANDS, beat=0.0)

        x = self._latest_window()
        rms_raw = float(np.sqrt(np.mean(x * x)) + 1e-12)

        silent = rms_raw < 1e-5
        if not silent:
            self._last_signal_t = now
        long_silent = (now - self._last_signal_t) > AUDIO_SILENCE_HINT_SEC
        self._status = "silent" if (silent and long_silent) else "ok"

        spec = np.abs(np.fft.rfft(x * self._window))
        freqs = np.fft.rfftfreq(AUDIO_FFT_SIZE, 1.0 / self._samplerate)
        power = spec * spec

        def band_energy(f_lo: float, f_hi: float) -> float:
            m = (freqs >= f_lo) & (freqs < f_hi)
            if not np.any(m):
                return 0.0
            return float(np.sqrt(np.mean(power[m])))

        bass_raw = band_energy(25.0, 130.0)
        mid_raw = band_energy(130.0, 2000.0)
        high_raw = band_energy(2000.0, 9000.0)

        edges = _band_edges(self.N_BANDS, 30.0, 12000.0)
        bands_raw = np.array([band_energy(edges[k], edges[k + 1]) for k in range(self.N_BANDS)])

        frame_peak = max(bass_raw, mid_raw, high_raw, float(bands_raw.max(initial=0.0)))
        self._agc_peak = max(self._agc_peak * math.exp(-dt / 6.0), frame_peak, 1e-6)
        g = 1.0 / self._agc_peak

        def norm(v: float) -> float:
            return float(min(1.0, max(0.0, (v * g) ** 0.8)))

        bass_n, mid_n, high_n = norm(bass_raw), norm(mid_raw), norm(high_raw)
        rms_n = float(min(1.0, (rms_raw * g * 2.2) ** 0.7))

        self._sm_rms = self._smooth(self._sm_rms, rms_n, dt, 0.045, 0.35)
        self._sm_bass = self._smooth(self._sm_bass, bass_n, dt, 0.035, 0.28)
        self._sm_mid = self._smooth(self._sm_mid, mid_n, dt, 0.045, 0.30)
        self._sm_high = self._smooth(self._sm_high, high_n, dt, 0.045, 0.30)

        for k in range(self.N_BANDS):
            self._sm_bands[k] = self._smooth(
                float(self._sm_bands[k]), norm(float(bands_raw[k])), dt, 0.04, 0.30
            )

        self._bass_slow = self._smooth(self._bass_slow, bass_n, dt, 0.6, 0.6)
        onset = max(0.0, bass_n - self._bass_slow * 1.15)
        self._pulse = max(self._pulse * math.exp(-dt / 0.18), min(1.0, onset * 2.4))

        bpm, bconf, bphase, beat, e_ahead = self._beat_fields(now, gate_ok=not silent)
        wave_idx = np.linspace(0, max(0, len(x) - 1), 64).astype(np.int64)
        wave = x[wave_idx] if len(x) else np.zeros(64, dtype=np.float32)
        wave_scale = max(1e-6, float(np.percentile(np.abs(wave), 98.0)))
        wave = np.clip(wave / wave_scale, -1.0, 1.0)

        return AudioLevels(
            ok=True,
            silent=silent,
            rms=self._sm_rms,
            bass=self._sm_bass,
            mid=self._sm_mid,
            high=self._sm_high,
            pulse=self._pulse,
            bands=[float(b) for b in self._sm_bands],
            waveform=[float(v) for v in wave],
            status=self._status,
            source="live",
            bpm=bpm,
            beat_conf=bconf,
            beat_phase=bphase,
            beat=beat,
            energy_ahead=e_ahead,
        )

    # ---------- debug ----------

    def debug_info(self) -> Dict[str, Any]:
        env, env_fps = self._tracker.debug_envelope(3.0)
        now = time.monotonic()
        q = None
        if self._stream_mono_off is not None:
            q = self._tracker.query(now - self._stream_mono_off)
        with self._track_lock:
            profile, pair = self._full_profile, self._pos_pair
        file_state = self._files.status()
        use_file = profile is not None and pair is not None
        bpm = profile.bpm if use_file else (q.bpm if q else 0.0)
        conf = profile.confidence if use_file else (q.conf if q else 0.0)
        period = (60.0 / bpm) if bpm > 0 else 0.0
        beat_idx = (q.beat_index % 4) if q else 0
        next_in_ms = ((q.period - q.time_since_beat) * 1000.0) if q else 0.0
        if use_file and pair is not None and period > 0:
            pos, _, _ = pair
            rel = pos - profile.beat_offset
            beat_idx = int(math.floor(rel / period)) % 4
            next_in_ms = (period - (rel % period)) * 1000.0
        return {
            "env": env,
            "env_fps": env_fps,
            "bpm": bpm,
            "conf": conf,
            "period": period,
            "next_in_ms": next_in_ms,
            "beat_idx": beat_idx,
            "lookahead_ms": BEAT_PREDICT_LOOKAHEAD_SEC * 1000.0,
            "analysis_source": "full-file" if use_file else "live",
            "record_mode": file_state.get("mode", "idle"),
            "record_coverage": float(file_state.get("coverage", 0.0) or 0.0),
            "record_bitrate": int(file_state.get("bitrate", 0) or 0),
            "record_dropped": int(file_state.get("dropped", 0) or 0),
            "capture_queue": self._capture_queue.qsize(),
            "capture_ready": self._capture_ready.is_set(),
            "capture_dropped": int(self._capture_queue_drops),
            "input_overflows": int(self._input_overflows),
            "capture_peak": float(self._capture_peak),
            "device": self._device_name,
            "sr": int(self._samplerate),
            "status": "file" if use_file else self._status,
            "clock_ok": use_file or self._stream_mono_off is not None,
        }
