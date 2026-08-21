from __future__ import annotations

"""
Live audio capture + analysis from a loopback input device (BlackHole).

Two background threads:
  * device manager — owns the sounddevice InputStream, reconnects if the
    device vanishes. The PortAudio callback writes mono samples into a ring
    buffer.
  * beat thread — consumes the ring in fixed 512-sample hops, feeds the
    BeatTracker (onset envelope -> tempo -> phase), records the per-track
    analysis profile, and persists it to the AnalysisCache.

`snapshot()` (UI thread, every frame) returns smoothed band levels plus the
PREDICTED beat clock evaluated slightly in the future
(BEAT_PREDICT_LOOKAHEAD_SEC), so visuals pulse with the speakers instead of
trailing them. `levels.beat` is the blend: predicted pulse when the tracker
is confident, reactive pulse otherwise.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from beat import HOP, AnalysisCache, BeatTracker, LoadedProfile, TrackProfile
from settings import (
    AUDIO_BLOCK_SIZE,
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
    status: str = "off"              # off | no-device | starting | ok | silent
    # --- predicted beat clock (evaluated with lookahead) ---
    bpm: float = 0.0
    beat_conf: float = 0.0
    beat_phase: float = 0.0          # 0 on the (predicted) beat
    beat: float = 0.0                # blended pulse: use THIS to drive visuals
    energy_ahead: float = 0.0        # cached profile's loudness ~1.5s ahead


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
        self._cache = AnalysisCache()
        self._beat_thread: Optional[threading.Thread] = None
        self._total_read = 0
        self._prev_tail = np.zeros(HOP, dtype=np.float32)
        self._stream_mono_off: Optional[float] = None   # mono_t - stream_t
        self._last_estimate_t = 0.0
        self._last_profile_t = 0.0
        self._last_save_t = 0.0

        self._track_lock = threading.Lock()
        self._sig = ""
        self._duration = 0
        self._profile: Optional[TrackProfile] = None
        self._loaded: Optional[LoadedProfile] = None
        self._pending_grid_seed = False
        self._pos_pair: Optional[Tuple[float, float]] = None  # (track_pos, mono_t)

    # ---------- lifecycle ----------

    def start(self) -> None:
        if not AUDIO_ENABLED or sd is None:
            self._status = "off"
            return
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="audio-analyzer", daemon=True)
        self._thread.start()
        if BEAT_ENABLED:
            self._beat_thread = threading.Thread(target=self._beat_run, name="beat-tracker", daemon=True)
            self._beat_thread.start()

    def stop(self) -> None:
        self._stop = True
        self._save_analysis(final=True)
        self._close_stream()

    # ---------- track context (called from the UI thread) ----------

    def set_track(self, sig: str, duration_sec: float) -> None:
        """New now-playing track: persist the previous track's analysis and
        warm-start from this track's cached profile if we have one."""
        with self._track_lock:
            if sig == self._sig:
                return
        self._save_analysis(final=True)
        with self._track_lock:
            self._sig = sig
            self._duration = int(max(0, duration_sec or 0))
            self._profile = TrackProfile(sig, self._duration) if self._duration > 0 else None
            self._loaded = self._cache.load(sig)
            self._pending_grid_seed = bool(
                self._loaded and self._loaded.bpm > 0 and self._loaded.beat_offset is not None
            )
            if self._loaded and self._loaded.bpm > 0:
                self._tracker.set_prior_bpm(self._loaded.bpm)

    def note_position(self, pos_sec: float) -> None:
        """UI feeds the current track position every frame (cheap)."""
        self._pos_pair = (float(pos_sec), time.monotonic())

    def _track_pos_now(self) -> Optional[float]:
        pair = self._pos_pair
        if pair is None:
            return None
        pos, stamp = pair
        age = time.monotonic() - stamp
        if age > 2.0:
            return None
        return pos + min(age, 0.1)

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

        for sr in (float(AUDIO_PREFERRED_SAMPLERATE), default_sr):
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
                stream.start()
                self._stream = stream
                self._samplerate = sr
                return True
            except Exception:
                continue
        return False

    def _run(self) -> None:
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
                self._status = "ok"
            else:
                self._status = "no-device"
                time.sleep(AUDIO_RECONNECT_SEC)

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        try:
            mono = indata.mean(axis=1) if indata.ndim == 2 else indata
            n = len(mono)
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
        except Exception:
            pass

    # ---------- beat thread ----------

    def _beat_run(self) -> None:
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
                self._maybe_seed_grid()

            if now - self._last_estimate_t >= 0.5:
                self._last_estimate_t = now
                try:
                    self._tracker.estimate()
                except Exception:
                    pass
            if now - self._last_profile_t >= 1.0:
                self._last_profile_t = now
                self._sample_profile()
            if now - self._last_save_t >= 10.0:
                self._last_save_t = now
                self._save_analysis(final=False)
            time.sleep(0.03)

    def _maybe_seed_grid(self) -> None:
        with self._track_lock:
            if not self._pending_grid_seed or self._loaded is None:
                return
            loaded = self._loaded
        if self._stream_mono_off is None or loaded.bpm <= 0 or loaded.beat_offset is None:
            return
        pos = self._track_pos_now()
        if pos is None:
            return
        period = 60.0 / loaded.bpm
        beat_track_t = loaded.beat_offset + math.floor((pos - loaded.beat_offset) / period) * period
        stream_now = time.monotonic() - self._stream_mono_off
        anchor_stream = stream_now - (pos - beat_track_t)
        self._tracker.seed_grid(period, anchor_stream, conf=0.4)
        with self._track_lock:
            self._pending_grid_seed = False

    def _sample_profile(self) -> None:
        if self._status != "ok" or self._sm_rms < 0.03:
            return
        pos = self._track_pos_now()
        if pos is None:
            return
        with self._track_lock:
            if self._profile is not None:
                self._profile.add_sample(pos, self._sm_bass, self._sm_mid, self._sm_high, self._sm_rms)

    def _beat_offset_in_track(self) -> Optional[float]:
        if self._stream_mono_off is None:
            return None
        pos = self._track_pos_now()
        if pos is None:
            return None
        q = self._tracker.query(time.monotonic() - self._stream_mono_off)
        if q is None or q.period <= 0:
            return None
        return float((pos - q.time_since_beat) % q.period)

    def _save_analysis(self, final: bool) -> None:
        with self._track_lock:
            sig, dur, profile = self._sig, self._duration, self._profile
        if not sig:
            return
        q = self._tracker.query(self._tracker.stream_time)
        if q is None or q.bpm <= 0:
            return
        worthy = q.conf >= 0.55 or (final and q.conf >= 0.35 and profile is not None
                                    and profile.seconds_recorded >= 20)
        if not worthy:
            return
        self._cache.save(
            sig, q.bpm, q.conf,
            beat_offset=self._beat_offset_in_track(),
            profile=profile, duration=dur,
        )

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
        with self._track_lock:
            loaded = self._loaded
        if loaded is not None and loaded.has_energy:
            pos = self._track_pos_now()
            if pos is not None:
                v = loaded.energy_at(pos + 1.5, "rms")
                if v is not None:
                    e_ahead = v
        return bpm, conf, phase, beat, e_ahead

    def snapshot(self) -> AudioLevels:
        now = time.monotonic()
        dt = max(1e-3, min(0.1, now - self._last_t))
        self._last_t = now

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

        return AudioLevels(
            ok=True,
            silent=silent,
            rms=self._sm_rms,
            bass=self._sm_bass,
            mid=self._sm_mid,
            high=self._sm_high,
            pulse=self._pulse,
            bands=[float(b) for b in self._sm_bands],
            status=self._status,
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
            loaded, profile = self._loaded, self._profile
        return {
            "env": env,
            "env_fps": env_fps,
            "bpm": q.bpm if q else 0.0,
            "conf": q.conf if q else 0.0,
            "period": q.period if q else 0.0,
            "next_in_ms": ((q.period - q.time_since_beat) * 1000.0) if q else 0.0,
            "beat_idx": (q.beat_index % 4) if q else 0,
            "lookahead_ms": BEAT_PREDICT_LOOKAHEAD_SEC * 1000.0,
            "cache": "hit" if loaded is not None else "none",
            "cache_bpm": loaded.bpm if loaded is not None else 0.0,
            "profile_cov": profile.coverage if profile is not None else 0.0,
            "profile_secs": profile.seconds_recorded if profile is not None else 0,
            "device": self._device_name,
            "sr": int(self._samplerate),
            "status": self._status,
            "clock_ok": self._stream_mono_off is not None,
        }
