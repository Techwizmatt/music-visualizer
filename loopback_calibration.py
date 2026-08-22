from __future__ import annotations

"""Audible end-to-end BlackHole/Multi-Output fidelity calibration.

Plays a safe-level left 440 Hz tone for five seconds followed by a right
880 Hz tone for five seconds. The normal application capture path records,
timeline-aligns, normalizes, tags, and encodes the loopback. A reference MP3
is encoded through the same finishing path, then both files are decoded and
compared. No manifest or analysis data is written.
"""

import argparse
import math
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd

import recording
from audio import AudioAnalyzer
from recording import (
    TrackIdentity,
    TrackPaths,
    _WriteSession,
    make_audio_signature,
    paths_for_track,
)


SAMPLERATE = 48000
DURATION = 10.0
TONE_SECONDS = 5.0
AMPLITUDE = 0.08
ROUTE_PREROLL_SECONDS = 0.50
TRACK_HANDOFF_LEAD_SECONDS = 0.008
TAIL_SECONDS = 0.50


def _device_index(name: str, *, input_device: bool) -> int:
    needle = name.casefold()
    channel_key = "max_input_channels" if input_device else "max_output_channels"
    matches = [
        index
        for index, device in enumerate(sd.query_devices())
        if needle in str(device.get("name", "")).casefold()
        and int(device.get(channel_key, 0)) >= 2
    ]
    if not matches:
        direction = "input" if input_device else "output"
        raise RuntimeError(f"No stereo {direction} device contains {name!r}")
    return matches[0]


def _test_signal() -> np.ndarray:
    frames = int(round(DURATION * SAMPLERATE))
    split = int(round(TONE_SECONDS * SAMPLERATE))
    output = np.zeros((frames, 2), dtype=np.float32)
    left_t = np.arange(split, dtype=np.float32) / SAMPLERATE
    right_t = np.arange(frames - split, dtype=np.float32) / SAMPLERATE
    output[:split, 0] = AMPLITUDE * np.sin(2.0 * np.pi * 440.0 * left_t)
    output[split:, 1] = AMPLITUDE * np.sin(2.0 * np.pi * 880.0 * right_t)

    # Short equal-power fades avoid speaker clicks without changing either
    # five-second channel assignment.
    fade_n = int(round(0.025 * SAMPLERATE))
    fade = np.sin(np.linspace(0.0, math.pi / 2.0, fade_n, dtype=np.float32)) ** 2
    output[:fade_n, 0] *= fade
    output[split - fade_n:split, 0] *= fade[::-1]
    output[split:split + fade_n, 1] *= fade
    output[-fade_n:, 1] *= fade[::-1]
    return output


def _write_reference(signal: np.ndarray, path: Path) -> None:
    track = TrackIdentity(
        sig=make_audio_signature("Loopback Reference", "MusicVisualizer", "Calibration", DURATION),
        title="Loopback Reference",
        artist="MusicVisualizer",
        album="Calibration",
        duration=DURATION,
        genre="Test Tone",
        source_app="MusicVisualizer Loopback Calibration",
    )
    paths = TrackPaths(base=path.with_suffix(""), audio=path)
    partial = paths.partial_audio(SAMPLERATE)
    for target in (path, partial):
        try:
            target.unlink()
        except FileNotFoundError:
            pass
    session = _WriteSession(track, paths, SAMPLERATE, 2)
    for start in range(0, len(signal), 1024):
        session.write(
            start / SAMPLERATE,
            signal[start:start + 1024],
            start / SAMPLERATE,
        )
    session.finalize()


def _decode_mp3(path: Path) -> np.ndarray:
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for loopback calibration")
    raw = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(path),
            "-map", "0:a:0",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "2",
            "-ar", str(SAMPLERATE),
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 2).copy()


def _aligned_views(
    reference: np.ndarray,
    captured: np.ndarray,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        count = min(len(reference), len(captured) - lag)
        return reference[:count], captured[lag:lag + count]
    count = min(len(reference) + lag, len(captured))
    return reference[-lag:-lag + count], captured[:count]


def _best_lag(reference: np.ndarray, captured: np.ndarray) -> int:
    # First align the left/right energy handoff at 1 ms resolution, then find
    # the sample-accurate waveform peak nearby.
    hop = 48
    count = min(len(reference), len(captured)) // hop
    ref_blocks = reference[:count * hop].reshape(count, hop, 2)
    cap_blocks = captured[:count * hop].reshape(count, hop, 2)
    ref_env = np.sqrt(np.mean(ref_blocks * ref_blocks, axis=1))
    cap_env = np.sqrt(np.mean(cap_blocks * cap_blocks, axis=1))
    ref_shape = (ref_env[:, 0] - ref_env[:, 1]) - np.mean(ref_env[:, 0] - ref_env[:, 1])
    cap_shape = (cap_env[:, 0] - cap_env[:, 1]) - np.mean(cap_env[:, 0] - cap_env[:, 1])
    correlation = np.correlate(cap_shape, ref_shape, mode="full")
    coarse = (int(np.argmax(correlation)) - (len(ref_shape) - 1)) * hop

    best_lag = coarse
    best_score = -1.0
    for lag in range(coarse - hop * 2, coarse + hop * 2 + 1):
        ref_view, cap_view = _aligned_views(reference, captured, lag)
        if len(ref_view) < SAMPLERATE:
            continue
        numerator = float(np.sum(ref_view * cap_view))
        denominator = math.sqrt(
            float(np.sum(ref_view * ref_view)) * float(np.sum(cap_view * cap_view))
        ) + 1e-20
        score = abs(numerator / denominator)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def _tone_amplitude(channel: np.ndarray, frequency: float) -> float:
    times = np.arange(len(channel), dtype=np.float64) / SAMPLERATE
    phase = 2.0 * np.pi * frequency * times
    sine = 2.0 * float(np.dot(channel, np.sin(phase))) / len(channel)
    cosine = 2.0 * float(np.dot(channel, np.cos(phase))) / len(channel)
    return math.hypot(sine, cosine)


def _compare(reference: np.ndarray, captured: np.ndarray) -> dict[str, float]:
    lag = _best_lag(reference, captured)
    ref_view, cap_view = _aligned_views(reference, captured, lag)
    trim = int(round(0.10 * SAMPLERATE))
    if len(ref_view) > trim * 2:
        ref_view = ref_view[trim:-trim]
        cap_view = cap_view[trim:-trim]

    correlations = []
    gains = []
    for channel in range(2):
        ref_ch = ref_view[:, channel].astype(np.float64)
        cap_ch = cap_view[:, channel].astype(np.float64)
        ref_ch -= np.mean(ref_ch)
        cap_ch -= np.mean(cap_ch)
        gain = float(np.dot(cap_ch, ref_ch) / (np.dot(ref_ch, ref_ch) + 1e-20))
        gains.append(gain)
        correlations.append(float(np.corrcoef(ref_ch, cap_ch)[0, 1]))

    split = min(int(round(TONE_SECONDS * SAMPLERATE)), len(cap_view))
    guard = int(round(0.10 * SAMPLERATE))
    left_part = cap_view[guard:max(guard + 1, split - guard)]
    right_part = cap_view[min(len(cap_view), split + guard):max(split + guard + 1, len(cap_view) - guard)]
    left_main = _tone_amplitude(left_part[:, 0], 440.0)
    left_leak = _tone_amplitude(left_part[:, 1], 440.0)
    right_main = _tone_amplitude(right_part[:, 1], 880.0)
    right_leak = _tone_amplitude(right_part[:, 0], 880.0)
    separation_db = min(
        20.0 * math.log10((left_main + 1e-20) / (left_leak + 1e-20)),
        20.0 * math.log10((right_main + 1e-20) / (right_leak + 1e-20)),
    )
    return {
        "match_percent": 100.0 * min(correlations),
        "left_correlation": correlations[0],
        "right_correlation": correlations[1],
        "timeline_offset_ms": 1000.0 * lag / SAMPLERATE,
        "left_gain": gains[0],
        "right_gain": gains[1],
        "channel_separation_db": separation_db,
    }


def run(output_name: str, input_name: str, destination: Path) -> int:
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    recording.AUDIO_RECORDING_DIR = str(destination)

    input_device = _device_index(input_name, input_device=True)
    output_device = _device_index(output_name, input_device=False)
    signal = _test_signal()
    title, artist, album = "BlackHole Loopback Test", "MusicVisualizer", "Calibration"
    signature = make_audio_signature(title, artist, album, DURATION)
    track = TrackIdentity(signature, title, artist, album, DURATION)
    captured_path = paths_for_track(track).audio
    reference_path = destination / "Loopback Calibration Reference.mp3"
    for target in (captured_path, track_path := paths_for_track(track).partial_audio(SAMPLERATE)):
        try:
            target.unlink()
        except FileNotFoundError:
            pass

    analyzer = AudioAnalyzer()
    final_debug: dict[str, object] = {}
    analyzer.start()
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if str(analyzer.debug_info().get("status")) in ("ok", "silent"):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("BlackHole did not become ready")
        if not analyzer._capture_ready.wait(2.0):
            raise RuntimeError("BlackHole opened but delivered no capture blocks")
        warm_frames = analyzer._total_written + SAMPLERATE // 2
        warm_deadline = time.monotonic() + 2.0
        while analyzer._total_written < warm_frames and time.monotonic() < warm_deadline:
            time.sleep(0.01)

        playback = np.vstack((
            np.zeros((int(round(ROUTE_PREROLL_SECONDS * SAMPLERATE)), 2), dtype=np.float32),
            signal,
            np.zeros((int(round(TAIL_SECONDS * SAMPLERATE)), 2), dtype=np.float32),
        ))
        cursor = 0
        total_frames = len(playback)
        dac_zero: Optional[float] = None
        finished = threading.Event()
        output_errors: list[str] = []

        def output_callback(outdata, frames, time_info, status) -> None:  # noqa: ANN001
            nonlocal cursor, dac_zero
            if status:
                output_errors.append(str(status))
            if dac_zero is None:
                dac_zero = float(time_info.outputBufferDacTime)
            outdata.fill(0.0)
            available = max(0, min(frames, len(playback) - cursor))
            if available:
                outdata[:available] = playback[cursor:cursor + available]
            cursor += frames
            if cursor >= total_frames:
                raise sd.CallbackStop

        print(
            f"Playing 440 Hz LEFT for 5 seconds, then 880 Hz RIGHT for 5 seconds "
            f"through {sd.query_devices(output_device)['name']}…",
            flush=True,
        )
        with sd.OutputStream(
            device=output_device,
            channels=2,
            samplerate=SAMPLERATE,
            blocksize=1024,
            dtype="float32",
            latency="low",
            callback=output_callback,
            finished_callback=finished.set,
        ) as stream:
            while dac_zero is None and not finished.wait(0.005):
                pass
            tone_zero = float(dac_zero or stream.time) + ROUTE_PREROLL_SECONDS
            while (
                float(stream.time) < tone_zero - TRACK_HANDOFF_LEAD_SECONDS
                and not finished.wait(0.001)
            ):
                pass
            analyzer.set_track(signature, DURATION, title, artist, album, genre="Test Tone")
            analyzer.note_position(max(0.0, float(stream.time) - tone_zero), True)
            while not finished.wait(0.02):
                position = max(0.0, min(DURATION, float(stream.time) - tone_zero))
                analyzer.note_position(position, True)
            analyzer.note_position(DURATION, True)
            time.sleep(0.15)
        analyzer.note_position(DURATION, False)
        if output_errors:
            raise RuntimeError(f"Output stream reported: {', '.join(output_errors)}")

        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            state = analyzer.debug_info()
            if captured_path.exists() and state.get("record_mode") == "file-ready":
                break
            if state.get("record_mode") in ("recording-error", "analysis-error"):
                raise RuntimeError(str(state))
            time.sleep(0.10)
        if not captured_path.exists():
            raise RuntimeError(f"Recorder did not finish {captured_path}")
        final_debug = analyzer.debug_info()
    finally:
        analyzer.stop()

    _write_reference(signal, reference_path)
    reference = _decode_mp3(reference_path)
    captured = _decode_mp3(captured_path)
    results = _compare(reference, captured)
    print(f"Reference: {reference_path}")
    print(f"Captured:  {captured_path}")
    print(f"Match: {results['match_percent']:.5f}%")
    print(f"Timeline offset: {results['timeline_offset_ms']:+.3f} ms")
    print(f"Channel separation: {results['channel_separation_db']:.1f} dB")
    print(f"Decoded gain L/R: {results['left_gain']:.6f} / {results['right_gain']:.6f}")
    capture_drops = int(final_debug.get("capture_dropped", 0) or 0)
    recorder_drops = int(final_debug.get("record_dropped", 0) or 0)
    input_overflows = int(final_debug.get("input_overflows", 0) or 0)
    print(
        f"Dropped capture/recorder blocks: {capture_drops}/{recorder_drops}; "
        f"input overruns: {input_overflows}"
    )
    passed = (
        results["match_percent"] >= 99.0
        and capture_drops == 0
        and recorder_drops == 0
        and input_overflows == 0
    )
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="Multi-Output Device")
    parser.add_argument("--input", default="BlackHole 2ch")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("audio_recordings") / ".diagnostics",
    )
    args = parser.parse_args()
    return run(args.output, args.input, args.destination)


if __name__ == "__main__":
    raise SystemExit(main())
