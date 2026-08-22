from __future__ import annotations

import tempfile
import threading
import time
import unittest
import json
import math
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np

from recording import (
    FullTrackProfile,
    TrackFileManager,
    TrackIdentity,
    _WriteSession,
    analyze_audio_file,
    make_audio_signature,
    paths_for_track,
)


def _click_track(duration: float, samplerate: int = 48000) -> np.ndarray:
    t = np.arange(int(duration * samplerate), dtype=np.float32) / samplerate
    mono = (0.06 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    for beat_t in np.arange(0.0, duration, 0.5):
        start = int(beat_t * samplerate)
        count = min(int(0.045 * samplerate), len(mono) - start)
        if count > 0:
            mono[start:start + count] += (
                0.8 * np.exp(-np.arange(count) / (0.008 * samplerate))
            ).astype(np.float32)
    return np.column_stack((mono, mono))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is required")
class RecordingTests(unittest.TestCase):
    def test_recording_wait_states_identify_missing_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as root, patch("recording.AUDIO_RECORDING_DIR", root):
            manager = TrackFileManager(lambda _sig, _profile: None)
            track = TrackIdentity("State||||||10", "State", "", "", 10.0)
            try:
                manager.set_track(track)
                self.assertEqual(manager.status()["mode"], "waiting-for-position")
                manager.note_position(0.25, True)
                self.assertEqual(manager.status()["mode"], "waiting-for-audio")
                manager.note_position(0.25, False)
                self.assertEqual(manager.status()["mode"], "paused")
            finally:
                manager.stop()

    def test_filename_uses_complete_identity_and_duration(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            a = TrackIdentity("A|||Artist|||Album|||10", "A", "Artist", "Album", 10.0)
            b = TrackIdentity("A|||Artist|||Album|||11", "A", "Artist", "Album", 11.0)
            pa = paths_for_track(a, root)
            pb = paths_for_track(b, root)
            self.assertNotEqual(pa.audio, pb.audio)
            self.assertIn("Artist - A - Album [10s]", pa.audio.name)
            self.assertEqual(pa.audio.suffix, ".mp3")
            self.assertNotEqual(
                make_audio_signature("A", "Artist", "Album", 10.001),
                make_audio_signature("A", "Artist", "Album", 10.002),
            )

    def test_timeline_mp3_metadata_size_and_whole_file_profile(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            samplerate, duration = 48000, 8.0
            track = TrackIdentity(
                "Project|||GarageBand|||Demo|||8",
                "Project",
                "GarageBand",
                "Demo",
                duration,
                genre="Electronic",
                track_number=3,
                total_track_count=9,
                source_app="com.apple.garageband10",
                content_identifier="project-123",
            )
            paths = paths_for_track(track, root)
            audio = _click_track(duration, samplerate)
            session = _WriteSession(track, paths, samplerate, 2)
            complete = False
            for start in range(0, len(audio), 1024):
                block = audio[start:start + 1024]
                complete = session.write(start / samplerate, block, start / samplerate) or complete
            self.assertTrue(complete)
            artwork = subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-f", "lavfi",
                    "-i", "color=c=blue:s=16x16", "-frames:v", "1",
                    "-f", "image2pipe", "-vcodec", "png", "pipe:1",
                ],
                check=True, capture_output=True,
            ).stdout
            session.finalize(artwork)
            self.assertLess(paths.audio.stat().st_size, len(audio) * 2 * 4 // 4)
            self.assertEqual([p.name for p in Path(root).glob("*")], [paths.audio.name])

            probe = subprocess.run(
                [
                    shutil.which("ffprobe") or "ffprobe", "-v", "error",
                    "-show_entries", "format=bit_rate:format_tags:stream=codec_type:stream_disposition=attached_pic",
                    "-of", "json", str(paths.audio),
                ],
                check=True, capture_output=True, text=True,
            )
            info = json.loads(probe.stdout)["format"]
            tags = {str(k).lower(): str(v) for k, v in info.get("tags", {}).items()}
            self.assertEqual(tags.get("title"), "Project")
            self.assertEqual(tags.get("artist"), "GarageBand")
            self.assertEqual(tags.get("album"), "Demo")
            self.assertEqual(tags.get("genre"), "Electronic")
            self.assertEqual(tags.get("track"), "3/9")
            self.assertEqual(tags.get("now_playing_duration_ms"), "8000")
            self.assertEqual(tags.get("now_playing_source_app"), "com.apple.garageband10")
            self.assertEqual(tags.get("now_playing_content_identifier"), "project-123")
            self.assertEqual(tags.get("recording_format_version"), "2")
            self.assertGreaterEqual(int(info["bit_rate"]), 315000)
            streams = json.loads(probe.stdout).get("streams", [])
            self.assertTrue(any(s.get("disposition", {}).get("attached_pic") == 1 for s in streams))

            profile = analyze_audio_file(str(paths.audio), track.sig, duration, 40.0)
            self.assertAlmostEqual(profile.bpm, 120.0, delta=5.0)
            frame = profile.sample(4.0)
            self.assertIsNotNone(frame)
            self.assertEqual(len(frame or ()), 105)
            self.assertTrue(all(-1.0001 <= v <= 1.0001 for v in (frame or ())[36:100]))
            self.assertEqual([p.name for p in Path(root).glob("*")], [paths.audio.name])

    def test_full_profile_detects_richer_mix_features_and_sections(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            samplerate, duration = 48000, 10.0
            track = TrackIdentity("Sections|||Artist|||Album|||10", "Sections", "Artist", "Album", duration)
            paths = paths_for_track(track, root)
            t = np.arange(int(duration * samplerate), dtype=np.float32) / samplerate
            left = np.zeros_like(t)
            right = np.zeros_like(t)
            first = t < 3.0
            middle = (t >= 3.0) & (t < 6.0)
            last = t >= 6.0
            left[first] = 0.30 * np.sin(2.0 * np.pi * 95.0 * t[first])
            right[first] = left[first]
            left[middle] = 0.22 * np.sin(2.0 * np.pi * 5200.0 * t[middle])
            right[middle] = 0.22 * np.sin(2.0 * np.pi * 3700.0 * t[middle] + 1.3)
            vocal_like = (
                0.22 * np.sin(2.0 * np.pi * 260.0 * t[last])
                + 0.13 * np.sin(2.0 * np.pi * 520.0 * t[last])
                + 0.08 * np.sin(2.0 * np.pi * 780.0 * t[last])
            )
            left[last] = vocal_like
            right[last] = vocal_like
            audio = np.column_stack((left, right)).astype(np.float32)
            session = _WriteSession(track, paths, samplerate, 2)
            for start in range(0, len(audio), 1024):
                session.write(start / samplerate, audio[start:start + 1024], start / samplerate)
            session.finalize()

            profile = analyze_audio_file(str(paths.audio), track.sig, duration, 30.0)
            frames = [profile.sample(i / 30.0) for i in range(int(duration * 30))]
            matrix = np.asarray([row for row in frames if row is not None])
            self.assertEqual(matrix.shape[1], 105)
            self.assertGreater(float(np.max(matrix[:, 6])), 0.30)   # vocal presence
            self.assertGreater(float(np.max(matrix[:, 9])), 0.20)   # stereo width
            self.assertGreater(float(np.max(matrix[:, 11])), 0.10)  # section boundary
            self.assertGreater(float(np.max(np.abs(matrix[:, 36:100]))), 0.50)
            self.assertGreater(float(np.max(matrix[:, 100])), 0.70)  # relative intensity
            self.assertGreater(float(np.max(matrix[:, 102])), 0.08)  # pre-jump anticipation
            self.assertGreater(float(np.max(matrix[:, 103])), 0.10)  # jump/drop impact
            self.assertGreater(float(np.max(matrix[:, 104])), 0.45)  # relative calmness
            bass_section = profile.sample(1.5)
            wide_section = profile.sample(4.5)
            vocal_section = profile.sample(8.0)
            assert bass_section is not None and wide_section is not None and vocal_section is not None
            self.assertGreater(wide_section[7], bass_section[7] + 0.20)   # brightness follows time
            self.assertGreater(wide_section[9], bass_section[9] + 0.35)   # stereo width follows time
            self.assertGreater(vocal_section[6], bass_section[6] + 0.35)  # vocal cue follows time

    def test_quiet_float_capture_is_normalized_before_mp3_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            samplerate, duration = 48000, 2.0
            track = TrackIdentity("Quiet||||||2", "Quiet", "", "", duration)
            paths = paths_for_track(track, root)
            t = np.arange(int(duration * samplerate), dtype=np.float32) / samplerate
            mono = (0.004 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
            audio = np.column_stack((mono, mono))
            session = _WriteSession(track, paths, samplerate, 2)
            for start in range(0, len(audio), 1024):
                session.write(start / samplerate, audio[start:start + 1024], start / samplerate)
            session.finalize()

            decoded = subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-i", str(paths.audio),
                    "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "48000", "pipe:1",
                ],
                check=True,
                capture_output=True,
            ).stdout
            values = np.frombuffer(decoded, dtype="<f4")
            self.assertGreater(float(np.max(np.abs(values))), 0.72)
            self.assertLess(float(np.max(np.abs(values))), 1.0)

            probe = subprocess.run(
                [
                    shutil.which("ffprobe") or "ffprobe", "-v", "error",
                    "-show_entries", "format_tags=recording_source_peak_dbfs,recording_normalization_gain_db",
                    "-of", "json", str(paths.audio),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            tags = json.loads(probe.stdout)["format"]["tags"]
            self.assertLess(float(tags["recording_source_peak_dbfs"]), -45.0)
            self.assertGreater(float(tags["recording_normalization_gain_db"]), 40.0)

    def test_quiet_stereo_capture_keeps_tone_fidelity_and_channel_separation(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            samplerate, duration = 48000, 2.0
            track = TrackIdentity("Fidelity||||||2", "Fidelity", "", "", duration)
            paths = paths_for_track(track, root)
            t = np.arange(int(duration * samplerate), dtype=np.float32) / samplerate
            audio = np.column_stack((
                0.004 * np.sin(2.0 * np.pi * 1000.0 * t),
                0.003 * np.sin(2.0 * np.pi * 2000.0 * t),
            )).astype(np.float32)
            session = _WriteSession(track, paths, samplerate, 2)
            for start in range(0, len(audio), 1024):
                session.write(start / samplerate, audio[start:start + 1024], start / samplerate)
            session.finalize()

            decoded = subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-i", str(paths.audio),
                    "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", "48000", "pipe:1",
                ],
                check=True,
                capture_output=True,
            ).stdout
            values = np.frombuffer(decoded, dtype="<f4").reshape(-1, 2)
            middle = values[int(0.25 * samplerate):int(1.75 * samplerate)]
            times = np.arange(len(middle), dtype=np.float64) / samplerate

            def tone_amplitude(channel: np.ndarray, frequency: float) -> float:
                phase = 2.0 * np.pi * frequency * times
                sine = 2.0 * float(np.dot(channel, np.sin(phase))) / len(channel)
                cosine = 2.0 * float(np.dot(channel, np.cos(phase))) / len(channel)
                return math.hypot(sine, cosine)

            left_1k = tone_amplitude(middle[:, 0], 1000.0)
            left_2k = tone_amplitude(middle[:, 0], 2000.0)
            right_1k = tone_amplitude(middle[:, 1], 1000.0)
            right_2k = tone_amplitude(middle[:, 1], 2000.0)
            self.assertGreater(left_1k, 0.55)
            self.assertGreater(right_2k, 0.40)
            self.assertGreater(left_1k, left_2k * 100.0)
            self.assertGreater(right_2k, right_1k * 100.0)

    def test_track_change_closes_old_partial_before_new_audio(self) -> None:
        with tempfile.TemporaryDirectory() as root, patch("recording.AUDIO_RECORDING_DIR", root):
            manager = TrackFileManager(lambda _sig, _profile: None)
            manager.start()
            a = TrackIdentity("A||||||10", "A", "", "", 10.0)
            b = TrackIdentity("B||||||10", "B", "", "", 10.0)
            block = np.full((1024, 2), 0.2, dtype=np.float32)
            try:
                manager.set_track(a)
                manager.note_position(1.0, True)
                manager.push_audio(block, 48000, time.monotonic())
                manager.set_track(b)
                manager.note_position(0.0, True)
                manager.push_audio(block, 48000, time.monotonic())
                time.sleep(0.25)
            finally:
                manager.stop()
            pa, pb = paths_for_track(a, root), paths_for_track(b, root)
            self.assertTrue(pa.partial_audio(48000).exists())
            self.assertTrue(pb.partial_audio(48000).exists())
            self.assertNotEqual(pa.partial_audio(48000), pb.partial_audio(48000))
            self.assertEqual(list(Path(root).rglob("*.json")), [])

    def test_media_timestamp_jitter_does_not_fragment_audio(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            samplerate, duration = 48000, 4.0
            track = TrackIdentity("Jitter||||||4000ms", "Jitter", "", "", duration)
            paths = paths_for_track(track, root)
            t = np.arange(int(duration * samplerate), dtype=np.float32) / samplerate
            mono = (0.25 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
            audio = np.column_stack((mono, mono))
            session = _WriteSession(track, paths, samplerate, 2)
            for block_i, start in enumerate(range(0, len(audio), 1024)):
                jitter = 0.0 if block_i == 0 else 0.18 * np.sin(block_i * 1.7)
                session.write(
                    start / samplerate + float(jitter),
                    audio[start:start + 1024],
                    start / samplerate,
                )
            session.finalize()
            decoded = subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-i", str(paths.audio),
                    "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "48000", "pipe:1",
                ],
                check=True, capture_output=True,
            ).stdout
            samples = np.frombuffer(decoded, dtype="<f4")
            middle = samples[int(0.15 * samplerate): -int(0.15 * samplerate)]
            self.assertLess(float(np.mean(np.abs(middle) < 1e-7)), 0.001)
            self.assertLess(float(np.percentile(np.abs(np.diff(middle)), 99.9)), 0.08)

    def test_existing_audio_is_analyzed_asynchronously(self) -> None:
        with tempfile.TemporaryDirectory() as root, patch("recording.AUDIO_RECORDING_DIR", root):
            samplerate, duration = 48000, 8.0
            track = TrackIdentity("Async||||||8", "Async", "", "", duration)
            paths = paths_for_track(track, root)
            audio = _click_track(duration, samplerate)
            session = _WriteSession(track, paths, samplerate, 2)
            for start in range(0, len(audio), 1024):
                block = audio[start:start + 1024]
                session.write(start / samplerate, block, start / samplerate)
            session.finalize()

            ready = threading.Event()
            received = []

            def on_profile(sig: str, profile: FullTrackProfile) -> None:
                received.append((sig, profile))
                ready.set()

            manager = TrackFileManager(on_profile)
            manager.start()
            try:
                manager.set_track(track)
                self.assertTrue(ready.wait(15.0), manager.status())
            finally:
                manager.stop()
            self.assertEqual(received[0][0], track.sig)
            self.assertAlmostEqual(received[0][1].bpm, 120.0, delta=5.0)


if __name__ == "__main__":
    unittest.main()
