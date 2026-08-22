from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from audio import AudioAnalyzer


class AudioCaptureTests(unittest.TestCase):
    def test_realtime_callback_only_copies_to_priority_worker(self) -> None:
        analyzer = AudioAnalyzer()
        received = threading.Event()
        captured = []

        def take_owned(samples, samplerate, packet_mono) -> None:
            captured.append((samples, samplerate, packet_mono))
            received.set()

        with (
            patch.object(analyzer._files, "push_audio_owned", side_effect=take_owned),
            patch("audio.time.monotonic", return_value=200.0),
        ):
            analyzer._capture_thread = threading.Thread(
                target=analyzer._capture_run,
                daemon=True,
            )
            analyzer._capture_thread.start()
            source = np.full((1024, 2), 0.125, dtype=np.float32)
            analyzer._samplerate = 48000.0
            analyzer._callback(
                source,
                len(source),
                SimpleNamespace(currentTime=100.0, inputBufferAdcTime=99.95),
                SimpleNamespace(input_overflow=True),
            )
            source[:] = 0.0
            self.assertTrue(received.wait(1.0))
            self.assertTrue(np.all(captured[0][0] == np.float32(0.125)))
            self.assertEqual(captured[0][1], 48000.0)
            expected_end = 200.0 - (100.0 - (99.95 + len(source) / 48000.0))
            self.assertAlmostEqual(captured[0][2], expected_end, places=6)
            self.assertEqual(analyzer._input_overflows, 1)
            self.assertEqual(analyzer._capture_queue_drops, 0)
            self.assertTrue(analyzer._capture_ready.is_set())
            self.assertGreater(analyzer._total_written, 0)
        analyzer.stop()


if __name__ == "__main__":
    unittest.main()
