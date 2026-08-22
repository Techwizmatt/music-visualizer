from __future__ import annotations

import math
import time
import unittest
from unittest.mock import patch

import numpy as np

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter

from audio import AudioLevels
from settings import VIS_FOREGROUND_RENDER_MAX_PX
from visualizer import AsyncSphereVisualizer, SphereVisualizer


class VisualizerModeTests(unittest.TestCase):
    @staticmethod
    def _full_levels(**overrides) -> AudioLevels:
        values = dict(
            ok=True,
            silent=False,
            source="full-file",
            rms=0.64,
            bass=0.72,
            mid=0.58,
            high=0.48,
            beat=0.81,
            bands=[(i + 1) / 24.0 for i in range(24)],
            waveform=[0.8 * math.sin(i * 0.42) for i in range(64)],
            vocal=0.68,
            brightness=0.57,
            spectral_flux=0.73,
            stereo_width=0.62,
            section=0.5,
            section_change=0.77,
            energy_ahead=0.8,
            music_motion=0.72,
            energy_flow=0.48,
            spectral_shift=-0.31,
            climax=0.84,
            track_intensity=0.76,
            buildup=0.58,
            anticipation=0.66,
            drop=0.82,
            calmness=0.12,
        )
        values.update(overrides)
        return AudioLevels(**values)

    def test_all_nine_artwork_modes_render_with_full_file_detail(self) -> None:
        artwork = QImage(64, 64, QImage.Format_RGB32)
        for y in range(64):
            for x in range(64):
                artwork.setPixelColor(x, y, QColor(30 + x * 3, 50 + y * 2, 220 - x * 2))

        vis = SphereVisualizer(n_dots=500)
        vis.set_artwork(artwork)
        vis.set_mode(True)
        levels = self._full_levels()
        canvas = QImage(600, 600, QImage.Format_ARGB32_Premultiplied)
        now = 1.0
        for style in range(1, 10):
            vis.set_style(style)
            self.assertEqual(vis.style(), style)
            self.assertTrue(vis.style_name())
            for _ in range(6):
                canvas.fill(QColor(0, 0, 0, 255))
                painter = QPainter(canvas)
                vis.render(painter, QRectF(40, 40, 520, 520), now, levels, QRectF(180, 180, 240, 240))
                painter.end()
                now += 0.1

        self.assertGreaterEqual(len(vis._relief_uv), 7_500)
        colored = sum(
            canvas.pixelColor(x, y) != QColor(0, 0, 0, 255)
            for y in range(0, canvas.height(), 4)
            for x in range(0, canvas.width(), 4)
        )
        self.assertGreater(colored, 20)

    def test_component_router_keeps_musical_roles_independent(self) -> None:
        channels = SphereVisualizer._music_channels(self._full_levels(
            bass=0.91,
            mid=0.12,
            vocal=0.73,
            high=0.21,
            spectral_flux=0.84,
            pulse=0.66,
            stereo_width=0.37,
            buildup=0.52,
            anticipation=0.78,
            drop=0.95,
            section_change=0.20,
            climax=0.61,
        ))
        self.assertEqual(channels.shape, (8,))
        self.assertAlmostEqual(channels[0], 0.91)
        self.assertAlmostEqual(channels[2], 0.73)
        self.assertGreater(channels[3], channels[4])  # snare proxy vs steady highs
        self.assertAlmostEqual(channels[5], 0.37)
        self.assertAlmostEqual(channels[6], 0.78)
        self.assertAlmostEqual(channels[7], 0.95)

    def test_artwork_vortex_segments_regions_and_reveals_full_art_at_peaks(self) -> None:
        artwork = QImage(96, 96, QImage.Format_RGB32)
        artwork.fill(QColor(205, 205, 198))
        for y in range(16, 43):
            for x in range(26, 70):
                artwork.setPixelColor(x, y, QColor(220, 34 + y, 64 + x))
        for y in range(57, 84):
            for x in range(31, 65):
                artwork.setPixelColor(x, y, QColor(25 + x, 55, 225 - y))

        vis = SphereVisualizer(n_dots=500)
        vis.set_artwork(artwork)
        uv = vis._relief_uv
        upper = (uv[:, 0] > 0.28) & (uv[:, 0] < 0.72) & (uv[:, 1] > 0.16) & (uv[:, 1] < 0.45)
        lower = (uv[:, 0] > 0.32) & (uv[:, 0] < 0.68) & (uv[:, 1] > 0.58) & (uv[:, 1] < 0.86)
        upper_region = int(np.bincount(vis._relief_region[upper]).argmax())
        lower_region = int(np.bincount(vis._relief_region[lower]).argmax())
        background = (uv[:, 0] < 0.16) & (uv[:, 1] < 0.16)
        background_region = int(np.bincount(vis._relief_region[background]).argmax())
        self.assertNotEqual(upper_region, lower_region)
        self.assertNotIn(background_region, (upper_region, lower_region))
        self.assertEqual(len(vis._relief_region_indices), vis._relief_regions)
        self.assertEqual(set(vis._relief_foreground_regions), {upper_region, lower_region})
        self.assertIsNotNone(vis._artwork_square)
        self.assertFalse(hasattr(vis, "_relief_region_images"))

        slow = self._full_levels(
            rms=0.08,
            music_motion=0.05,
            track_intensity=0.04,
            climax=0.0,
            drop=0.0,
            calmness=1.0,
        )
        vis._sm_motion = 0.05
        vis._sm_intensity = 0.04
        vis._sm_climax = 0.0
        vis._sm_drop = 0.0
        vis._sm_vortex_energy = 0.03
        vis._vortex_dissolve = 0.12
        slow_visibility = vis._artwork_vortex_visibility()
        self.assertGreater(float(np.mean(slow_visibility)), 0.001)
        self.assertLess(float(np.mean(slow_visibility)), 0.55)

        peak = self._full_levels(
            music_motion=1.0,
            track_intensity=1.0,
            climax=1.0,
            drop=1.0,
            calmness=0.0,
        )
        vis._sm_motion = 1.0
        vis._sm_intensity = 1.0
        vis._sm_climax = 1.0
        vis._sm_drop = 1.0
        vis._sm_vortex_energy = 1.0
        vis._vortex_dissolve = 1.0
        peak_visibility = vis._artwork_vortex_visibility()
        self.assertGreater(float(np.min(peak_visibility)), 0.99)
        self.assertGreater(float(np.mean(peak_visibility)), float(np.mean(slow_visibility)) + 0.45)

        vis._morph = 1.0
        vis._ang = 0.0
        first = np.column_stack(vis._artwork_vortex_geometry())
        vis._ang = 0.8
        rotated = np.column_stack(vis._artwork_vortex_geometry())
        self.assertGreater(float(np.mean(np.linalg.norm(first - rotated, axis=1))), 0.02)
        for angle in np.linspace(0.0, math.tau, 9):
            vis._ang = float(angle)
            x, y, _ = vis._artwork_vortex_geometry()
            aspect = float(np.ptp(x) / max(1e-9, np.ptp(y)))
            self.assertGreater(aspect, 0.62)
            self.assertLess(aspect, 1.55)
        vis._view_yaw = 50.0
        vis._view_pitch = -50.0
        turn, tilt = vis._artwork_vortex_camera_angles(1.0, 1.0)
        self.assertLessEqual(abs(math.degrees(turn)), 15.0001)
        self.assertLessEqual(abs(math.degrees(tilt)), 15.0001)

    def test_artwork_vortex_music_handoff_has_no_frame_jumps(self) -> None:
        artwork = QImage(96, 96, QImage.Format_RGB32)
        artwork.fill(QColor(205, 205, 198))
        for y in range(22, 75):
            for x in range(28, 68):
                artwork.setPixelColor(x, y, QColor(40 + x * 2, 45 + y, 220 - y))

        vis = SphereVisualizer(n_dots=500)
        vis.set_artwork(artwork)
        vis.set_mode(True)
        vis.set_style(8)
        canvas = QImage(420, 420, QImage.Format_ARGB32_Premultiplied)
        peak = self._full_levels(
            music_motion=1.0,
            track_intensity=1.0,
            climax=1.0,
            drop=1.0,
            calmness=0.0,
        )
        now = 0.0
        rising = []
        geometry_frames = []
        for _ in range(180):
            painter = QPainter(canvas)
            vis.render(painter, QRectF(30, 30, 360, 360), now, peak)
            painter.end()
            rising.append(vis._vortex_dissolve)
            geometry_frames.append(
                np.column_stack(vis._artwork_vortex_geometry())[::128]
            )
            now += 1.0 / 60.0
        rise_steps = np.diff(rising)
        self.assertGreater(rising[-1], 0.78)
        self.assertGreaterEqual(float(np.min(rise_steps)), -1e-9)
        self.assertLess(float(np.max(rise_steps)), 0.035)
        geometry_steps = [
            float(np.mean(np.linalg.norm(current - previous, axis=1)))
            for previous, current in zip(geometry_frames, geometry_frames[1:])
        ]
        self.assertLess(max(geometry_steps), 0.035)

        calm = self._full_levels(
            rms=0.03,
            music_motion=0.0,
            track_intensity=0.0,
            buildup=0.0,
            anticipation=0.0,
            climax=0.0,
            drop=0.0,
            calmness=1.0,
        )
        falling = []
        for _ in range(240):
            painter = QPainter(canvas)
            vis.render(painter, QRectF(30, 30, 360, 360), now, calm)
            painter.end()
            falling.append(vis._vortex_dissolve)
            now += 1.0 / 60.0
        fall_steps = np.diff(falling)
        self.assertLess(falling[-1], falling[0] - 0.25)
        # The smoothed energy envelope can finish its rise for a few frames
        # after the source turns calm; it must still remain imperceptibly small.
        self.assertLess(float(np.max(fall_steps)), 0.005)
        self.assertLess(float(np.max(np.abs(fall_steps))), 0.025)

    def test_dense_artwork_modes_do_not_sort_points_each_frame(self) -> None:
        vis = SphereVisualizer(n_dots=500)
        vis.set_artwork(QImage(64, 64, QImage.Format_RGB32))
        vis.set_mode(True)
        vis._morph = 1.0
        vis._style_fx = 1.0
        levels = self._full_levels()
        canvas = QImage(420, 420, QImage.Format_ARGB32_Premultiplied)
        for style in (8, 9):
            vis.set_style(style)
            vis._morph = 1.0
            vis._style_fx = 1.0
            painter = QPainter(canvas)
            with patch("visualizer.np.argsort", side_effect=AssertionError("per-frame sort")):
                vis.render(painter, QRectF(30, 30, 360, 360), 1.0 + style / 60.0, levels)
            painter.end()

    def test_sphere_full_file_implodes_before_larger_drop_release(self) -> None:
        vis = SphereVisualizer(n_dots=500)
        vis.set_style(1)
        levels = self._full_levels(vocal=0.0, section_change=0.0)

        vis._sm_anticipation = 1.0
        vis._sm_flow = -0.8
        vis._sm_drop = 0.0
        vis._sm_climax = 0.0
        imploded = vis._target_geometry(1.0, levels)
        self.assertLess(float(np.mean(np.linalg.norm(imploded, axis=1))), 0.52)

        vis._sm_anticipation = 0.0
        vis._sm_flow = 0.6
        vis._sm_drop = 1.0
        vis._sm_climax = 1.0
        exploded = vis._target_geometry(1.1, levels)
        self.assertGreater(
            float(np.mean(np.linalg.norm(exploded, axis=1))),
            float(np.mean(np.linalg.norm(imploded, axis=1))) * 1.55,
        )

    def test_drag_orbit_keeps_audio_motion_running_and_releases_with_inertia(self) -> None:
        vis = SphereVisualizer(n_dots=400)
        vis.set_mode(True)
        levels = self._full_levels()
        canvas = QImage(420, 420, QImage.Format_ARGB32_Premultiplied)
        now = 0.0
        for _ in range(70):
            painter = QPainter(canvas)
            vis.render(painter, QRectF(30, 30, 360, 360), now, levels)
            painter.end()
            now += 1.0 / 60.0

        phase_before = vis._flow_phase
        vis.begin_drag()
        vis.drag_view(54.0, -28.0, 1.0 / 60.0)
        dragged_yaw = vis._view_yaw
        dragged_pitch = vis._view_pitch
        painter = QPainter(canvas)
        vis.render(painter, QRectF(30, 30, 360, 360), now, levels)
        painter.end()
        now += 1.0 / 60.0
        self.assertTrue(vis.is_dragging())
        self.assertNotEqual(dragged_yaw, 0.0)
        self.assertNotEqual(dragged_pitch, 0.0)
        self.assertGreater(vis._flow_phase, phase_before)

        vis.end_drag()
        released_yaw = vis._view_yaw
        initial_velocity = abs(vis._view_yaw_velocity)
        for style in range(1, 10):
            vis.set_style(style)
            painter = QPainter(canvas)
            vis.render(painter, QRectF(30, 30, 360, 360), now, levels)
            painter.end()
            now += 1.0 / 60.0
        self.assertFalse(vis.is_dragging())
        self.assertNotEqual(vis._view_yaw, released_yaw)
        self.assertLess(abs(vis._view_yaw_velocity), initial_velocity)

    def test_async_visual_worker_keeps_only_latest_rendered_frame(self) -> None:
        vis = AsyncSphereVisualizer(n_dots=400)
        vis.set_mode(True)
        levels = self._full_levels()
        canvas = QImage(360, 360, QImage.Format_ARGB32_Premultiplied)
        try:
            for frame in range(70):
                canvas.fill(QColor(0, 0, 0, 255))
                painter = QPainter(canvas)
                vis.render(painter, QRectF(20, 20, 320, 320), frame / 60.0, levels)
                painter.end()
                time.sleep(0.003)
            deadline = time.monotonic() + 2.0
            while vis.morph_value() < 0.5 and time.monotonic() < deadline:
                painter = QPainter(canvas)
                vis.render(painter, QRectF(20, 20, 320, 320), 1.5, levels)
                painter.end()
                time.sleep(0.01)
            canvas.fill(QColor(0, 0, 0, 255))
            painter = QPainter(canvas)
            vis.render(painter, QRectF(20, 20, 320, 320), 1.6, levels)
            painter.end()
            self.assertGreater(vis.morph_value(), 0.5)
            colored = sum(
                canvas.pixelColor(x, y) != QColor(0, 0, 0, 255)
                for y in range(20, 340, 8)
                for x in range(20, 340, 8)
            )
            self.assertGreater(colored, 5)
        finally:
            vis.stop()

    def test_async_visual_worker_caps_retina_buffer_size(self) -> None:
        vis = AsyncSphereVisualizer(n_dots=400)
        vis.set_mode(True)
        vis.set_style(8)
        canvas = QImage(1000, 1000, QImage.Format_ARGB32_Premultiplied)
        canvas.setDevicePixelRatio(2.0)
        try:
            for frame in range(18):
                painter = QPainter(canvas)
                vis.render(
                    painter,
                    QRectF(0, 0, 500, 500),
                    frame / 60.0,
                    self._full_levels(),
                    QRectF(135, 135, 230, 230),
                )
                painter.end()
                time.sleep(0.006)
            deadline = time.monotonic() + 2.0
            image = None
            while time.monotonic() < deadline:
                with vis._latest_lock:
                    image = vis._latest_image
                if image is not None and not image.isNull():
                    break
                time.sleep(0.01)
            self.assertIsNotNone(image)
            assert image is not None
            self.assertLessEqual(max(image.width(), image.height()), VIS_FOREGROUND_RENDER_MAX_PX)
        finally:
            vis.stop()

    def test_full_file_release_reverses_animation_flow(self) -> None:
        vis = SphereVisualizer(n_dots=400)
        vis.set_style(5)
        vis.set_mode(True)
        levels = self._full_levels(
            music_motion=1.0,
            energy_flow=-1.0,
            spectral_shift=-0.8,
            climax=0.15,
        )
        canvas = QImage(360, 360, QImage.Format_ARGB32_Premultiplied)
        phase_mid = 0.0
        for frame in range(150):
            painter = QPainter(canvas)
            vis.render(painter, QRectF(20, 20, 320, 320), frame / 60.0, levels)
            painter.end()
            if frame == 80:
                phase_mid = vis._flow_phase
        self.assertLess(vis._flow_phase, phase_mid - 0.20)

    def test_crt_steady_state_skips_particle_geometry(self) -> None:
        vis = SphereVisualizer(n_dots=400)
        vis.set_style(7)
        vis.set_mode(True)
        levels = self._full_levels()
        canvas = QImage(360, 360, QImage.Format_ARGB32_Premultiplied)
        for frame in range(70):
            painter = QPainter(canvas)
            vis.render(painter, QRectF(20, 20, 320, 320), frame / 60.0, levels)
            painter.end()
        vis._target_geometry = lambda *_args: (_ for _ in ()).throw(AssertionError("particle path used"))
        painter = QPainter(canvas)
        vis.render(painter, QRectF(20, 20, 320, 320), 71 / 60.0, levels)
        painter.end()

    def test_crt_uses_past_and_future_traces_only_for_full_file(self) -> None:
        vis = SphereVisualizer(n_dots=400)
        vis.set_style(7)
        vis.set_mode(True)
        timeline = [
            [0.75 * math.sin(i * 0.35 + row * 0.24) for i in range(64)]
            for row in range(25)
        ]
        full = self._full_levels(
            waveform_timeline=timeline,
            waveform_timeline_center=12,
        )
        canvas = QImage(360, 360, QImage.Format_ARGB32_Premultiplied)
        for frame in range(8):
            painter = QPainter(canvas)
            vis.render(painter, QRectF(20, 20, 320, 320), frame / 60.0, full)
            painter.end()
        self.assertTrue(vis._wave_history_timeline)
        self.assertEqual(vis._wave_timeline_center, 12)
        self.assertGreater(
            float(np.mean(np.abs(vis._wave_history[2] - vis._wave_history[-3]))),
            0.05,
        )

        live = self._full_levels(
            source="live",
            waveform_timeline=[],
            waveform_timeline_center=0,
        )
        painter = QPainter(canvas)
        vis.render(painter, QRectF(20, 20, 320, 320), 9 / 60.0, live)
        painter.end()
        self.assertFalse(vis._wave_history_timeline)

    def test_structure_build_and_drop_accelerate_without_phase_cut(self) -> None:
        vis = SphereVisualizer(n_dots=400)
        vis.set_style(4)
        vis.set_mode(True)
        slow = self._full_levels(
            rms=0.12,
            beat=0.0,
            music_motion=0.06,
            energy_flow=0.0,
            spectral_shift=0.0,
            climax=0.04,
            track_intensity=0.05,
            buildup=0.0,
            anticipation=0.0,
            drop=0.0,
            calmness=0.95,
        )
        build = self._full_levels(
            music_motion=0.62,
            energy_flow=0.52,
            spectral_shift=0.18,
            climax=0.28,
            track_intensity=0.65,
            buildup=0.78,
            anticipation=0.90,
            drop=0.0,
            calmness=0.20,
        )
        impact = self._full_levels(
            music_motion=1.0,
            energy_flow=0.72,
            spectral_shift=0.35,
            climax=1.0,
            track_intensity=1.0,
            buildup=0.35,
            anticipation=1.0,
            drop=1.0,
            calmness=0.0,
        )
        canvas = QImage(360, 360, QImage.Format_ARGB32_Premultiplied)
        now = 0.0

        def advance(levels: AudioLevels, frames: int) -> list[float]:
            nonlocal now
            increments = []
            for _ in range(frames):
                before = vis._flow_phase
                painter = QPainter(canvas)
                vis.render(painter, QRectF(20, 20, 320, 320), now, levels)
                painter.end()
                increments.append(vis._flow_phase - before)
                now += 1.0 / 60.0
            return increments

        advance(slow, 75)
        slow_motion = advance(slow, 30)
        build_motion = advance(build, 45)[-20:]
        impact_motion = advance(impact, 24)[-12:]
        self.assertTrue(all(step >= 0.0 for step in slow_motion + build_motion + impact_motion))
        self.assertGreater(sum(build_motion) / len(build_motion), sum(slow_motion) / len(slow_motion) * 2.0)
        self.assertGreater(sum(impact_motion) / len(impact_motion), sum(build_motion) / len(build_motion) * 1.35)
        self.assertLess(max(impact_motion), 0.16)  # acceleration is eased, never a phase jump
        advance(slow, 1)
        self.assertGreater(vis._sm_drop, 0.65)  # impact eases away instead of cutting out


if __name__ == "__main__":
    unittest.main()
