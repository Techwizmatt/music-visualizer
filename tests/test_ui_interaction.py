from __future__ import annotations

import unittest
from types import SimpleNamespace

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from audio import AudioLevels
from ui import LyricsInfoWidget


class _PointerEvent:
    def __init__(self, x: float, y: float, button=Qt.LeftButton) -> None:
        self._position = QPointF(x, y)
        self._button = button

    def position(self) -> QPointF:
        return QPointF(self._position)

    def button(self):
        return self._button


class UiInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_transport_has_only_working_controls_and_visual_drag_orbits(self) -> None:
        widget = LyricsInfoWidget(None, None)
        widget._timer.stop()
        canvas = QImage(480, 240, QImage.Format_ARGB32_Premultiplied)
        canvas.fill(QColor(0, 0, 0, 255))
        painter = QPainter(canvas)
        widget._draw_controls(
            painter,
            SimpleNamespace(is_playing=True),
            240.0,
            120.0,
            240.0,
            1.0,
        )
        painter.end()
        self.assertEqual(set(widget._buttons), {"prev", "playpause", "next"})

        artwork = QImage(80, 80, QImage.Format_RGB32)
        artwork.fill(QColor(210, 40, 70))
        hidden_canvas = QImage(480, 320, QImage.Format_ARGB32_Premultiplied)
        hidden_canvas.fill(QColor(0, 0, 0, 255))
        hidden_painter = QPainter(hidden_canvas)
        widget._draw_info_column(
            hidden_painter,
            SimpleNamespace(
                title="Visible Artwork",
                artist="Artist",
                album="Album",
                duration_seconds=180.0,
                is_playing=True,
            ),
            10.0,
            artwork,
            480.0,
            320.0,
            0.0,
            0.0,
            1.0,
            None,
            AudioLevels(),
            controls_visible=False,
        )
        hidden_painter.end()
        self.assertEqual(widget._buttons, {})
        self.assertTrue(widget._bar_rect.isEmpty())
        self.assertNotEqual(hidden_canvas.pixelColor(240, 75), QColor(0, 0, 0, 255))

        widget._visual_rect = QRectF(20, 20, 120, 120)
        assert widget._sphere is not None
        widget._sphere_mode = True
        widget._sphere.set_mode(True)
        yaw_before = widget._sphere._view_yaw
        widget.mousePressEvent(_PointerEvent(50, 50))
        self.assertTrue(widget._dragging_visual)
        widget.mouseMoveEvent(_PointerEvent(100, 72))
        widget.mouseReleaseEvent(_PointerEvent(100, 72))
        self.assertFalse(widget._dragging_visual)
        self.assertFalse(widget._sphere.is_dragging())
        self.assertNotEqual(widget._sphere._view_yaw, yaw_before)
        widget.shutdown()
        widget.close()


if __name__ == "__main__":
    unittest.main()
