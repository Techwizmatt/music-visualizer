from __future__ import annotations

"""
Apple Music-style fullscreen player.

Layouts
  centered : no synced lyrics -> artwork + info centered (screenshot 1)
  split    : synced lyrics    -> info column left, lyrics right (screenshot 2)
The two layouts are endpoints of one continuously animated blend, so the
column glides between them when lyrics appear/disappear.

All motion is driven from a single 60fps tick: exponential smoothing for
blends/colors, a slightly-underdamped spring for lyric scrolling.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

import now_playing as npc
from audio import AudioAnalyzer, AudioLevels
from lyrics import LyricsManager, make_sig
from now_playing import NowPlaying, NowPlayingState
from settings import (
    DEBUG_PANEL_DEFAULT,
    DEFAULT_SHOW_INFO,
    FONT_SCALE_HUD,
    FONT_SCALE_LYRICS,
    FONT_SCALE_LYRICS_MIN_PX,
    FONT_SCALE_SUB,
    FONT_SCALE_TITLE,
    LYRICS_FOCUS_FRAC,
    LYRICS_PANEL_RIGHT_MARGIN_FRAC,
    UI_FPS,
    VIS_DEFAULT_SPHERE,
    VISUALIZER_ENABLED,
)
from visualizer import BackgroundVisualizer, SphereVisualizer

# ---------------------------------------------------------------- helpers


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_in_out(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def format_time(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or math.isnan(seconds):
        return "--:--"
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def format_remaining(pos: Optional[float], dur: Optional[float]) -> str:
    if pos is None or dur is None or dur <= 0:
        return "--:--"
    r = max(0.0, dur - pos)
    s = int(round(r))
    return f"-{s // 60}:{s % 60:02d}"


class Smooth:
    """Exponential approach toward a target, with a snap deadband so values
    truly settle (an asymptotic tail makes text crawl by subpixels for
    seconds, which reads as shimmer)."""

    def __init__(self, value: float = 0.0, tau: float = 0.15, eps: float = 1e-3) -> None:
        self.value = value
        self.target = value
        self.tau = tau
        self.eps = eps

    def update(self, dt: float) -> float:
        d = self.target - self.value
        if self.tau <= 0 or abs(d) < self.eps:
            self.value = self.target
        else:
            self.value += d * (1.0 - math.exp(-dt / self.tau))
        return self.value


class Spring:
    """Critically damped spring for lyric scrolling: fast, eased, and it
    lands — a snap deadband kills the subpixel tail that looks like shaking."""

    def __init__(self, value: float = 0.0, omega: float = 11.0, zeta: float = 1.0) -> None:
        self.value = value
        self.target = value
        self.vel = 0.0
        self.omega = omega
        self.zeta = zeta

    def snap(self, v: float) -> None:
        self.value = v
        self.target = v
        self.vel = 0.0

    def update(self, dt: float) -> float:
        if abs(self.value - self.target) < 0.4 and abs(self.vel) < 2.0:
            self.value = self.target
            self.vel = 0.0
            return self.value
        dt = min(dt, 1.0 / 30.0)
        a = -(self.omega ** 2) * (self.value - self.target) - 2.0 * self.zeta * self.omega * self.vel
        self.vel += a * dt
        self.value += self.vel * dt
        return self.value


# ---------------------------------------------------------------- lyric layout


@dataclass
class LyricBlock:
    index: int
    ts: float
    sublines: List[str]
    y: float = 0.0       # top offset in flow (before dot insertion)
    height: float = 0.0


@dataclass
class LyricsLayout:
    blocks: List[LyricBlock] = field(default_factory=list)
    total_h: float = 0.0
    font_px: float = 0.0
    line_h: float = 0.0
    block_gap: float = 0.0


def wrap_text(fm: QFontMetricsF, text: str, width: float) -> List[str]:
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    cur = words[0]
    for word in words[1:]:
        trial = cur + " " + word
        if fm.horizontalAdvance(trial) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def build_lyrics_layout(
    lines: List[Tuple[float, str]], font: QFont, panel_w: float
) -> LyricsLayout:
    fm = QFontMetricsF(font)
    font_px = float(font.pixelSize())
    line_h = fm.height() * 1.08
    block_gap = font_px * 0.62

    layout = LyricsLayout(font_px=font_px, line_h=line_h, block_gap=block_gap)
    y = 0.0
    for i, (ts, text) in enumerate(lines):
        sub = wrap_text(fm, text, panel_w)
        if not sub:
            continue
        h = len(sub) * line_h
        layout.blocks.append(LyricBlock(index=i, ts=ts, sublines=sub, y=y, height=h))
        y += h + block_gap
    layout.total_h = y
    return layout


# ---------------------------------------------------------------- icons


def icon_path(name: str, r: QRectF) -> QPainterPath:
    """Vector transport icons drawn inside rect r."""
    p = QPainterPath()
    cx, cy = r.center().x(), r.center().y()
    w, h = r.width(), r.height()

    if name == "play":
        p.moveTo(r.left() + w * 0.18, r.top())
        p.lineTo(r.left() + w * 0.18, r.bottom())
        p.lineTo(r.right(), cy)
        p.closeSubpath()
    elif name == "pause":
        bw = w * 0.30
        p.addRoundedRect(QRectF(r.left() + w * 0.08, r.top(), bw, h), 2, 2)
        p.addRoundedRect(QRectF(r.right() - w * 0.08 - bw, r.top(), bw, h), 2, 2)
    elif name in ("next", "prev"):
        # two triangles + a thin bar
        def tri(x0: float, x1: float) -> None:
            p.moveTo(x0, r.top())
            p.lineTo(x0, r.bottom())
            p.lineTo(x1, cy)
            p.closeSubpath()

        if name == "next":
            tri(r.left(), cx + w * 0.02)
            tri(cx - w * 0.02, r.right() - w * 0.14)
            p.addRect(QRectF(r.right() - w * 0.10, r.top(), w * 0.10, h))
        else:
            tri(r.right(), cx - w * 0.02)
            tri(cx + w * 0.02, r.left() + w * 0.14)
            p.addRect(QRectF(r.left(), r.top(), w * 0.10, h))
    elif name == "shuffle":
        lw = h * 0.13

        def stroked(pts: List[Tuple[float, float]]) -> QPainterPath:
            c = QPainterPath()
            c.moveTo(r.left() + pts[0][0] * w, r.top() + pts[0][1] * h)
            for px_, py_ in pts[1:]:
                c.lineTo(r.left() + px_ * w, r.top() + py_ * h)
            st = QPainterPathStroker()
            st.setWidth(lw)
            st.setCapStyle(Qt.RoundCap)
            st.setJoinStyle(Qt.RoundJoin)
            return st.createStroke(c)

        p = p.united(stroked([(0.02, 0.78), (0.28, 0.78), (0.64, 0.22), (0.82, 0.22)]))
        p = p.united(stroked([(0.02, 0.22), (0.28, 0.22), (0.64, 0.78), (0.82, 0.78)]))
        for ty in (0.22, 0.78):
            tri = QPainterPath()
            tri.moveTo(r.left() + 0.80 * w, r.top() + (ty - 0.16) * h)
            tri.lineTo(r.left() + 1.00 * w, r.top() + ty * h)
            tri.lineTo(r.left() + 0.80 * w, r.top() + (ty + 0.16) * h)
            tri.closeSubpath()
            p = p.united(tri)
    elif name == "repeat":
        lw = h * 0.13
        rr = QRectF(r.left() + lw / 2, r.top() + h * 0.14, w - lw, h * 0.72)
        ring = QPainterPath()
        ring.addRoundedRect(rr, h * 0.26, h * 0.26)
        st = QPainterPathStroker()
        st.setWidth(lw)
        st.setCapStyle(Qt.RoundCap)
        st.setJoinStyle(Qt.RoundJoin)
        p = st.createStroke(ring)
        tri = QPainterPath()
        tri.moveTo(cx - w * 0.06, r.top())
        tri.lineTo(cx + w * 0.16, r.top() + h * 0.14)
        tri.lineTo(cx - w * 0.06, r.top() + h * 0.30)
        tri.closeSubpath()
        p = p.united(tri)
    return p


# ---------------------------------------------------------------- main widget


class LyricsInfoWidget(QWidget):
    def __init__(
        self,
        np_state: Optional[NowPlayingState],
        lyrics: Optional[LyricsManager],
        audio: Optional[AudioAnalyzer] = None,
        vis: Optional[BackgroundVisualizer] = None,
    ) -> None:
        super().__init__()
        self._np_state = np_state
        self._lyrics = lyrics
        self._audio = audio
        self._vis = vis if VISUALIZER_ENABLED else None
        self._vis_enabled = self._vis is not None
        self._sphere = SphereVisualizer() if VISUALIZER_ENABLED else None
        self._sphere_mode: bool = VIS_DEFAULT_SPHERE and self._sphere is not None
        if self._sphere is not None:
            self._sphere.set_mode(self._sphere_mode)

        self._show_info: bool = DEFAULT_SHOW_INFO
        self._show_lyrics: bool = True
        self._show_debug: bool = DEBUG_PANEL_DEFAULT
        self._audio_sig: str = ""

        # --- animation state ---
        self._last_tick = time.monotonic()
        self._split = Smooth(0.0, tau=0.35)          # 0 centered .. 1 split
        self._lyr_alpha = Smooth(0.0, tau=0.28)      # lyrics layer fade
        self._scroll = Spring(0.0, omega=8.5, zeta=1.0)
        self._line_anim: Dict[int, Smooth] = {}      # per-line activation
        self._dots_amount = Smooth(0.0, tau=0.22)    # dot block presence
        self._art_pulse = Smooth(0.0, tau=0.10)
        self._active_idx_prev = -2

        # --- caches ---
        self._layout_cache_key: Tuple = ()
        self._layout: Optional[LyricsLayout] = None
        self._art_cache_key: Tuple = ()
        self._art_pix: Optional[QPixmap] = None
        self._vis_art_key: str = ""
        self._last_lyrics_sig = ""

        # --- interaction ---
        self._buttons: Dict[str, QRectF] = {}
        self._hover: str = ""
        self._pressed: str = ""
        self._press_t: Dict[str, float] = {}
        self._bar_rect = QRectF()
        self._dragging_bar = False
        self._drag_frac = 0.0
        self._seek_preview_until = 0.0
        self._seek_preview_pos = 0.0
        self._last_mouse_move = time.monotonic()
        self._cursor_hidden = False

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 360)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(max(8, int(1000 / UI_FPS)))

    # ------------------------------------------------------------ public

    def toggle_info(self) -> None:
        self._show_info = not self._show_info

    def toggle_visualizer(self) -> None:
        if self._vis is not None:
            self._vis_enabled = not self._vis_enabled

    # ------------------------------------------------------------ fonts

    def _font(self, px: float, weight: QFont.Weight = QFont.Normal) -> QFont:
        f = QFont()
        f.setFamilies(["SF Pro Display", "SF Pro Text", "Helvetica Neue", "Arial"])
        f.setPixelSize(max(9, int(px)))
        f.setWeight(weight)
        return f

    # ------------------------------------------------------------ lyrics helpers

    def _maybe_request_lyrics(self, np_: NowPlaying) -> None:
        if not self._lyrics or not np_.title or not np_.artist:
            return
        if not np_.duration_seconds or np_.duration_seconds <= 0:
            return
        sig = f"{np_.title}|||{np_.artist}|||{np_.album or ''}|||{int(round(np_.duration_seconds))}"
        if sig == self._last_lyrics_sig:
            return
        self._last_lyrics_sig = sig
        self._lyrics.request_for_track(
            np_.title, np_.artist, np_.album or "", np_.duration_seconds
        )

    @staticmethod
    def _active_index(lines: List[Tuple[float, str]], t: float) -> int:
        """Last line whose timestamp <= t (binary search); -1 before first."""
        lo, hi, idx = 0, len(lines) - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if lines[mid][0] <= t + 0.10:
                idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return idx

    @staticmethod
    def _gap_progress(
        lines: List[Tuple[float, str]], active: int, t: float, duration: Optional[float]
    ) -> Tuple[bool, float]:
        """(in_gap, progress 0..1). Gaps shorter than 5s don't count."""
        MIN_GAP = 5.0
        if not lines:
            return False, 0.0
        if active == -1:
            gs, ge = 0.0, lines[0][0]
        elif active < len(lines) - 1:
            gs, ge = lines[active][0], lines[active + 1][0]
        else:
            gs = lines[-1][0]
            ge = float(duration) if duration and duration > gs else gs + 8.0
        if ge - gs < MIN_GAP:
            return False, 0.0
        # Enter the gap only after the current line has had a moment on screen.
        enter = gs + (1.6 if active >= 0 else 0.4)
        if t < enter or t > ge:
            return False, 0.0
        return True, clamp((t - gs) / (ge - gs), 0.0, 1.0)

    # ------------------------------------------------------------ paint

    def paintEvent(self, _event) -> None:  # noqa: N802
        now = time.monotonic()
        dt = clamp(now - self._last_tick, 1e-4, 0.1)
        self._last_tick = now

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        w, h = float(self.width()), float(self.height())

        np_: Optional[NowPlaying] = None
        pos = 0.0
        artwork = None
        err = None
        track_key = ""
        if self._np_state is not None:
            np_, pos, artwork, err, track_key = self._np_state.snapshot()

        if now < self._seek_preview_until:
            pos = self._seek_preview_pos

        # Feed the beat engine the track identity + live position so it can
        # warm-start from the analysis cache and record the track profile.
        if self._audio is not None and np_ is not None:
            if hasattr(self._audio, "note_position"):
                self._audio.note_position(pos)
            if (
                np_.title and np_.artist and np_.duration_seconds
                and hasattr(self._audio, "set_track")
            ):
                a_sig = make_sig(
                    np_.title, np_.artist, np_.album or "", int(round(np_.duration_seconds))
                )
                if a_sig != self._audio_sig:
                    self._audio_sig = a_sig
                    self._audio.set_track(a_sig, np_.duration_seconds)

        levels = self._audio.snapshot() if self._audio else AudioLevels()

        # ---- lyrics state ----
        st = None
        if self._lyrics is not None:
            if np_ is not None:
                self._maybe_request_lyrics(np_)
            st = self._lyrics.snapshot()
        have_lyrics = bool(st and st.has_synced and st.lines) and self._show_lyrics

        # ---- background ----
        # Re-derive palettes when the track changes OR when artwork arrives
        # late (iTunes fallback fetch / delayed MediaRemote push).
        art_sig = f"{track_key}|{id(artwork)}"
        if art_sig != self._vis_art_key:
            if self._vis is not None:
                self._vis.set_artwork(artwork)
            if self._sphere is not None:
                self._sphere.set_artwork(artwork)
            self._vis_art_key = art_sig

        morph = self._sphere.morph_value() if self._sphere is not None else 0.0
        if self._vis is not None and self._vis_enabled:
            self._vis.render(painter, int(w), int(h), now, levels)
        else:
            painter.fillRect(self.rect(), QColor(12, 12, 20))
        if morph > 0.001 and self._sphere is not None:
            # Darken toward the globe's stage as the morph progresses — the
            # artwork color wash keeps glowing through, just subdued.
            stage = self._sphere.bg_color()
            stage.setAlpha(int(200 * morph))
            painter.fillRect(self.rect(), stage)

        # Legibility gradient — kept light so the artwork wash stays rich.
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(0, 0, 0, 72))
        grad.setColorAt(0.45, QColor(0, 0, 0, 48))
        grad.setColorAt(1.0, QColor(0, 0, 0, 110))
        painter.fillRect(self.rect(), QBrush(grad))

        # ---- layout blend ----
        # The column moves only on a *definitive* answer: while a new track's
        # lyrics are still being fetched the layout holds its current shape,
        # so the artwork never wiggles left-right during loading.
        if st is None or not self._show_lyrics:
            self._split.target = 0.0
        elif have_lyrics:
            self._split.target = 1.0
        elif st.resolved or st.status in ("retry", "idle"):
            self._split.target = 0.0
        # else: still searching -> hold the current target
        e = ease_in_out(self._split.update(dt))

        # Lyrics fade as a layer of their own (they can appear/disappear while
        # the layout itself doesn't move).
        self._lyr_alpha.target = 1.0 if have_lyrics else 0.0
        lyr_alpha = self._lyr_alpha.update(dt)

        if np_ is None:
            self._draw_idle(painter, w, h)
            painter.end()
            return

        # ---- artwork pulse from audio (predicted beat) ----
        self._art_pulse.target = levels.beat if levels.ok else 0.0
        pulse = self._art_pulse.update(dt)

        # ---- info column ----
        if self._show_info:
            self._draw_info_column(painter, np_, pos, artwork, w, h, e, pulse, now, err, levels)
        else:
            self._buttons.clear()
            self._bar_rect = QRectF()
            if self._sphere is not None and (self._sphere_mode or self._sphere.morph_value() > 0.001):
                # Info hidden: the globe becomes the ambient centerpiece.
                s = min(h * 0.62, w * 0.45)
                scx = lerp(w * 0.5, w * 0.26, e)
                self._sphere.render(
                    painter, QRectF(scx - s / 2, h * 0.5 - s / 2, s, s), now, levels
                )

        # ---- lyrics panel ----
        vis = e * lyr_alpha
        if st is not None and st.has_synced and st.lines and vis > 0.02:
            self._draw_lyrics(
                painter, st.lines, st.last_track_sig, pos, np_.duration_seconds, w, h, vis, dt, now
            )
        else:
            self._scroll.snap(0.0)
            self._line_anim.clear()
            self._active_idx_prev = -2
            self._dots_amount.value = 0.0
            self._dots_amount.target = 0.0

        # ---- overlays ----
        self._draw_status_overlays(painter, st, levels, np_, w, h, now)
        if self._show_debug:
            self._draw_debug_panel(painter, w, h, levels)
        self._update_cursor(now)
        painter.end()

    # ------------------------------------------------------------ idle

    def _draw_idle(self, painter: QPainter, w: float, h: float) -> None:
        f = self._font(h * FONT_SCALE_TITLE * 1.2, QFont.DemiBold)
        painter.setFont(f)
        painter.setPen(QPen(QColor(255, 255, 255, 120)))
        painter.drawText(QRectF(0, h * 0.44, w, h * 0.1), Qt.AlignCenter, "Nothing playing")
        f2 = self._font(h * FONT_SCALE_HUD, QFont.Normal)
        painter.setFont(f2)
        painter.setPen(QPen(QColor(255, 255, 255, 60)))
        painter.drawText(
            QRectF(0, h * 0.5, w, h * 0.06), Qt.AlignCenter, "Play something in Music"
        )

    # ------------------------------------------------------------ info column

    def _draw_info_column(
        self,
        painter: QPainter,
        np_: NowPlaying,
        pos: float,
        artwork,
        w: float,
        h: float,
        e: float,
        pulse: float,
        now: float,
        err: Optional[str],
        levels: Optional[AudioLevels] = None,
    ) -> None:
        art_c = min(h * 0.46, w * 0.34)
        art_s = min(h * 0.42, w * 0.24)
        art_size = lerp(art_c, art_s, e)

        cx = lerp(w * 0.5, w * 0.24, e)  # column center x

        col_top = lerp(h * 0.16, h * 0.185, e)
        art_x = cx - art_size / 2.0
        art_y = col_top

        art_rect = QRectF(art_x, art_y, art_size, art_size)
        morph = self._sphere.morph_value() if self._sphere is not None else 0.0

        # Artwork fades out over the first stretch of the morph while its
        # dots lift off toward the sphere (and back in on the way home).
        art_opacity = clamp(1.0 - morph / 0.35, 0.0, 1.0)
        if art_opacity > 0.003:
            painter.save()
            painter.setOpacity(art_opacity)

            # Soft glow behind artwork, pulsing gently with the beat.
            glow_r = art_size * (0.72 + 0.05 * pulse)
            glow = QRadialGradient(QPointF(cx, art_y + art_size * 0.55), glow_r * 1.6)
            gc = QColor(0, 0, 0, 0)
            wc = QColor(255, 255, 255, int(14 + 26 * pulse))
            glow.setColorAt(0.0, wc)
            glow.setColorAt(1.0, gc)
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(cx, art_y + art_size * 0.55), glow_r * 1.6, glow_r * 1.6)

            # Artwork: cached at the display's real pixel density so it stays
            # crisp on Retina screens.
            radius = art_size * 0.045
            dpr = float(self.devicePixelRatioF() or 1.0)
            key = (id(artwork), int(art_size), int(dpr * 100))
            if artwork is not None and not artwork.isNull():
                if key != self._art_cache_key or self._art_pix is None:
                    px = max(1, int(art_size * dpr))
                    self._art_pix = QPixmap.fromImage(artwork).scaled(
                        QSize(px, px),
                        Qt.KeepAspectRatioByExpanding,
                        Qt.SmoothTransformation,
                    )
                    self._art_pix.setDevicePixelRatio(dpr)
                    self._art_cache_key = key
                path = QPainterPath()
                path.addRoundedRect(art_rect, radius, radius)
                painter.setClipPath(path)
                painter.drawPixmap(QPointF(art_x, art_y), self._art_pix)
            else:
                painter.setBrush(QBrush(QColor(30, 30, 38, 200)))
                painter.setPen(QPen(QColor(70, 70, 82, 160)))
                painter.drawRoundedRect(art_rect, radius, radius)
            painter.restore()

        if self._sphere is not None and (self._sphere_mode or morph > 0.001):
            pad = art_size * 0.10
            self._sphere.render(
                painter,
                art_rect.adjusted(-pad, -pad, pad, pad),
                now,
                levels if levels is not None else AudioLevels(),
                img_rect=art_rect,
            )

        # --- text info ---
        col_w = art_size
        info_x = cx - col_w / 2.0
        y = art_y + art_size + h * 0.035

        title_f = self._font(h * FONT_SCALE_TITLE, QFont.Bold)
        sub_f = self._font(h * FONT_SCALE_SUB, QFont.Normal)
        hud_f = self._font(h * FONT_SCALE_HUD, QFont.Normal)

        # Title/subtitle: centered in the centered layout, left-aligned in the
        # split layout — the x position lerps so the transition glides.
        painter.setFont(title_f)
        fm_t = QFontMetricsF(title_f)
        title = fm_t.elidedText(np_.title or "Unknown Title", Qt.ElideRight, col_w)
        painter.setPen(QPen(QColor(255, 255, 255, 242)))
        tw = min(fm_t.horizontalAdvance(title), col_w)
        tx = info_x + (col_w - tw) / 2.0 * (1.0 - e)
        painter.drawText(QRectF(tx, y, col_w, fm_t.height()), Qt.AlignLeft | Qt.AlignVCenter, title)
        y += fm_t.height() * 1.12

        painter.setFont(sub_f)
        fm_s = QFontMetricsF(sub_f)
        subtitle = " — ".join([x for x in [(np_.artist or "").strip(), (np_.album or "").strip()] if x])
        subtitle = fm_s.elidedText(subtitle or "Unknown Artist", Qt.ElideRight, col_w)
        painter.setPen(QPen(QColor(255, 255, 255, 150)))
        sw = min(fm_s.horizontalAdvance(subtitle), col_w)
        sx = info_x + (col_w - sw) / 2.0 * (1.0 - e)
        painter.drawText(QRectF(sx, y, col_w, fm_s.height()), Qt.AlignLeft | Qt.AlignVCenter, subtitle)
        y += fm_s.height() * 1.5

        # --- progress bar ---
        dur = float(np_.duration_seconds or 0.0)
        frac = clamp(pos / dur, 0.0, 1.0) if dur > 0 else 0.0
        if self._dragging_bar:
            frac = self._drag_frac
        bar_h = max(4.0, h * 0.0042)
        bar_rect = QRectF(info_x, y, col_w, bar_h)
        self._bar_rect = QRectF(info_x, y - 8, col_w, bar_h + 16)  # generous hit area

        hover_boost = 1.6 if (self._hover == "bar" or self._dragging_bar) else 1.0
        r = bar_h * hover_boost / 2.0
        bh = bar_h * hover_boost
        by = y + bar_h / 2.0 - bh / 2.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 70)))
        painter.drawRoundedRect(QRectF(info_x, by, col_w, bh), r, r)
        painter.setBrush(QBrush(QColor(255, 255, 255, 235)))
        painter.drawRoundedRect(QRectF(info_x, by, col_w * frac, bh), r, r)
        y += bh + h * 0.008

        # --- times ---
        painter.setFont(hud_f)
        fm_h = QFontMetricsF(hud_f)
        painter.setPen(QPen(QColor(255, 255, 255, 130)))
        shown_pos = frac * dur if dur > 0 else pos
        painter.drawText(QRectF(info_x, y, col_w / 2, fm_h.height()), Qt.AlignLeft, format_time(shown_pos))
        painter.drawText(
            QRectF(info_x + col_w / 2, y, col_w / 2, fm_h.height()),
            Qt.AlignRight,
            format_remaining(shown_pos, dur if dur > 0 else None),
        )
        y += fm_h.height() + h * 0.022

        # --- transport controls ---
        self._draw_controls(painter, np_, cx, y, h, now)

        if err:
            painter.setFont(hud_f)
            painter.setPen(QPen(QColor(255, 120, 120, 200)))
            painter.drawText(
                QRectF(info_x, y + h * 0.06, col_w, fm_h.height() * 2),
                Qt.AlignLeft | Qt.TextWordWrap,
                fm_h.elidedText(err, Qt.ElideRight, col_w * 2),
            )

    def _draw_controls(
        self, painter: QPainter, np_: NowPlaying, cx: float, y: float, h: float, now: float
    ) -> None:
        small = h * 0.020
        med = h * 0.026
        big = h * 0.034
        gap = h * 0.052

        playing = None
        if self._np_state is not None:
            playing = self._np_state.effective_playing()
        if playing is None:
            playing = bool(np_.is_playing)

        names = ["shuffle", "prev", "playpause", "next", "repeat"]
        sizes = {"shuffle": small, "prev": med, "playpause": big, "next": med, "repeat": small}

        cy = y + big / 2.0
        xs = {
            "shuffle": cx - gap * 2.0,
            "prev": cx - gap,
            "playpause": cx,
            "next": cx + gap,
            "repeat": cx + gap * 2.0,
        }

        self._buttons.clear()
        for name in names:
            s = sizes[name]
            press_age = now - self._press_t.get(name, -10.0)
            press_k = math.exp(-press_age / 0.16) if press_age >= 0 else 0.0
            scale = 1.0 - 0.12 * press_k
            s_draw = s * scale

            rect = QRectF(xs[name] - s_draw / 2.0, cy - s_draw / 2.0, s_draw, s_draw)
            hit = QRectF(xs[name] - s / 2 - 12, cy - s / 2 - 12, s + 24, s + 24)
            self._buttons[name] = hit

            alpha = 235 if name == "playpause" else 175
            if self._hover == name:
                alpha = 255
            col = QColor(255, 255, 255, alpha)

            icon = "pause" if (name == "playpause" and playing) else (
                "play" if name == "playpause" else name
            )
            path = icon_path(icon, rect)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(col))
            painter.drawPath(path)

    # ------------------------------------------------------------ lyrics panel

    def _draw_lyrics(
        self,
        painter: QPainter,
        lines: List[Tuple[float, str]],
        sig: str,
        pos: float,
        duration: Optional[float],
        w: float,
        h: float,
        e: float,
        dt: float,
        now: float,
    ) -> None:
        if self._show_info or self._sphere_mode:
            # In sphere mode the globe keeps the left half even when the
            # info column is hidden.
            panel_x = w * 0.475
        else:
            panel_x = w * 0.14
        panel_w = w * (1.0 - LYRICS_PANEL_RIGHT_MARGIN_FRAC) - panel_x
        panel_y = h * 0.10
        panel_h = h * 0.80

        font_px = max(FONT_SCALE_LYRICS_MIN_PX, h * FONT_SCALE_LYRICS)
        font = self._font(font_px, QFont.Bold)

        # Keyed by CONTENT (track sig), never object identity — snapshots hand
        # out fresh list copies every frame, and rebuilding here would reset
        # the scroll spring and every line animation each frame (= jumpy).
        cache_key = (sig, len(lines), int(panel_w), int(font_px))
        if cache_key != self._layout_cache_key or self._layout is None:
            self._layout = build_lyrics_layout(lines, font, panel_w)
            self._layout_cache_key = cache_key
            self._scroll.snap(-1.0)  # force re-snap below
            self._line_anim.clear()
            self._active_idx_prev = -2

        layout = self._layout
        if not layout.blocks:
            return

        t = float(max(0.0, pos))
        active = self._active_index(lines, t)

        # --- interlude dots ---
        in_gap, gap_prog = self._gap_progress(lines, active, t, duration)
        self._dots_amount.target = 1.0 if in_gap else 0.0
        dots_amt = self._dots_amount.update(dt)
        dot_block_h = layout.font_px * 1.35 * dots_amt

        # --- per-line activation animations ---
        if active != self._active_idx_prev:
            self._active_idx_prev = active
        for blk in layout.blocks:
            an = self._line_anim.get(blk.index)
            if an is None:
                an = Smooth(0.0, tau=0.20)
                self._line_anim[blk.index] = an
            an.target = 1.0 if blk.index == active else 0.0

        # --- scroll target: active block centered at focus line ---
        focus_y = panel_y + panel_h * LYRICS_FOCUS_FRAC

        def block_y(blk: LyricBlock) -> float:
            """Flow y with the dot block's animated height inserted."""
            y = blk.y
            if dot_block_h > 0.0:
                if active == -1:
                    y += dot_block_h  # dots sit before the first line
                elif blk.index > active:
                    y += dot_block_h  # dots sit right after the active block
            return y

        if active >= 0:
            target_blk = next((b for b in layout.blocks if b.index == active), layout.blocks[0])
            target = block_y(target_blk) + target_blk.height / 2.0
        elif dot_block_h > 0.0:
            target = dot_block_h / 2.0  # focus the waiting dots
        else:
            target = layout.blocks[0].height / 2.0

        if self._scroll.value < -0.5:
            self._scroll.snap(target)
        else:
            self._scroll.target = target
        scroll = self._scroll.update(dt)

        base_alpha = e  # panel fades in with the split transition
        origin_y = focus_y - scroll

        painter.save()
        painter.setClipRect(QRectF(panel_x - w * 0.02, panel_y - h * 0.04, panel_w + w * 0.05, panel_h + h * 0.08))

        fm = QFontMetricsF(font)
        ascent = fm.ascent()

        for blk in layout.blocks:
            y0 = origin_y + block_y(blk)
            if y0 + blk.height < panel_y - h * 0.05 or y0 > panel_y + panel_h + h * 0.05:
                continue

            a = self._line_anim[blk.index].update(dt)

            # Role brightness: upcoming lines sit at a steady level, past
            # lines fade further the older they are (Apple-style), and every
            # line CROSSFADES toward/away from full white via its activation
            # value — the highlight never snaps.
            if blk.index < active:
                dist = active - blk.index
                base = max(0.06, 0.40 * (0.60 ** (dist - 1)))
            else:
                base = 0.42
            bright = lerp(base, 1.0, a)
            alpha = int(255 * clamp(bright, 0.0, 1.0) * base_alpha)

            # Edge fade: long, soft dissolve at the top so past lines melt
            # away; tighter fade at the bottom.
            mid = y0 + blk.height / 2.0
            edge = 1.0
            top_m = panel_y + panel_h * 0.18
            top_0 = panel_y - panel_h * 0.02
            bot_m = panel_y + panel_h * 0.90
            if mid < top_m:
                edge = clamp((mid - top_0) / (top_m - top_0), 0.0, 1.0)
            elif mid > bot_m:
                edge = clamp(1.0 - (mid - bot_m) / (panel_y + panel_h + h * 0.02 - bot_m), 0.0, 1.0)
            alpha = int(alpha * ease_in_out(edge))
            if alpha <= 2:
                continue

            # Whisper of scale — a gentle bloom, not a pop.
            scale = lerp(0.985, 1.0, a)
            painter.save()
            painter.translate(panel_x, mid)
            painter.scale(scale, scale)
            painter.translate(-panel_x, -mid)
            painter.setFont(font)
            painter.setPen(QPen(QColor(255, 255, 255, alpha)))
            yy = y0 + ascent
            for sub in blk.sublines:
                painter.drawText(QPointF(panel_x, yy), sub)
                yy += layout.line_h
            painter.restore()

        # --- draw the dots ---
        if dots_amt > 0.03:
            if active == -1:
                dy = origin_y + dot_block_h / 2.0
            else:
                target_blk = next((b for b in layout.blocks if b.index == active), None)
                if target_blk is not None:
                    dy = origin_y + target_blk.y + target_blk.height + layout.block_gap * 0.2 + dot_block_h / 2.0
                else:
                    dy = focus_y
            dot_r = layout.font_px * 0.16
            spacing = dot_r * 3.2
            breathe = 1.0 + 0.10 * math.sin(now * 2.4)
            for i in range(3):
                seg0 = i / 3.0
                seg1 = (i + 1) / 3.0
                fill = clamp((gap_prog - seg0) / (seg1 - seg0), 0.0, 1.0) if in_gap else 0.0
                a = (0.22 + 0.78 * fill) * dots_amt * base_alpha
                col = QColor(255, 255, 255, int(255 * a))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(col))
                rr = dot_r * breathe * (0.85 + 0.3 * fill)
                painter.drawEllipse(QPointF(panel_x + dot_r + i * spacing, dy), rr, rr)

        painter.restore()

    # ------------------------------------------------------------ overlays

    def _draw_status_overlays(
        self,
        painter: QPainter,
        st,
        levels: AudioLevels,
        np_: Optional[NowPlaying],
        w: float,
        h: float,
        now: float,
    ) -> None:
        hud_f = self._font(h * FONT_SCALE_HUD, QFont.Normal)
        painter.setFont(hud_f)
        fm = QFontMetricsF(hud_f)

        # Lyrics fetch status (top right, subtle)
        msg = ""
        if st is not None and np_ is not None and self._show_lyrics:
            if st.fetching:
                msg = "Searching lyrics…"
            elif st.status == "retry":
                sec = max(0, int(round(st.next_retry_t - time.time())))
                msg = f"Lyrics retry in {sec}s"
        if msg:
            spin_r = fm.height() * 0.32
            xx = w - w * 0.02 - spin_r
            yy = h * 0.035
            phase = (now * 1.8) % 1.0
            for i in range(8):
                ang = phase * math.tau + i * math.tau / 8
                aa = int(30 + 170 * (i / 7.0))
                pen = QPen(QColor(255, 255, 255, aa))
                pen.setWidthF(max(1.5, spin_r * 0.28))
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.drawLine(
                    QPointF(xx + spin_r * 0.45 * math.cos(ang), yy + spin_r * 0.45 * math.sin(ang)),
                    QPointF(xx + spin_r * math.cos(ang), yy + spin_r * math.sin(ang)),
                )
            painter.setPen(QPen(QColor(255, 255, 255, 110)))
            painter.drawText(
                QRectF(0, yy - fm.height() / 2, xx - spin_r - 8, fm.height()),
                Qt.AlignRight | Qt.AlignVCenter,
                msg,
            )

        # Audio routing hint (only when playing but loopback is silent)
        playing = bool(np_ and np_.is_playing)
        if playing and levels.status == "silent":
            hint = "No audio signal — set output to “Multi-Output Device” so BlackHole hears the music"
        elif playing and levels.status == "no-device":
            hint = "BlackHole input not found — visualizer running in ambient mode"
        else:
            hint = ""
        if hint:
            painter.setPen(QPen(QColor(255, 255, 255, 70)))
            painter.drawText(
                QRectF(w * 0.02, h - fm.height() * 2.0, w * 0.96, fm.height()),
                Qt.AlignLeft | Qt.AlignVCenter,
                hint,
            )

        # Keyboard hints, shown briefly after mouse movement
        idle = now - self._last_mouse_move
        if idle < 4.0 and np_ is not None:
            a = int(90 * clamp(1.0 - (idle - 3.0), 0.0, 1.0))
            painter.setPen(QPen(QColor(255, 255, 255, a)))
            painter.drawText(
                QRectF(w * 0.02, h - fm.height() * 3.6, w * 0.96, fm.height()),
                Qt.AlignLeft | Qt.AlignVCenter,
                "Space play/pause    ⇤ ⇥ track    F fullscreen    S sphere    L lyrics    I info    V visualizer    D debug    Q quit",
            )

    # ------------------------------------------------------------ debug panel

    def _mono_font(self, px: float) -> QFont:
        f = QFont()
        f.setFamilies(["Menlo", "Monaco", "Courier New"])
        f.setPixelSize(max(9, int(px)))
        return f

    def _draw_debug_panel(self, painter: QPainter, w: float, h: float, levels: AudioLevels) -> None:
        dbg: Dict = {}
        if self._audio is not None and hasattr(self._audio, "debug_info"):
            try:
                dbg = self._audio.debug_info()
            except Exception:
                dbg = {}

        base_px = max(11.0, h * 0.0115)
        f_small = self._mono_font(base_px)
        f_big = self._mono_font(base_px * 1.9)
        f_head = self._font(base_px * 1.05, QFont.DemiBold)
        fm_s = QFontMetricsF(f_small)
        fm_b = QFontMetricsF(f_big)

        pad = base_px * 1.2
        pw = max(320.0, w * 0.19)
        row = fm_s.height() * 1.28
        spark_h = base_px * 4.4
        meters_h = base_px * 4.0
        ph = (pad * 2 + fm_s.height() * 1.3 + fm_b.height() + base_px * 2.4
              + spark_h + row * 4 + meters_h + base_px * 2.2)
        x = w - pw - w * 0.015
        y = h * 0.075

        painter.save()
        painter.setPen(QPen(QColor(255, 255, 255, 26)))
        painter.setBrush(QBrush(QColor(8, 9, 13, 178)))
        painter.drawRoundedRect(QRectF(x, y, pw, ph), 12, 12)

        xx = x + pad
        yy = y + pad
        inner_w = pw - pad * 2

        # -- header + lock status
        bpm = float(dbg.get("bpm", levels.bpm))
        conf = float(dbg.get("conf", levels.beat_conf))
        clock_ok = bool(dbg.get("clock_ok", False))
        status = str(dbg.get("status", levels.status))
        if conf >= 0.6:
            st_txt, st_col = "LOCKED", QColor(120, 235, 160)
        elif conf >= 0.28:
            st_txt, st_col = "TRACKING", QColor(240, 210, 120)
        elif clock_ok and status in ("ok", "silent"):
            st_txt, st_col = "SEARCHING", QColor(200, 200, 210)
        else:
            st_txt, st_col = "IDLE", QColor(150, 150, 160)
        painter.setFont(f_head)
        painter.setPen(QPen(QColor(255, 255, 255, 170)))
        painter.drawText(QPointF(xx, yy + fm_s.height()), "BEAT ENGINE")
        painter.setPen(QPen(st_col))
        head_w = QFontMetricsF(f_head).horizontalAdvance(st_txt)
        painter.drawText(QPointF(x + pw - pad - head_w, yy + fm_s.height()), st_txt)
        yy += fm_s.height() * 1.55

        # -- BPM + confidence bar
        painter.setFont(f_big)
        painter.setPen(QPen(QColor(255, 255, 255, 235)))
        bpm_txt = f"{bpm:5.1f} BPM" if bpm > 0 else "  --.- BPM"
        painter.drawText(QPointF(xx, yy + fm_b.ascent()), bpm_txt)
        bar_w = inner_w * 0.34
        bar_x = x + pw - pad - bar_w
        bar_y = yy + fm_b.ascent() - base_px * 0.9
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 40)))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, base_px * 0.8), 3, 3)
        painter.setBrush(QBrush(st_col))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w * clamp(conf, 0, 1), base_px * 0.8), 3, 3)
        painter.setFont(f_small)
        painter.setPen(QPen(QColor(255, 255, 255, 140)))
        painter.drawText(QPointF(bar_x, bar_y - base_px * 0.4), f"conf {conf * 100:3.0f}%")
        yy += fm_b.height() * 1.1

        # -- metronome: 4 beats, current one kicks with the predicted pulse
        beat_idx = int(dbg.get("beat_idx", 0))
        dot_r = base_px * 0.55
        spacing = dot_r * 4.2
        cy = yy + dot_r * 1.4
        for i in range(4):
            cxx = xx + dot_r + i * spacing
            if i == beat_idx and conf > 0.2:
                rr = dot_r * (1.0 + 0.55 * levels.beat)
                col = QColor(255, 255, 255, 240)
            else:
                rr = dot_r * 0.72
                col = QColor(255, 255, 255, 70)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(col))
            painter.drawEllipse(QPointF(cxx, cy), rr, rr)
        # phase bar under the dots
        pb_y = cy + dot_r * 2.0
        pb_w = inner_w
        painter.setBrush(QBrush(QColor(255, 255, 255, 34)))
        painter.drawRoundedRect(QRectF(xx, pb_y, pb_w, 3), 1.5, 1.5)
        painter.setBrush(QBrush(QColor(255, 255, 255, 190)))
        painter.drawRoundedRect(QRectF(xx, pb_y, pb_w * clamp(levels.beat_phase, 0, 1), 3), 1.5, 1.5)
        yy = pb_y + base_px * 1.4

        # -- onset envelope sparkline + predicted beat grid
        env = dbg.get("env") or []
        sp_rect = QRectF(xx, yy, inner_w, spark_h)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 16)))
        painter.drawRoundedRect(sp_rect, 4, 4)
        if len(env) > 4:
            mx = max(env) or 1.0
            n = len(env)
            pts = [
                QPointF(
                    sp_rect.left() + i / (n - 1) * sp_rect.width(),
                    sp_rect.bottom() - clamp(v / mx, 0, 1) * (sp_rect.height() - 3) - 1.5,
                )
                for i, v in enumerate(env)
            ]
            # predicted beat grid lines over the same 3s window
            period = float(dbg.get("period", 0.0))
            next_in = float(dbg.get("next_in_ms", 0.0)) / 1000.0
            if period > 0 and conf > 0.2:
                painter.setPen(QPen(QColor(140, 220, 255, 90), 1))
                t = next_in - period
                while t >= -3.0:
                    gx = sp_rect.right() + (t / 3.0) * sp_rect.width()
                    if gx >= sp_rect.left():
                        painter.drawLine(QPointF(gx, sp_rect.top() + 2), QPointF(gx, sp_rect.bottom() - 2))
                    t -= period
            pen = QPen(QColor(255, 255, 255, 200))
            pen.setWidthF(1.4)
            painter.setPen(pen)
            painter.drawPolyline(pts)
        yy += spark_h + base_px * 0.9

        # -- data rows
        painter.setFont(f_small)
        painter.setPen(QPen(QColor(255, 255, 255, 150)))
        period_ms = float(dbg.get("period", 0.0)) * 1000.0
        rows = [
            f"next beat {float(dbg.get('next_in_ms', 0)):4.0f} ms   lookahead {float(dbg.get('lookahead_ms', 0)):3.0f} ms",
            f"period {period_ms:6.1f} ms   clock {'ok' if clock_ok else '--'}",
            (
                f"tempo cache HIT {float(dbg.get('cache_bpm', 0)):.1f}"
                if dbg.get("cache") == "hit"
                else "tempo cache --"
            )
            + f"   profile {float(dbg.get('profile_cov', 0)) * 100:3.0f}% ({int(dbg.get('profile_secs', 0))}s)",
            f"{str(dbg.get('device', ''))[:22]} @ {int(dbg.get('sr', 0))}   {status}",
        ]
        for r in rows:
            yy += row
            painter.drawText(QPointF(xx, yy), r)

        # -- band meters
        yy += base_px * 0.9
        names = ["B", "M", "H", "R", "P", "K"]
        vals = [levels.bass, levels.mid, levels.high, levels.rms, levels.pulse, levels.beat]
        bw = base_px * 0.9
        gap = (inner_w - bw * len(vals)) / (len(vals) - 1)
        m_h = meters_h - fm_s.height()
        for i, (nm, v) in enumerate(zip(names, vals)):
            bx = xx + i * (bw + gap)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 255, 30)))
            painter.drawRoundedRect(QRectF(bx, yy, bw, m_h), 2, 2)
            vv = clamp(v, 0.0, 1.0)
            painter.setBrush(QBrush(QColor(255, 255, 255, 200) if nm != "K" else QColor(140, 220, 255, 220)))
            painter.drawRoundedRect(QRectF(bx, yy + m_h * (1 - vv), bw, m_h * vv), 2, 2)
            painter.setPen(QPen(QColor(255, 255, 255, 120)))
            painter.drawText(
                QRectF(bx - gap / 2, yy + m_h + 2, bw + gap, fm_s.height()),
                Qt.AlignHCenter | Qt.AlignTop,
                nm,
            )
        painter.restore()

    # ------------------------------------------------------------ input

    def _update_cursor(self, now: float) -> None:
        idle = now - self._last_mouse_move
        window = self.window()
        fullscreen = bool(window and window.isFullScreen())
        if fullscreen and idle > 3.0 and not self._cursor_hidden:
            self.setCursor(Qt.BlankCursor)
            self._cursor_hidden = True
        elif (idle <= 3.0 or not fullscreen) and self._cursor_hidden:
            self.unsetCursor()
            self._cursor_hidden = False

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._last_mouse_move = time.monotonic()
        p = event.position()
        if self._dragging_bar and self._bar_rect.width() > 0:
            self._drag_frac = clamp((p.x() - self._bar_rect.left()) / self._bar_rect.width(), 0.0, 1.0)
            return
        hover = ""
        for name, rect in self._buttons.items():
            if rect.contains(p):
                hover = name
                break
        if not hover and self._bar_rect.contains(p):
            hover = "bar"
        if hover != self._hover:
            self._hover = hover
        self.setCursor(Qt.PointingHandCursor if hover else Qt.ArrowCursor)
        if not hover and self._cursor_hidden:
            self.unsetCursor()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._last_mouse_move = time.monotonic()
        p = event.position()
        for name, rect in self._buttons.items():
            if rect.contains(p):
                self._press_t[name] = time.monotonic()
                self._pressed = name
                return
        if self._bar_rect.contains(p) and self._bar_rect.width() > 0:
            self._dragging_bar = True
            self._drag_frac = clamp((p.x() - self._bar_rect.left()) / self._bar_rect.width(), 0.0, 1.0)
            return
        self._pressed = ""

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        p = event.position()
        if self._dragging_bar:
            self._dragging_bar = False
            np_, _, _, _, _ = self._np_state.snapshot() if self._np_state else (None, 0, None, None, "")
            dur = float(np_.duration_seconds or 0.0) if np_ else 0.0
            if dur > 0:
                target = self._drag_frac * dur
                npc.seek(target)
                self._seek_preview_pos = target
                self._seek_preview_until = time.monotonic() + 1.2
            return
        if self._pressed and self._buttons.get(self._pressed, QRectF()).contains(p):
            self._activate_button(self._pressed)
        self._pressed = ""

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        p = event.position()
        over_ui = self._bar_rect.contains(p) or any(r.contains(p) for r in self._buttons.values())
        if not over_ui:
            win = self.window()
            if isinstance(win, QMainWindow):
                win.showNormal() if win.isFullScreen() else win.showFullScreen()

    def _activate_button(self, name: str) -> None:
        if name == "playpause":
            npc.send_command("toggle-play-pause")
            if self._np_state is not None:
                cur = self._np_state.effective_playing()
                self._np_state.note_optimistic_playing(not bool(cur))
        elif name == "next":
            npc.send_command("next-track")
        elif name == "prev":
            npc.send_command("previous-track")
        elif name == "shuffle":
            npc.send_command("toggle-shuffle")
        elif name == "repeat":
            npc.send_command("toggle-repeat")

    def handle_key(self, key: int) -> bool:
        if key == Qt.Key_Space:
            self._activate_button("playpause")
            return True
        if key == Qt.Key_Right:
            npc.send_command("next-track")
            return True
        if key == Qt.Key_Left:
            npc.send_command("previous-track")
            return True
        if key == Qt.Key_I:
            self.toggle_info()
            return True
        if key == Qt.Key_V:
            self.toggle_visualizer()
            return True
        if key == Qt.Key_S:
            if self._sphere is not None:
                self._sphere_mode = not self._sphere_mode
                self._sphere.set_mode(self._sphere_mode)
            return True
        if key == Qt.Key_L:
            self._show_lyrics = not self._show_lyrics
            return True
        if key == Qt.Key_D:
            self._show_debug = not self._show_debug
            return True
        return False


# ---------------------------------------------------------------- window


class MainWindow(QMainWindow):
    def __init__(self, widget: LyricsInfoWidget) -> None:
        super().__init__()
        self.setWindowTitle("Music Visualizer")
        self.setCentralWidget(widget)
        self.resize(1280, 800)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        w = self.centralWidget()

        if key == Qt.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            event.accept()
            return
        if key == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
            event.accept()
            return
        if key == Qt.Key_Q:
            QApplication.instance().quit()
            event.accept()
            return
        if isinstance(w, LyricsInfoWidget) and w.handle_key(key):
            event.accept()
            return
        super().keyPressEvent(event)
