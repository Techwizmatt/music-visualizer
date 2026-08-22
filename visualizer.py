from __future__ import annotations

"""
Audio-reactive background renderer.

Aesthetic target: Apple Music's fullscreen player — a soft, slowly swirling
color wash derived from the album artwork. Here the wash is a field of large
radial-gradient blobs rendered into a tiny offscreen image and upscaled with
smooth filtering (which doubles as a free, high-quality blur). Audio levels
(bass/mid/high/pulse) modulate blob size, brightness and drift speed.

With no audio signal the field keeps drifting gently on its own so the
background never looks frozen.
"""

import math
import queue
import random
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF, QRadialGradient

from audio import AudioLevels
from settings import (
    VIS_AUDIO_GAIN,
    VIS_BG_BRIGHTNESS,
    VIS_BLOB_COUNT,
    VIS_FOREGROUND_RENDER_MAX_PX,
    VIS_IDLE_MOTION,
    VIS_RENDER_MAX_W,
    VIS_SPHERE_DOTS,
)

try:
    import shiboken6 as _shiboken
except Exception:  # pragma: no cover
    _shiboken = None


def np_to_qpolygonf(xs: np.ndarray, ys: np.ndarray) -> QPolygonF:
    """Build a QPolygonF from numpy arrays without a per-point Python loop
    (memcpy into the polygon's buffer — the pyqtgraph technique). Falls back
    to append() if the fast path is unavailable."""
    n = int(len(xs))
    poly = QPolygonF()
    if _shiboken is not None and n:
        try:
            poly.resize(n)
            vp = _shiboken.VoidPtr(poly.data(), n * 16, True)
            mem = np.frombuffer(vp, dtype=np.float64).reshape((-1, 2))
            mem[:, 0] = xs
            mem[:, 1] = ys
            return poly
        except Exception:
            poly = QPolygonF()
    for i in range(n):
        poly.append(QPointF(float(xs[i]), float(ys[i])))
    return poly

_FALLBACK_PALETTE = [
    QColor(64, 96, 176),
    QColor(120, 70, 170),
    QColor(36, 130, 150),
    QColor(160, 80, 120),
    QColor(40, 70, 130),
    QColor(90, 110, 190),
]


def extract_palette(img: Optional[QImage], count: int = 6) -> List[QColor]:
    """
    Cheap, fast dominant-color extraction: downscale, then greedily pick
    high-scoring colors that are far enough apart in hue.
    """
    if img is None or img.isNull():
        return list(_FALLBACK_PALETTE)

    small = img.scaled(QSize(28, 28), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    small = small.convertToFormat(QImage.Format_RGB32)

    candidates = []
    for y in range(small.height()):
        for x in range(small.width()):
            c = QColor(small.pixel(x, y))
            h, s, v, _ = c.getHsv()
            if v < 24:
                continue
            # Favor saturated, mid-bright colors; tolerate pastels.
            score = (s / 255.0) ** 1.2 * (0.35 + 0.65 * (1.0 - abs(v - 165.0) / 165.0))
            candidates.append((score, h if h >= 0 else 0, s, v))
    if not candidates:
        return list(_FALLBACK_PALETTE)

    candidates.sort(key=lambda t: -t[0])
    picked: List[QColor] = []
    for score, h, s, v in candidates:
        ok = True
        for pc in picked:
            ph = pc.getHsv()[0]
            ph = ph if ph >= 0 else 0
            dh = min(abs(h - ph), 360 - abs(h - ph))
            if dh < 26:
                ok = False
                break
        if not ok:
            continue
        # Clamp into a range that glows nicely on a dark base.
        s2 = max(70, min(235, int(s * 1.15)))
        v2 = max(95, min(215, int(v)))
        picked.append(QColor.fromHsv(h, s2, v2))
        if len(picked) >= count:
            break

    # Pad with lightness variants of REAL artwork colors — hue-rotating here
    # paints monochrome covers in colors they don't contain.
    i = 0
    while len(picked) < count:
        base = picked[i % len(picked)] if picked else _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)]
        variant = base.lighter(145) if (i % 2 == 0) else base.darker(140)
        h, s, v, _ = variant.getHsv()
        picked.append(QColor.fromHsv(max(0, h), s, max(90, min(230, v))))
        i += 1
    return picked


@dataclass
class _Blob:
    color: QColor
    band: str          # "bass" | "mid" | "high" | "rms"
    cx: float          # base center (0..1)
    cy: float
    ax: float          # drift amplitudes (0..1)
    ay: float
    wx: float          # drift angular speeds
    wy: float
    px: float          # phases
    py: float
    radius: float      # base radius as fraction of min(w,h) of render target
    sm_level: float = 0.0


class BackgroundVisualizer:
    def __init__(self) -> None:
        self._rng = random.Random(7)
        self._blobs: List[_Blob] = []
        self._palette: List[QColor] = list(_FALLBACK_PALETTE)
        self._base_color = QColor(10, 10, 18)
        self._canvas: Optional[QImage] = None
        self._t_drift = 0.0
        self._last_t: Optional[float] = None
        self._rebuild_blobs()

    # ---------- palette ----------

    def set_artwork(self, img: Optional[QImage]) -> None:
        self._palette = extract_palette(img, VIS_BLOB_COUNT)
        # Base = darkened average of the palette, keeps the wash cohesive.
        r = sum(c.red() for c in self._palette) // len(self._palette)
        g = sum(c.green() for c in self._palette) // len(self._palette)
        b = sum(c.blue() for c in self._palette) // len(self._palette)
        vb = VIS_BG_BRIGHTNESS
        self._base_color = QColor(
            min(255, int(r * 0.30 * vb)),
            min(255, int(g * 0.30 * vb)),
            min(255, int(b * 0.36 * vb)),
        )
        for i, blob in enumerate(self._blobs):
            blob.color = self._palette[i % len(self._palette)]

    def _rebuild_blobs(self) -> None:
        rng = self._rng
        bands = ["bass", "mid", "high", "rms"]
        self._blobs = []
        for i in range(VIS_BLOB_COUNT):
            band = bands[i % len(bands)]
            big = band in ("bass", "rms")
            self._blobs.append(
                _Blob(
                    color=self._palette[i % len(self._palette)],
                    band=band,
                    cx=rng.uniform(0.12, 0.88),
                    cy=rng.uniform(0.15, 0.85),
                    ax=rng.uniform(0.10, 0.30),
                    ay=rng.uniform(0.10, 0.28),
                    wx=rng.uniform(0.05, 0.16) * (1.0 if i % 2 else -1.0),
                    wy=rng.uniform(0.04, 0.13) * (-1.0 if i % 3 else 1.0),
                    px=rng.uniform(0, math.tau),
                    py=rng.uniform(0, math.tau),
                    radius=rng.uniform(0.55, 0.85) if big else rng.uniform(0.32, 0.5),
                )
            )

    # ---------- rendering ----------

    def render(self, painter: QPainter, w: int, h: int, now: float, levels: AudioLevels) -> None:
        if w <= 0 or h <= 0:
            return

        dt = 0.0 if self._last_t is None else max(0.0, min(0.1, now - self._last_t))
        self._last_t = now

        # Audio drives drift speed a little; silence still drifts slowly.
        energy = (levels.rms * 0.7 + levels.pulse * 0.3) if levels.ok else 0.0
        idle = VIS_IDLE_MOTION
        self._t_drift += dt * (0.55 + idle * 0.4 + energy * 1.1)
        t = self._t_drift

        rw = min(VIS_RENDER_MAX_W, max(64, w // 8))
        rh = max(36, int(rw * h / max(1, w)))
        if self._canvas is None or self._canvas.width() != rw or self._canvas.height() != rh:
            self._canvas = QImage(rw, rh, QImage.Format_ARGB32_Premultiplied)

        img = self._canvas
        img.fill(self._base_color)

        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode_Plus)  # additive glow
        p.setPen(Qt.NoPen)

        mn = float(min(rw, rh))
        for i, blob in enumerate(self._blobs):
            if levels.ok and not levels.silent:
                lvl = {
                    "bass": levels.bass,
                    "mid": levels.mid,
                    "high": levels.high,
                    "rms": levels.rms,
                }[blob.band] * VIS_AUDIO_GAIN
                if blob.band == "bass":
                    lvl = min(1.0, lvl + levels.beat * 0.55)
            else:
                # Gentle autonomous breathing when no signal.
                lvl = idle * (0.45 + 0.35 * math.sin(t * 0.35 + i * 1.7))
                lvl = max(0.22, lvl)

            # Extra smoothing per blob keeps motion creamy.
            blob.sm_level += (lvl - blob.sm_level) * (1.0 - math.exp(-dt / 0.11)) if dt else 0.0
            lvl = max(0.0, min(1.0, blob.sm_level))

            cx = (blob.cx + blob.ax * math.sin(blob.wx * t + blob.px)) * rw
            cy = (blob.cy + blob.ay * math.sin(blob.wy * t + blob.py)) * rh
            radius = blob.radius * mn * (0.72 + 0.55 * lvl)

            c = QColor(blob.color)
            alpha = int(min(230.0, (42 + 152 * (lvl ** 1.05)) * VIS_BG_BRIGHTNESS))
            grad = QRadialGradient(QPointF(cx, cy), max(4.0, radius))
            c.setAlpha(alpha)
            grad.setColorAt(0.0, c)
            mid_c = QColor(c)
            mid_c.setAlpha(int(alpha * 0.45))
            grad.setColorAt(0.55, mid_c)
            out_c = QColor(c)
            out_c.setAlpha(0)
            grad.setColorAt(1.0, out_c)
            p.setBrush(grad)
            p.drawEllipse(QPointF(cx, cy), radius, radius)

        p.end()

        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(QRect(0, 0, w, h), img)
        painter.restore()


# ================================================================ sphere


def weighted_palette(img: Optional[QImage], k: int = 6) -> List[Tuple[QColor, float]]:
    """
    Dominant artwork colors WITH their frequency share, most common first.
    Quantizes a downscaled copy, merges near-duplicates, and lifts very dark
    colors just enough to glow on a black stage without losing character.
    """
    fallback = [
        (QColor(150, 155, 175), 0.42),
        (QColor(90, 110, 190), 0.22),
        (QColor(120, 70, 170), 0.14),
        (QColor(36, 130, 150), 0.10),
        (QColor(200, 120, 90), 0.07),
        (QColor(220, 210, 200), 0.05),
    ]
    if img is None or img.isNull():
        return fallback

    small = img.scaled(QSize(40, 40), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    small = small.convertToFormat(QImage.Format_RGB32)

    bins: dict = {}
    for y in range(small.height()):
        for x in range(small.width()):
            c = small.pixel(x, y)
            r, g, b = (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF
            q = (r >> 4, g >> 4, b >> 4)
            if q in bins:
                e = bins[q]
                e[0] += 1; e[1] += r; e[2] += g; e[3] += b
            else:
                bins[q] = [1, r, g, b]

    total = float(small.width() * small.height())
    ranked = sorted(bins.values(), key=lambda e: -e[0])

    merged: List[List[float]] = []  # [count, r, g, b] averaged
    for cnt, sr, sg, sb in ranked:
        r, g, b = sr / cnt, sg / cnt, sb / cnt
        placed = False
        for m in merged:
            if abs(m[1] - r) + abs(m[2] - g) + abs(m[3] - b) < 72:
                w0, w1 = m[0], float(cnt)
                m[1] = (m[1] * w0 + r * w1) / (w0 + w1)
                m[2] = (m[2] * w0 + g * w1) / (w0 + w1)
                m[3] = (m[3] * w0 + b * w1) / (w0 + w1)
                m[0] += cnt
                placed = True
                break
        if not placed:
            merged.append([float(cnt), r, g, b])
    merged.sort(key=lambda m: -m[0])

    out: List[Tuple[QColor, float]] = []
    for m in merged[:k]:
        col = QColor(int(m[1]), int(m[2]), int(m[3]))
        h, s, v, _ = col.getHsv()
        # Lift toward visibility on black, preserving hue/sat character.
        v2 = max(115, min(235, int(v * 0.85 + 60)))
        s2 = min(255, int(s * 1.08))
        out.append((QColor.fromHsv(max(0, h), s2, v2), m[0] / total))
    # Pad with lightness variants of REAL artwork colors — never invent hues
    # the artwork doesn't contain.
    i = 0
    while len(out) < k:
        base = out[i % len(out)][0] if out else fallback[i][0]
        variant = base.lighter(140) if (i % 2 == 0) else base.darker(135)
        h, s, v, _ = variant.getHsv()
        out.append((QColor.fromHsv(max(0, h), s, max(110, v)), 0.02))
        i += 1
    return out


class SphereVisualizer:
    """
    Nine particle/mesh geometries in the artwork's own colors, weighted by how
    common each color is. The sphere remains style 1; rarer artwork colors get
    fewer but bigger, livelier dots that pop with the music in every style.
    Class colors crossfade smoothly when the track changes.

    Also owns the artwork<->geometry morph: every dot has a "home" position
    sampled from the album art (matched by color), so toggling the foreground
    dissolves the artwork into the selected shape and back.

    All math is numpy-vectorized; dots are drawn in (class, depth, size)
    buckets with round-cap pens — thousands of dots at 60fps.
    """

    N_CLASSES = 6
    STYLE_NAMES = {
        1: "Particle Sphere",
        2: "Chroma Ribbons",
        3: "Waveform Ring",
        4: "Liquid Orbit",
        5: "Harmonic Knot",
        6: "Double Helix",
        7: "CRT Wavefield",
        8: "Artwork Vortex",
        9: "Artwork Relief",
    }
    _DEPTH_ALPHA = (0.30, 0.60, 1.0)   # back, mid, front
    _MORPH_SEC = 0.9

    def __init__(self, n_dots: int = VIS_SPHERE_DOTS) -> None:
        rng = np.random.default_rng(11)
        self._rng = rng
        n = max(400, int(n_dots))
        self._n = n

        i = np.arange(n, dtype=np.float64)
        y = 1.0 - 2.0 * (i + 0.5) / n
        r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
        phi = i * (math.pi * (3.0 - math.sqrt(5.0)))
        base = np.stack([r * np.cos(phi), y, r * np.sin(phi)], axis=1)
        # Shuffle so class assignment isn't correlated with latitude.
        perm = rng.permutation(n)
        self._base = base[perm]
        coord_i = i[perm]
        self._coord = (coord_i + 0.5) / n
        self._u = np.mod(coord_i * (math.pi * (3.0 - math.sqrt(5.0))), math.tau)
        self._v = np.mod(coord_i * math.sqrt(2.0) * 2.137, math.tau)
        grid_cols = max(24, int(round(math.sqrt(n * 1.6))))
        grid_rows = int(math.ceil(n / grid_cols))
        self._grid_x = ((np.arange(n) % grid_cols) / max(1, grid_cols - 1)) * 2.0 - 1.0
        self._grid_z = ((np.arange(n) // grid_cols) / max(1, grid_rows - 1)) * 2.0 - 1.0

        # Fixed class membership (colors change per artwork, membership never
        # does — that keeps color transitions perfectly clean).
        shares = np.array([0.50, 0.20, 0.12, 0.08, 0.06, 0.04])
        self._cls = rng.choice(self.N_CLASSES, size=n, p=shares)
        self._feature_lane = rng.integers(0, 8, size=n)
        self._music_drive = np.zeros(8, dtype=np.float64)

        # Rarity rank 0..1 per class (0 = most common).
        rarity = np.linspace(0.0, 1.0, self.N_CLASSES)
        self._cls_size = 0.85 + 1.05 * rarity          # rare dots are bigger
        self._cls_jitter = 0.006 + 0.055 * rarity      # ...and livelier
        self._cls_burst = 0.12 + 0.95 * rarity         # ...and pop on the beat
        self._cls_alpha = 0.86 + 0.14 * rarity         # ...and a touch brighter

        self._shell = 1.0 + rng.normal(0.0, 0.010, n)
        self._jit_phase = rng.uniform(0.0, math.tau, n)
        self._jit_speed = rng.uniform(0.5, 1.9, n)
        self._burst_i = rng.uniform(0.3, 1.0, n)
        self._size_sub = (rng.random(n) < 0.30).astype(np.int64)  # 30% "large" variant
        self._stagger = rng.uniform(0.0, 1.0, n)       # morph departure offsets

        # Class colors as float RGB arrays for smooth crossfading.
        self._col_cur = np.zeros((self.N_CLASSES, 3))
        self._col_tgt = np.zeros((self.N_CLASSES, 3))
        self._have_colors = False

        # Mode 9 gets its own denser, regular artwork grid. Keeping it separate
        # raises relief resolution without slowing the sphere/galaxy modes.
        self._relief_classes = 12
        self._relief_side = max(88, int(round(math.sqrt(n * 2.25))))
        relief_axis = (np.arange(self._relief_side, dtype=np.float64) + 0.5) / self._relief_side
        relief_x, relief_y = np.meshgrid(relief_axis, relief_axis)
        self._relief_uv = np.column_stack((relief_x.ravel(), relief_y.ravel()))
        self._relief_x = (self._relief_uv[:, 0] - 0.5) * 1.72
        self._relief_y = (0.5 - self._relief_uv[:, 1]) * 1.72
        self._relief_cls = np.zeros(len(self._relief_uv), dtype=np.int64)
        self._relief_phase = rng.uniform(0.0, math.tau, len(self._relief_uv))
        self._relief_stagger = np.clip(
            np.linalg.norm(self._relief_uv - 0.5, axis=1) / math.sqrt(0.5),
            0.0,
            1.0,
        )
        self._relief_col_cur = np.zeros((self._relief_classes, 3), dtype=np.float64)
        self._relief_col_tgt = np.zeros((self._relief_classes, 3), dtype=np.float64)
        self._relief_regions = 8
        self._relief_region = np.zeros(len(self._relief_uv), dtype=np.int64)
        self._relief_region_centers = np.zeros((self._relief_regions, 2), dtype=np.float64)
        self._relief_region_phase = rng.uniform(0.0, math.tau, self._relief_regions)
        self._relief_region_saliency = np.zeros(self._relief_regions, dtype=np.float64)
        self._relief_reveal_threshold = np.zeros(len(self._relief_uv), dtype=np.float64)
        self._relief_region_indices: list[np.ndarray] = []
        self._relief_foreground_regions: list[int] = []
        self._relief_background_region = 0
        self._artwork_square: Optional[QImage] = None
        self._relief_group_indices: list[np.ndarray] = []
        self._relief_highlight_indices: list[np.ndarray] = []
        self._refresh_relief_groups()

        # Morph state: 0 = artwork, 1 = selected foreground geometry.
        self._morph = 0.0
        self._mode_sphere = False
        self._style = 1
        self._style_fx = 1.0
        self._shape_pts = self._base.copy()
        self._img_uv = np.column_stack([rng.random(n), rng.random(n)])
        self._wave_history = np.zeros((25, 72), dtype=np.float64)
        self._wave_history_valid = False
        self._wave_history_timeline = False
        self._wave_timeline_center = len(self._wave_history) // 2
        self._wave_accum = 0.0
        self._wave_phase = np.zeros(3, dtype=np.float64)
        self._wave_head = np.zeros(72, dtype=np.float64)

        self._ang = 0.0
        self._flow_phase = 0.0
        # User orbit is layered over the autonomous/music-driven motion.
        # Releasing a drag leaves a short, damped angular glide so the object
        # never snaps or interrupts the current musical phrase.
        self._view_yaw = 0.0
        self._view_pitch = 0.0
        self._view_yaw_velocity = 0.0
        self._view_pitch_velocity = 0.0
        self._view_dragging = False
        self._last_t: Optional[float] = None
        self._sm_bass = 0.0
        self._sm_rms = 0.0
        self._dance = 0.0
        self._sm_motion = 0.0
        self._sm_flow = 0.0
        self._sm_shift = 0.0
        self._sm_climax = 0.0
        self._sm_intensity = 0.0
        self._sm_buildup = 0.0
        self._sm_anticipation = 0.0
        self._sm_drop = 0.0
        self._vortex_dissolve = 0.0
        self._sm_vortex_energy = 0.0
        self._bg = QColor(5, 5, 9)
        self.set_artwork(None)

    # ------------------------------ mode / morph

    def set_mode(self, sphere: bool) -> None:
        self._mode_sphere = bool(sphere)

    def set_style(self, style: int) -> None:
        style = max(1, min(9, int(style)))
        if style != self._style:
            self._style = style
            self._style_fx = 0.0
            if style == 8:
                self._vortex_dissolve = 0.0
            if style == 7:
                self._wave_history_valid = False

    def style(self) -> int:
        return self._style

    def style_name(self) -> str:
        return self.STYLE_NAMES[self._style]

    def morph_value(self) -> float:
        return self._morph

    # ------------------------------ pointer interaction

    def begin_drag(self) -> None:
        self._view_dragging = True
        self._view_yaw_velocity = 0.0
        self._view_pitch_velocity = 0.0

    def drag_view(self, dx: float, dy: float, dt: float) -> None:
        """Orbit the current animation without touching its audio timeline."""
        if not self._view_dragging:
            self.begin_drag()
        dt = max(1.0 / 240.0, min(0.08, float(dt)))
        yaw_delta = float(dx) * 0.0082
        pitch_delta = float(dy) * 0.0072
        self._view_yaw += yaw_delta
        self._view_pitch = float(np.clip(
            self._view_pitch + pitch_delta,
            -1.22,
            1.22,
        ))
        # Blend recent pointer speed so release inertia is stable even when
        # mouse events arrive at irregular intervals.
        yaw_velocity = float(np.clip(yaw_delta / dt, -7.5, 7.5))
        pitch_velocity = float(np.clip(pitch_delta / dt, -6.0, 6.0))
        self._view_yaw_velocity += (yaw_velocity - self._view_yaw_velocity) * 0.48
        self._view_pitch_velocity += (pitch_velocity - self._view_pitch_velocity) * 0.48

    def end_drag(self) -> None:
        self._view_dragging = False

    def is_dragging(self) -> bool:
        return self._view_dragging

    def _interactive_flat_view(self, painter: QPainter, rect: QRectF) -> None:
        """Give the flatter mesh modes a restrained camera-like response."""
        amount = self._smoothstep01((self._morph - 0.16) / 0.68)
        if amount <= 0.0:
            return
        yaw = self._view_yaw * amount
        pitch = self._view_pitch * amount
        cx, cy = rect.center().x(), rect.center().y()
        painter.translate(cx, cy)
        painter.rotate(math.degrees(0.055 * math.sin(yaw) - 0.035 * math.sin(pitch)))
        painter.shear(0.14 * math.sin(yaw), -0.10 * math.sin(pitch))
        painter.scale(
            0.91 + 0.09 * abs(math.cos(yaw)),
            0.90 + 0.10 * abs(math.cos(pitch)),
        )
        painter.translate(-cx, -cy)

    # ------------------------------ palette / artwork

    def set_artwork(self, img: Optional[QImage]) -> None:
        self._vortex_dissolve = 0.0
        pal = weighted_palette(img, self.N_CLASSES)
        for c in range(self.N_CLASSES):
            col = pal[c][0]
            self._col_tgt[c] = (col.red(), col.green(), col.blue())
        if not self._have_colors:
            self._col_cur[:] = self._col_tgt
            self._have_colors = True

        relief_pal = weighted_palette(img, self._relief_classes)
        while len(relief_pal) < self._relief_classes:
            base_color, share = relief_pal[len(relief_pal) % max(1, len(relief_pal))]
            relief_pal.append((base_color.lighter(112 + (len(relief_pal) % 3) * 8), share * 0.5))
        for c in range(self._relief_classes):
            color = relief_pal[c][0]
            self._relief_col_tgt[c] = (color.red(), color.green(), color.blue())
        if not np.any(self._relief_col_cur):
            self._relief_col_cur[:] = self._relief_col_tgt

        avg = pal[0][0]
        self._bg = QColor(
            int(4 + avg.red() * 0.030),
            int(4 + avg.green() * 0.030),
            int(8 + avg.blue() * 0.045),
        )
        self._assign_home_positions(img)
        self._assign_relief_artwork(img)

    def _assign_home_positions(self, img: Optional[QImage]) -> None:
        """Give each dot a home pixel in the artwork whose color matches its
        class, so the morph reads as the image dissolving into the globe."""
        n = self._n
        g = int(math.ceil(math.sqrt(n)))
        if img is None or img.isNull():
            self._img_uv = np.column_stack([self._rng.random(n), self._rng.random(n)])
            return

        side = min(img.width(), img.height())
        sq = img.copy((img.width() - side) // 2, (img.height() - side) // 2, side, side)
        small = sq.scaled(QSize(g, g), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        small = small.convertToFormat(QImage.Format_RGB32)

        cols = np.empty((g * g, 3))
        for yy in range(g):
            for xx in range(g):
                c = small.pixel(xx, yy)
                cols[yy * g + xx] = ((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)
        uv = np.column_stack([
            (np.arange(g * g) % g + 0.5) / g,
            (np.arange(g * g) // g + 0.5) / g,
        ])

        # Nearest class per sample (against target colors, pre-lift is fine).
        d = np.linalg.norm(cols[:, None, :] - self._col_tgt[None, :, :], axis=2)
        sample_cls = np.argmin(d, axis=1)

        out = np.empty((n, 2))
        for c in range(self.N_CLASSES):
            dots = np.where(self._cls == c)[0]
            if len(dots) == 0:
                continue
            samples = np.where(sample_cls == c)[0]
            if len(samples) == 0:
                samples = np.argsort(d[:, c])[: max(8, len(dots))]
            picks = samples[self._rng.integers(0, len(samples), len(dots))] if len(samples) < len(dots) \
                else self._rng.permutation(samples)[: len(dots)]
            out[dots] = uv[picks]
        self._img_uv = out

    def _assign_relief_artwork(self, img: Optional[QImage]) -> None:
        if img is None or img.isNull():
            self._artwork_square = None
            self._relief_cls = np.arange(len(self._relief_uv)) % self._relief_classes
            self._segment_relief_artwork(None)
            self._refresh_relief_groups()
            return
        side = min(img.width(), img.height())
        square = img.copy((img.width() - side) // 2, (img.height() - side) // 2, side, side)
        artwork_side = min(512, max(1, side))
        self._artwork_square = square.scaled(
            QSize(artwork_side, artwork_side),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        ).convertToFormat(QImage.Format_ARGB32_Premultiplied)
        small = square.scaled(
            QSize(self._relief_side, self._relief_side),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        ).convertToFormat(QImage.Format_RGB32)
        pixels = np.empty((len(self._relief_uv), 3), dtype=np.float64)
        for yy in range(self._relief_side):
            for xx in range(self._relief_side):
                value = small.pixel(xx, yy)
                pixels[yy * self._relief_side + xx] = (
                    (value >> 16) & 0xFF,
                    (value >> 8) & 0xFF,
                    value & 0xFF,
                )
        distance = np.linalg.norm(
            pixels[:, None, :] - self._relief_col_tgt[None, :, :],
            axis=2,
        )
        self._relief_cls = np.argmin(distance, axis=1)
        self._segment_relief_artwork(pixels)
        self._refresh_relief_groups()

    def _segment_relief_artwork(self, pixels: Optional[np.ndarray]) -> None:
        """Cut the cover into clean, connected foreground subjects.

        The border color supplies a stable background estimate. A binary
        foreground mask is cleaned and split into connected components, so a
        logo, face, or separated illustration moves as an intact silhouette
        instead of being divided into arbitrary color tiles.
        """
        uv = self._relief_uv
        count = len(uv)
        k = self._relief_regions
        foreground_regions: list[int] = []
        if pixels is None or len(pixels) != count:
            cols = 4
            rows = max(1, int(math.ceil(k / cols)))
            labels = np.minimum(
                k - 1,
                (uv[:, 1] * rows).astype(np.int64) * cols
                + (uv[:, 0] * cols).astype(np.int64),
            )
            saliency = 1.0 - np.clip(
                np.linalg.norm(uv - 0.5, axis=1) / math.sqrt(0.5),
                0.0,
                1.0,
            )
            background_region = k - 1
        else:
            rgb = np.asarray(pixels, dtype=np.float64) / 255.0
            side = self._relief_side
            image = rgb.reshape(side, side, 3)
            border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0)
            background = np.median(border, axis=0)
            contrast_2d = np.linalg.norm(image - background, axis=2) / math.sqrt(3.0)
            contrast = contrast_2d.ravel()
            gradient = np.zeros((side, side), dtype=np.float64)
            gradient[:, 1:] += np.linalg.norm(image[:, 1:] - image[:, :-1], axis=2)
            gradient[1:, :] += np.linalg.norm(image[1:, :] - image[:-1, :], axis=2)
            gradient = gradient.ravel()
            gradient /= float(np.percentile(gradient, 95.0) + 1e-9)
            saliency = np.clip(0.82 * contrast + 0.18 * gradient, 0.0, 1.0)

            # The adaptive floor rejects album-paper grain while retaining
            # saturated subjects. Neighbour cleanup closes tiny cracks but
            # deliberately preserves true negative space inside a shape.
            threshold = max(
                0.11,
                min(0.30, float(np.percentile(contrast, 75.0)) * 1.10),
            )
            mask = contrast_2d > threshold
            for _ in range(2):
                padded = np.pad(mask, 1, mode="constant", constant_values=False)
                neighbours = np.zeros_like(contrast_2d, dtype=np.int16)
                for oy in range(3):
                    for ox in range(3):
                        if ox == 1 and oy == 1:
                            continue
                        neighbours += padded[oy:oy + side, ox:ox + side]
                mask = (
                    (mask & (neighbours >= 2))
                    | ((contrast_2d > threshold * 0.82) & (neighbours >= 5))
                )

            seen = np.zeros((side, side), dtype=bool)
            components: list[np.ndarray] = []
            minimum_size = max(28, int(round(count * 0.004)))
            for yy in range(side):
                for xx in range(side):
                    if not mask[yy, xx] or seen[yy, xx]:
                        continue
                    stack = [(yy, xx)]
                    seen[yy, xx] = True
                    members: list[int] = []
                    touches_edge = False
                    while stack:
                        cy, cx = stack.pop()
                        members.append(cy * side + cx)
                        touches_edge = touches_edge or cx == 0 or cy == 0 or cx == side - 1 or cy == side - 1
                        for oy in (-1, 0, 1):
                            for ox in (-1, 0, 1):
                                if ox == 0 and oy == 0:
                                    continue
                                ny, nx = cy + oy, cx + ox
                                if (
                                    0 <= ny < side
                                    and 0 <= nx < side
                                    and mask[ny, nx]
                                    and not seen[ny, nx]
                                ):
                                    seen[ny, nx] = True
                                    stack.append((ny, nx))
                    if len(members) < minimum_size:
                        continue
                    # Rounded cover corners and screenshot shadows often touch
                    # an edge; retain them only when they are genuinely large.
                    if touches_edge and len(members) < int(count * 0.10):
                        continue
                    components.append(np.asarray(members, dtype=np.int64))

            components.sort(
                key=lambda ids: (
                    float(np.mean(saliency[ids])) * math.sqrt(float(len(ids))),
                    len(ids),
                ),
                reverse=True,
            )
            components = components[: max(0, k - 1)]
            background_region = min(len(components), k - 1)
            labels = np.full(count, background_region, dtype=np.int64)
            for region, members in enumerate(components):
                labels[members] = region
                foreground_regions.append(region)

        self._relief_region = labels
        self._relief_foreground_regions = foreground_regions
        self._relief_background_region = background_region
        self._relief_region_indices = [
            np.flatnonzero(labels == region)
            for region in range(k)
        ]
        scores = np.zeros(k, dtype=np.float64)
        for region in range(k):
            members = labels == region
            if not np.any(members):
                self._relief_region_centers[region] = (0.5, 0.5)
                continue
            self._relief_region_centers[region] = np.mean(uv[members], axis=0)
            region_saliency = saliency[members]
            scores[region] = 0.72 * float(np.mean(region_saliency)) + 0.28 * float(np.max(region_saliency))
        self._relief_region_saliency = scores
        active = [region for region in range(k) if len(self._relief_region_indices[region])]
        rank_order = sorted(active, key=lambda region: (-scores[region], region))
        ranks = np.full(k, max(0, len(active) - 1), dtype=np.float64)
        for rank, region in enumerate(rank_order):
            ranks[region] = rank
        base_threshold = 0.035 + 0.74 * ranks[labels] / max(1, len(active) - 1)
        point_stagger = 0.12 * (0.5 + 0.5 * np.sin(self._relief_phase * 2.17))
        self._relief_reveal_threshold = np.clip(
            base_threshold + point_stagger,
            0.0,
            0.94,
        )

    def _refresh_relief_groups(self) -> None:
        """Cache stable artwork-color groups outside the 60 fps render path."""
        self._relief_group_indices = [
            np.flatnonzero(self._relief_cls == cls)
            for cls in range(self._relief_classes)
        ]
        # A deterministic sparse layer supplies bright depth/musical accents
        # without sorting the entire high-resolution cloud every frame.
        self._relief_highlight_indices = [
            indices[(indices + cls * 3) % 5 == 0]
            for cls, indices in enumerate(self._relief_group_indices)
        ]

    def bg_color(self) -> QColor:
        return QColor(self._bg)

    # ------------------------------ geometry

    @staticmethod
    def _music_channels(levels: AudioLevels) -> np.ndarray:
        """Eight independent animation feeds derived from the current frame.

        Snare is a high-frequency transient proxy; vocals are a presence cue.
        Neither claims source-separated stems.
        """
        full = levels.source == "full-file"
        vocal = levels.vocal if full else levels.mid * 0.72
        snare = np.clip(
            0.38 * levels.high + 0.46 * levels.spectral_flux
            + 0.28 * levels.pulse,
            0.0,
            1.0,
        )
        width = levels.stereo_width if full else np.clip(
            0.55 * levels.mid + 0.35 * levels.high,
            0.0,
            1.0,
        )
        buildup = max(0.0, levels.buildup) if full else levels.beat * 0.35
        chorus = max(levels.drop, levels.section_change, levels.climax) if full else levels.beat
        return np.clip(
            np.array((
                levels.bass,
                levels.mid,
                vocal,
                snare,
                levels.high,
                width,
                max(buildup, levels.anticipation),
                chorus,
            ), dtype=np.float64),
            0.0,
            1.0,
        )

    def _target_geometry(self, now: float, levels: AudioLevels) -> np.ndarray:
        """Return the selected normalized 3D particle layout.

        Every layout exists for live audio. The complete-file descriptors add
        section jumps, vocal ripples, mix-width depth, and transient detail
        only once the exact now-playing recording has finished analysis.
        """
        full = levels.source == "full-file"
        vocal = levels.vocal if full else 0.0
        flux = levels.spectral_flux if full else 0.0
        width = levels.stereo_width if full else 0.0
        change = levels.section_change if full else 0.0
        phase = levels.section * math.tau if full else 0.0
        pulse = levels.beat if levels.ok and not levels.silent else 0.0
        bass, mid, high = levels.bass, levels.mid, levels.high

        if self._style == 1:
            pts = self._base.copy()
            if full:
                implode = np.clip(
                    0.48 * self._sm_anticipation + 0.28 * max(0.0, -self._sm_flow),
                    0.0,
                    0.66,
                )
                explode = 0.25 * self._sm_drop + 0.06 * self._sm_climax
                radial = 1.0 - implode + explode * (0.28 + 0.72 * self._burst_i)
                pts *= radial[:, None]
                ripple = 1.0 + (0.035 * vocal + 0.07 * self._sm_drop) * np.sin(
                    self._base[:, 1] * 11.0 + self._flow_phase * 2.4 + phase
                )
                pts *= ripple[:, None]
        elif self._style == 2:
            ring = 1.0 + (0.035 + 0.08 * vocal) * np.sin(self._u * 8.0 + now * 2.2 + phase)
            pts = self._base * ring[:, None]
            pts[:, 0] *= 1.0 + 0.20 * width
            pts[:, 1] *= 0.92 + 0.10 * mid
        elif self._style == 3:
            u = self._u
            wave = (0.10 + 0.22 * vocal + 0.10 * high + 0.10 * self._sm_drop) * np.sin(
                u * (5.0 + 3.0 * levels.brightness) - self._flow_phase * 3.1 + phase
            )
            radius = 0.78 + wave + 0.10 * pulse * np.sin(u * 3.0) ** 8
            pts = np.column_stack((
                radius * np.cos(u),
                0.30 * np.sin(self._v + self._flow_phase * 1.7) * (0.3 + vocal),
                radius * np.sin(u),
            ))
        elif self._style == 4:
            u, v = self._u, self._v
            minor = 0.22 + 0.08 * bass + 0.09 * vocal * np.sin(u * 4.0 + phase)
            major = 0.67 + 0.05 * pulse
            pts = np.column_stack(((major + minor * np.cos(v)) * np.cos(u), minor * np.sin(v), (major + minor * np.cos(v)) * np.sin(u)))
            pts *= 1.16
        elif self._style == 5:
            knot_t = self._coord * math.tau
            lane_phase = self._feature_lane * math.tau / 8.0
            tube = (
                0.18 + 0.07 * vocal + 0.06 * self._sm_drop
                + 0.035 * np.sin(lane_phase + self._flow_phase)
            )
            knot_phase = knot_t * 3.0 + lane_phase + self._flow_phase * 0.72
            turn = knot_t * 2.0 + self._flow_phase * 0.28
            ring = 0.58 + tube * np.cos(knot_phase)
            pts = np.column_stack((
                ring * np.cos(turn),
                tube * np.sin(knot_phase) * (1.15 + 0.25 * width),
                ring * np.sin(turn),
            ))
        elif self._style == 6:
            y = self._coord * 2.0 - 1.0
            strand = np.where((self._cls % 2) == 0, 0.0, math.pi)
            angle = self._coord * math.tau * 4.0 + strand + self._flow_phase * (0.8 + flux) + phase
            radius = 0.48 + 0.12 * vocal + 0.06 * pulse + 0.12 * self._sm_drop
            pts = np.column_stack((radius * np.cos(angle), y, radius * np.sin(angle)))
        elif self._style == 7:
            x, z = self._grid_x, self._grid_z
            carrier = np.sin(x * (8.0 + 5.0 * levels.brightness) - now * 4.0 - z * 5.0 + phase)
            cross = np.sin(x * 19.0 + now * 2.2 + z * 7.0) * (0.25 + high)
            y = (0.055 + 0.16 * levels.rms + 0.18 * vocal) * (carrier + 0.32 * cross)
            y += change * 0.20 * np.exp(-x * x * 4.0)
            pts = np.column_stack((x, y - 0.12, z * 0.80))
        elif self._style == 8:
            depth = self._coord
            angle = self._u + depth * 1.1 + self._flow_phase * 0.30
            tunnel_radius = 0.16 + 0.82 * depth ** 1.55
            tunnel_radius *= 1.0 - 0.18 * self._sm_anticipation + 0.22 * self._sm_drop
            pts = np.column_stack((
                tunnel_radius * np.cos(angle),
                tunnel_radius * np.sin(angle) * (0.72 + 0.12 * width),
                (0.55 - depth) * 1.25,
            ))
        else:
            x = (self._img_uv[:, 0] - 0.5) * 1.75
            y = (0.5 - self._img_uv[:, 1]) * 1.75
            z = (0.10 + 0.28 * vocal) * np.sin(x * 7.0 + y * 5.0 - now * 3.0 + phase)
            z += (self._cls / max(1, self.N_CLASSES - 1) - 0.5) * (0.16 + 0.30 * width)
            pts = np.column_stack((x, y, z))

        if full and self._style != 8:
            pts *= (
                1.0 + (0.18 * change + 0.24 * self._sm_drop) * self._burst_i
            )[:, None]
            pts[:, 1] += 0.035 * flux * np.sin(
                self._jit_phase + self._flow_phase * 6.0
            )
        return pts

    @staticmethod
    def _smoothstep01(value: float) -> float:
        value = max(0.0, min(1.0, float(value)))
        return value * value * (3.0 - 2.0 * value)

    def _source_rect(self, rect: QRectF, img_rect: Optional[QRectF]) -> QRectF:
        if img_rect is not None and img_rect.width() > 2 and img_rect.height() > 2:
            return img_rect
        side = min(rect.width(), rect.height()) * 0.58
        return QRectF(
            rect.center().x() - side * 0.5,
            rect.center().y() - side * 0.5,
            side,
            side,
        )

    def _transition_points(
        self,
        dest_x: np.ndarray,
        dest_y: np.ndarray,
        source_x: np.ndarray,
        source_y: np.ndarray,
        stagger: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        span = max(0.2, 1.0 - stagger)
        amount = self._smoothstep01((self._morph - stagger) / span)
        return (
            source_x + (dest_x - source_x) * amount,
            source_y + (dest_y - source_y) * amount,
        )

    def _update_wave_history(self, levels: AudioLevels, dt: float) -> None:
        """Build a phase-continuous, spectrally driven waveform history.

        The old implementation copied raw windows directly, so arbitrary audio
        phase changes made the surface snap. Here the complete spectrum drives
        continuous oscillators; a small smoothed real-wave component preserves
        the recording's texture without sacrificing fluid motion.
        """
        timeline = levels.waveform_timeline
        if levels.source == "full-file" and len(timeline) >= 3:
            rows = len(self._wave_history)
            source_rows = np.asarray(timeline, dtype=np.float64)
            if source_rows.ndim == 2 and source_rows.shape[1] >= 4:
                row_positions = np.linspace(0.0, len(source_rows) - 1, rows)
                target = np.empty_like(self._wave_history)
                source_x = np.arange(source_rows.shape[1], dtype=np.float64)
                target_x = np.linspace(0.0, source_rows.shape[1] - 1, target.shape[1])
                for row_i, position in enumerate(row_positions):
                    lo = int(math.floor(position))
                    hi = min(len(source_rows) - 1, lo + 1)
                    frac = position - lo
                    source_trace = source_rows[lo] * (1.0 - frac) + source_rows[hi] * frac
                    target[row_i] = np.interp(target_x, source_x, source_trace)
                target = np.clip(target, -1.0, 1.0)
                amount = 1.0 - math.exp(-dt / 0.075) if dt else 1.0
                if not self._wave_history_valid or not self._wave_history_timeline:
                    self._wave_history[:] = target
                else:
                    self._wave_history += (target - self._wave_history) * amount
                self._wave_history_valid = True
                self._wave_history_timeline = True
                center_ratio = levels.waveform_timeline_center / max(1, len(source_rows) - 1)
                self._wave_timeline_center = int(round(center_ratio * (rows - 1)))
                self._wave_head[:] = self._wave_history[self._wave_timeline_center]
                return

        if self._wave_history_timeline:
            self._wave_history_valid = False
            self._wave_history_timeline = False

        x = np.linspace(-1.0, 1.0, self._wave_history.shape[1])
        direction = 0.45 + 1.25 * self._sm_flow + 0.55 * self._sm_shift
        speed = (
            0.42 + 1.35 * self._sm_motion + 0.52 * self._sm_intensity
            + 0.72 * self._sm_anticipation + 1.20 * self._sm_drop
        )
        self._wave_phase += dt * speed * np.array((1.15, -0.72, 2.05)) * direction

        bass = max(0.05, levels.bass)
        mid = max(0.04, levels.mid)
        high = max(0.025, levels.high)
        target = (
            (0.32 + 0.44 * bass) * np.sin(math.pi * 1.35 * x + self._wave_phase[0])
            + (0.18 + 0.34 * mid) * np.sin(math.pi * 3.15 * x + self._wave_phase[1])
            + (0.07 + 0.20 * high) * np.sin(math.pi * 6.2 * x + self._wave_phase[2])
        )
        if levels.source == "full-file":
            target += 0.22 * levels.vocal * np.sin(
                math.pi * 2.1 * x - self._wave_phase[1] * 0.7 + levels.section * math.tau
            )
            target *= (
                0.78 + 0.34 * levels.stereo_width + 0.20 * self._sm_climax
                + 0.30 * self._sm_drop
            )

        raw = np.asarray(levels.waveform, dtype=np.float64)
        if len(raw) >= 4:
            raw = np.interp(
                np.linspace(0.0, len(raw) - 1, len(x)),
                np.arange(len(raw)),
                raw,
            )
            raw = np.convolve(raw, np.array((1, 2, 3, 2, 1), dtype=np.float64) / 9.0, mode="same")
            target += raw * (0.035 + 0.075 * levels.spectral_flux)
        target = np.clip(target * 0.62, -1.0, 1.0)

        head_k = 1.0 - math.exp(-dt / 0.085) if dt else 1.0
        self._wave_head += (target - self._wave_head) * head_k
        if not self._wave_history_valid:
            self._wave_history[:] = self._wave_head
            self._wave_history_valid = True

        self._wave_accum += dt
        frame_sec = 1.0 / 24.0
        if self._wave_accum >= frame_sec:
            steps = min(2, max(1, int(self._wave_accum / frame_sec)))
            for _ in range(steps):
                self._wave_history[1:] = self._wave_history[:-1]
                self._wave_history[0] = self._wave_head
            self._wave_accum %= frame_sec

    def _draw_crt_wavefield(
        self,
        painter: QPainter,
        rect: QRectF,
        levels: AudioLevels,
        img_rect: Optional[QRectF],
    ) -> None:
        """Smooth perspective wavefield that unfolds directly from artwork."""
        source = self._source_rect(rect, img_rect)
        cx = rect.center().x() + rect.width() * 0.06 * self._sm_shift
        top = rect.top() + rect.height() * 0.17
        bottom = rect.bottom() - rect.height() * 0.10
        half_near = rect.width() * (0.43 + 0.05 * max(0.0, self._sm_flow))
        half_far = rect.width() * 0.16
        fx_alpha = self._style_fx * self._smoothstep01(self._morph / 0.28)

        painter.save()
        self._interactive_flat_view(painter, rect)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        rows = len(self._wave_history)
        xs = np.linspace(-1.0, 1.0, self._wave_history.shape[1])
        for row in range(rows - 1, -1, -1):
            if self._wave_history_timeline:
                center_row = max(1, min(rows - 2, self._wave_timeline_center))
                signed = (
                    (row - center_row) / max(1, rows - 1 - center_row)
                    if row >= center_row
                    else (row - center_row) / max(1, center_row)
                )
                depth = (signed + 1.0) * 0.5
                near = 1.0 - depth
                playhead_y = rect.center().y() + rect.height() * 0.035
                dest_y = playhead_y - signed * rect.height() * 0.305
                half = half_far + near * (half_near - half_far)
            else:
                depth = row / max(1, rows - 1)
                near = 1.0 - depth
                dest_y = top + (near ** 1.30) * (bottom - top)
                half = half_far + near * (half_near - half_far)
            row_drive = float(self._music_drive[row % len(self._music_drive)])
            trace = self._wave_history[row]
            amplitude = rect.height() * (0.025 + 0.040 * near) * (
                0.62 + 0.48 * levels.rms + 0.30 * levels.vocal + 0.24 * self._sm_climax
                + 0.28 * self._sm_drop + 0.34 * row_drive
            )
            dx = cx + xs * half
            dy = dest_y - trace * amplitude

            sx = source.center().x() + xs * source.width() * 0.48
            sy = np.full_like(xs, source.top() + depth * source.height())
            sy -= trace * source.height() * 0.018
            px, py = self._transition_points(dx, dy, sx, sy, depth * 0.08)

            rgb = self._col_cur[row % self.N_CLASSES]
            if self._wave_history_timeline:
                distance = abs(row - self._wave_timeline_center)
                current_boost = math.exp(-distance / 1.35)
                alpha_base = 62 + 80 * near + 100 * current_boost
            else:
                current_boost = 0.0
                alpha_base = 45 + 125 * near
            alpha = int(np.clip(
                alpha_base * fx_alpha
                * (0.78 + 0.24 * self._sm_climax + 0.30 * row_drive),
                0,
                245,
            ))
            if row % 3 == 0 or current_boost > 0.45:
                halo = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), max(3, alpha // 7))
                halo_pen = QPen(halo)
                halo_pen.setWidthF(max(
                    1.7,
                    rect.height() * (0.0045 + 0.0035 * current_boost),
                ))
                halo_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(halo_pen)
                painter.drawPolyline(np_to_qpolygonf(px, py))
            line = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), alpha)
            pen = QPen(line)
            pen.setWidthF(max(
                0.7,
                rect.height() * (0.00135 + 0.0013 * current_boost),
            ))
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawPolyline(np_to_qpolygonf(px, py))

        # Sparse phosphor scanlines retain the CRT character without painting
        # hundreds of extra primitives over the surrounding interface.
        painter.setRenderHint(QPainter.Antialiasing, False)
        scan = QColor(0, 0, 0, int(18 * fx_alpha))
        painter.setPen(QPen(scan, 1.0))
        for yy in np.linspace(rect.top(), rect.bottom(), 16):
            painter.drawLine(QPointF(rect.left(), float(yy)), QPointF(rect.right(), float(yy)))
        painter.restore()

    def _draw_chroma_ribbons(
        self,
        painter: QPainter,
        rect: QRectF,
        levels: AudioLevels,
        img_rect: Optional[QRectF],
    ) -> None:
        """Eight flowing 3D ribbons unspool from horizontal artwork slices."""
        source = self._source_rect(rect, img_rect)
        t = np.linspace(-1.0, 1.0, 84)
        cx = rect.center().x() + rect.width() * 0.08 * self._sm_shift
        cy = rect.center().y() - rect.height() * 0.07 * self._sm_flow
        radius = min(rect.width(), rect.height()) * 0.47
        lanes = 8
        expansion = np.clip(
            0.76 + 0.20 * levels.rms + 0.22 * self._sm_climax
            + 0.20 * self._sm_flow - 0.08 * self._sm_anticipation
            + 0.30 * self._sm_drop,
            0.55,
            1.30,
        )
        fx_alpha = self._style_fx * self._smoothstep01(self._morph / 0.25)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        for lane in range(lanes):
            lane_n = lane / (lanes - 1) - 0.5
            lane_drive = float(self._music_drive[lane])
            phase = self._flow_phase * (0.82 + lane * 0.025) + lane * 0.62
            carrier = t * math.pi * (1.45 + 1.6 * levels.brightness)
            x3 = t * (0.88 + 0.12 * levels.stereo_width)
            y3 = lane_n * 0.28 + np.sin(carrier + phase) * (
                0.10 + 0.13 * levels.mid + 0.15 * levels.vocal
                + 0.10 * self._sm_anticipation + 0.15 * self._sm_drop
                + 0.16 * lane_drive
            )
            z3 = np.cos(t * math.pi * 1.25 - phase * 0.72) * (
                0.16 + 0.26 * levels.stereo_width + 0.10 * lane_drive
            )
            y3 += 0.055 * levels.spectral_flux * np.sin(t * math.pi * 8.0 - phase)
            view_amount = self._smoothstep01((self._morph - 0.16) / 0.68)
            yaw = self._view_yaw * view_amount
            pitch = self._view_pitch * view_amount
            cya, sya = math.cos(yaw), math.sin(yaw)
            cpi, spi = math.cos(pitch), math.sin(pitch)
            xv = x3 * cya + z3 * sya
            zv = -x3 * sya + z3 * cya
            yv = y3 * cpi - zv * spi
            zv = y3 * spi + zv * cpi
            perspective = 3.0 / (3.0 - zv)
            dx = cx + xv * radius * expansion * perspective
            dy = cy - yv * radius * expansion * perspective

            sx = source.left() + (t + 1.0) * 0.5 * source.width()
            sy = np.full_like(t, source.top() + (lane + 0.5) / lanes * source.height())
            px, py = self._transition_points(dx, dy, sx, sy, lane / lanes * 0.10)
            rgb = self._col_cur[lane % self.N_CLASSES]
            alpha = int(np.clip(
                (118 + lane * 10) * fx_alpha
                * (0.76 + 0.20 * self._sm_climax + 0.34 * lane_drive),
                0,
                240,
            ))
            if lane % 2 == 0:
                halo = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), max(4, alpha // 8))
                halo_pen = QPen(halo)
                halo_pen.setWidthF(max(2.0, radius * 0.012))
                painter.setPen(halo_pen)
                painter.drawPolyline(np_to_qpolygonf(px, py))
            color = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), alpha)
            pen = QPen(color)
            pen.setWidthF(max(0.9, radius * (0.0038 + lane * 0.00035)))
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawPolyline(np_to_qpolygonf(px, py))
        painter.restore()

    def _draw_harmonic_knot(
        self,
        painter: QPainter,
        rect: QRectF,
        levels: AudioLevels,
        img_rect: Optional[QRectF],
    ) -> None:
        """Artwork-colored musical lanes braided into a luminous torus knot."""
        source = self._source_rect(rect, img_rect)
        t = np.linspace(0.0, math.tau, 168)
        cx = rect.center().x() + rect.width() * 0.055 * self._sm_shift
        cy = rect.center().y() - rect.height() * 0.035 * self._sm_flow
        radius = min(rect.width(), rect.height()) * 0.39
        expansion = float(np.clip(
            0.80 + 0.13 * self._sm_intensity + 0.12 * self._sm_flow
            - 0.18 * self._sm_anticipation + 0.27 * self._sm_drop,
            0.50,
            1.24,
        ))
        fx_alpha = self._style_fx * self._smoothstep01(self._morph / 0.22)
        view_amount = self._smoothstep01((self._morph - 0.16) / 0.68)
        yaw = 0.22 * self._flow_phase + self._view_yaw * view_amount
        pitch = 0.42 + 0.12 * self._sm_shift + self._view_pitch * view_amount
        cya, sya = math.cos(yaw), math.sin(yaw)
        cpi, spi = math.cos(pitch), math.sin(pitch)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        for lane in range(8):
            drive = float(self._music_drive[lane])
            lane_phase = lane * math.tau / 8.0
            knot_phase = t * 3.0 + lane_phase + self._flow_phase * (0.68 + lane * 0.018)
            turn = t * 2.0 + self._flow_phase * 0.26
            tube = (
                0.18 + lane * 0.006 + 0.055 * levels.vocal
                + 0.075 * drive + 0.070 * self._sm_drop
            )
            ring = 0.57 + tube * np.cos(knot_phase)
            ring += 0.026 * drive * np.sin(t * (4.0 + lane % 3) + self._flow_phase * 1.7)
            if lane == 6:
                ring *= 1.0 - 0.16 * self._sm_anticipation
            elif lane == 7:
                ring *= 1.0 + 0.22 * self._sm_drop
            x3 = ring * np.cos(turn)
            y3 = tube * np.sin(knot_phase) * (1.08 + 0.24 * levels.stereo_width)
            z3 = ring * np.sin(turn)
            xv = x3 * cya + z3 * sya
            zv = -x3 * sya + z3 * cya
            yv = y3 * cpi - zv * spi
            zv = y3 * spi + zv * cpi
            perspective = 3.2 / (3.2 - zv)
            dx = cx + xv * radius * expansion * perspective
            dy = cy - yv * radius * expansion * perspective

            source_radius = min(source.width(), source.height()) * (0.12 + lane * 0.043)
            sx = source.center().x() + np.cos(t) * source_radius
            sy = source.center().y() + np.sin(t) * source_radius
            px, py = self._transition_points(dx, dy, sx, sy, lane * 0.009)

            rgb = self._col_cur[lane % self.N_CLASSES]
            alpha = int(np.clip(
                (98 + 116 * drive + 36 * self._sm_climax) * fx_alpha,
                0,
                242,
            ))
            if drive > 0.20 or lane in (0, 7):
                halo = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), max(3, alpha // 9))
                halo_pen = QPen(halo)
                halo_pen.setWidthF(max(2.0, radius * (0.011 + 0.008 * drive)))
                halo_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(halo_pen)
                painter.drawPolyline(np_to_qpolygonf(px, py))
            color = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), alpha)
            pen = QPen(color)
            pen.setWidthF(max(0.75, radius * (0.0027 + 0.0020 * drive)))
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawPolyline(np_to_qpolygonf(px, py))

            # Bright beads race along each knot lane at its own component
            # strength, making vocals, snares, bass, and drops visibly distinct.
            bead_step = max(8, 15 - int(round(drive * 6.0)))
            bead_idx = np.arange((lane * 3) % bead_step, len(px), bead_step)
            bead = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), min(255, alpha + 12))
            bead_pen = QPen(bead)
            bead_pen.setCapStyle(Qt.RoundCap)
            bead_pen.setWidthF(max(1.5, radius * (0.006 + 0.006 * drive)))
            painter.setPen(bead_pen)
            painter.drawPoints(np_to_qpolygonf(px[bead_idx], py[bead_idx]))
        painter.restore()

    def _draw_liquid_orbit(
        self,
        painter: QPainter,
        rect: QRectF,
        levels: AudioLevels,
        img_rect: Optional[QRectF],
    ) -> None:
        """Concentric artwork rings lift into a folding liquid 3D orbit."""
        source = self._source_rect(rect, img_rect)
        theta = np.linspace(0.0, math.tau, 96)
        cx, cy = rect.center().x(), rect.center().y()
        radius = min(rect.width(), rect.height()) * 0.44
        expansion = np.clip(
            0.76 + 0.20 * levels.rms + 0.24 * self._sm_climax
            + 0.18 * self._sm_flow - 0.10 * self._sm_anticipation
            + 0.42 * self._sm_drop,
            0.52,
            1.28,
        )
        view_amount = self._smoothstep01((self._morph - 0.16) / 0.68)
        tilt = (
            0.38 + 0.30 * self._sm_shift
            + 0.34 * self._sm_drop * math.sin(self._flow_phase * 0.72)
            + self._view_pitch * view_amount
        )
        ct, st = math.cos(tilt), math.sin(tilt)
        turn = self._flow_phase * 0.34 + self._view_yaw * view_amount
        ca, sa = math.cos(turn), math.sin(turn)
        fx_alpha = self._style_fx * self._smoothstep01(self._morph / 0.25)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        for lane in range(7):
            lane_drive = float(self._music_drive[lane])
            lane_phase = lane * math.tau / 7.0 + levels.section * math.pi
            major = 0.56 + lane * 0.025
            minor = (
                0.10 + 0.055 * levels.vocal + lane * 0.009
                + 0.035 * self._sm_anticipation + 0.065 * self._sm_drop
                + 0.070 * lane_drive
            )
            fold = (
                theta * (2.0 + lane % 2) + self._flow_phase + lane_phase
                + self._sm_drop * np.sin(theta * 3.0 + lane_phase) * 0.65
            )
            tube = minor * np.cos(fold)
            x = (major + tube) * np.cos(theta)
            y = minor * np.sin(fold) + 0.055 * levels.spectral_flux * np.sin(theta * 7.0 - lane_phase)
            z = (major + tube) * np.sin(theta) * (0.70 + 0.28 * levels.stereo_width)
            # Rotate without allocating a general matrix for every orbit.
            xr = x * ca + z * sa
            zr = -x * sa + z * ca
            yr = y * ct - zr * st
            zr = y * st + zr * ct
            perspective = 3.1 / (3.1 - zr)
            dx = cx + xr * radius * expansion * perspective
            dy = cy - yr * radius * expansion * perspective

            source_r = min(source.width(), source.height()) * (0.12 + lane * 0.045)
            sx = source.center().x() + np.cos(theta) * source_r
            sy = source.center().y() + np.sin(theta) * source_r
            px, py = self._transition_points(dx, dy, sx, sy, lane * 0.012)
            rgb = self._col_cur[lane % self.N_CLASSES]
            alpha = int(np.clip(
                (135 + lane * 9) * fx_alpha * (0.78 + 0.30 * lane_drive),
                0,
                240,
            ))
            if lane in (1, 4):
                halo = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), max(5, alpha // 8))
                halo_pen = QPen(halo)
                halo_pen.setWidthF(max(2.2, radius * 0.014))
                painter.setPen(halo_pen)
                painter.drawPolyline(np_to_qpolygonf(px, py))
            color = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), alpha)
            pen = QPen(color)
            pen.setWidthF(max(1.0, radius * (0.004 + lane * 0.0005)))
            painter.setPen(pen)
            painter.drawPolyline(np_to_qpolygonf(px, py))
        painter.restore()

    def _draw_relief_cloud(
        self,
        painter: QPainter,
        px: np.ndarray,
        py: np.ndarray,
        depth: np.ndarray,
        radius: float,
        fx_alpha: float,
        accent: float,
        visibility: Optional[np.ndarray] = None,
        highlights_enabled: bool = True,
    ) -> None:
        """Draw the shared dense artwork cloud without a per-frame sort."""
        depth_norm = np.clip((depth + 0.9) / 1.8, 0.0, 1.0)
        base_px = max(0.78, radius / self._relief_side * 0.46)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        for cls, indices in enumerate(self._relief_group_indices):
            visibility_alpha = 1.0
            if visibility is not None and len(indices):
                indices = indices[visibility[indices] > 0.025]
                if len(indices):
                    visibility_alpha = float(np.mean(visibility[indices]))
            if not len(indices):
                continue
            mean_depth = float(np.mean(depth_norm[indices]))
            rgb = self._relief_col_cur[cls]
            color = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            color.setAlpha(int(np.clip(
                255.0 * (0.50 + 0.40 * mean_depth) * fx_alpha * visibility_alpha,
                0.0,
                255.0,
            )))
            pen = QPen(color)
            pen.setCapStyle(Qt.RoundCap)
            pen.setWidthF(base_px * (0.86 + 0.24 * mean_depth))
            painter.setPen(pen)
            painter.drawPoints(np_to_qpolygonf(px[indices], py[indices]))

            if not highlights_enabled:
                continue
            highlights = self._relief_highlight_indices[cls]
            if visibility is not None and len(highlights):
                highlights = highlights[visibility[highlights] > 0.32]
            if len(highlights):
                highlights = highlights[depth_norm[highlights] > 0.56]
            if not len(highlights):
                continue
            glow = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            glow.setAlpha(int(np.clip(
                255.0 * (0.28 + 0.52 * accent) * fx_alpha,
                0.0,
                245.0,
            )))
            glow_pen = QPen(glow)
            glow_pen.setCapStyle(Qt.RoundCap)
            glow_pen.setWidthF(base_px * (1.22 + 0.46 * accent))
            painter.setPen(glow_pen)
            painter.drawPoints(np_to_qpolygonf(px[highlights], py[highlights]))
        painter.restore()

    def _artwork_vortex_geometry(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Move clean artwork sections without bending the cover into a tube."""
        vocal = float(self._music_drive[2])
        component = self._music_drive[self._relief_cls % len(self._music_drive)]
        energy = self._sm_vortex_energy
        offset_x, offset_y, offset_z, rotation, section_scale = self._artwork_section_motion()
        x = self._relief_x.copy()
        y = self._relief_y.copy()
        z = np.zeros_like(x)

        # A section receives one rigid transform, preserving its silhouette.
        # Fine musical ripples live in depth only and fade in with the dots.
        for region, indices in enumerate(self._relief_region_indices):
            if not len(indices):
                continue
            center_uv = self._relief_region_centers[region]
            center_x = (float(center_uv[0]) - 0.5) * 1.72
            center_y = (0.5 - float(center_uv[1])) * 1.72
            local_x = x[indices] - center_x
            local_y = y[indices] - center_y
            cr, sr = math.cos(rotation[region]), math.sin(rotation[region])
            scale = section_scale[region]
            x[indices] = (
                center_x + (local_x * cr - local_y * sr) * scale
                + offset_x[region]
            )
            y[indices] = (
                center_y + (local_x * sr + local_y * cr) * scale
                + offset_y[region]
            )
            z[indices] = offset_z[region]

        # Independent pieces must not drag the composition away from the
        # artwork's visual center as their musical roles separate.
        x -= float(np.mean(x))
        y -= float(np.mean(y))

        lane_rate = 1.02 + 0.065 * (self._relief_cls % len(self._music_drive))
        ripple = np.sin(self._relief_phase + self._flow_phase * lane_rate)
        z += energy * (
            (0.018 + 0.075 * vocal + 0.055 * self._sm_motion)
            * np.sin(self._relief_x * 6.2 + self._relief_y * 4.4 - self._flow_phase * 1.5)
            + (0.028 + 0.040 * energy)
            * np.sin(self._relief_x * 12.0 - self._relief_y * 7.0 + self._flow_phase)
            + (0.020 + 0.025 * component) * ripple
        )

        # Builds gently compress the sections; choruses release them without
        # changing the recognizable width/height of the complete cover.
        bloom = float(np.clip(
            0.94 + 0.035 * self._sm_flow - 0.040 * self._sm_anticipation
            + 0.12 * energy,
            0.87,
            1.09,
        ))
        x *= bloom
        y *= bloom * (0.98 + 0.035 * self._music_drive[5])
        z *= 0.94 + 0.18 * self._music_drive[5] + 0.08 * energy

        view_amount = self._smoothstep01((self._morph - 0.14) / 0.70)
        # Autonomous motion rocks around the front-facing view. It never
        # reaches the edge-on angle that caused the hourglass/narrow-strip bug.
        turn, tilt = self._artwork_vortex_camera_angles(energy, view_amount)
        ca, sa = math.cos(turn), math.sin(turn)
        ct, st = math.cos(tilt), math.sin(tilt)
        xr = x * ca + z * sa
        zr = -x * sa + z * ca
        yr = y * ct - zr * st
        zr = y * st + zr * ct
        return xr, yr, zr

    def _artwork_vortex_camera_angles(
        self,
        energy: float,
        view_amount: float,
    ) -> tuple[float, float]:
        """Return a centered, interactive camera constrained to 15 degrees."""
        auto_turn = (
            (0.11 + 0.16 * energy)
            * math.sin(self._ang * 0.62 + self._flow_phase * 0.10)
            + 0.055 * self._sm_shift
        )
        auto_tilt = (
            0.055 + 0.075 * self._sm_shift
            + (0.035 + 0.045 * energy) * math.sin(self._flow_phase * 0.21)
        )
        max_angle = math.radians(15.0)
        turn = float(np.clip(
            auto_turn + max_angle * 0.62 * math.tanh(self._view_yaw) * view_amount,
            -max_angle,
            max_angle,
        ))
        tilt = float(np.clip(
            auto_tilt + max_angle * 0.50 * math.tanh(self._view_pitch) * view_amount,
            -max_angle,
            max_angle,
        ))
        return turn, tilt

    def _artwork_section_motion(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return continuous rigid motion for the detected dot sections."""
        k = self._relief_regions
        offset_x = np.zeros(k, dtype=np.float64)
        offset_y = np.zeros(k, dtype=np.float64)
        offset_z = np.zeros(k, dtype=np.float64)
        rotation = np.zeros(k, dtype=np.float64)
        scale = np.ones(k, dtype=np.float64)
        energy = self._sm_vortex_energy
        separation = float(np.clip(
            0.010 + 0.17 * energy,
            0.008,
            0.18,
        ))
        max_score = float(np.max(self._relief_region_saliency) + 1e-9)
        for region in range(k):
            if not len(self._relief_region_indices[region]):
                continue
            drive = float(self._music_drive[region % len(self._music_drive)])
            importance = 0.55 + 0.45 * float(self._relief_region_saliency[region]) / max_score
            if region == self._relief_background_region:
                importance *= 0.22
            phase = (
                self._relief_region_phase[region]
                + self._flow_phase * (0.24 + 0.035 * region)
            )
            amplitude = separation * importance * (0.78 + 0.22 * drive)
            offset_x[region] = amplitude * math.cos(phase)
            offset_y[region] = amplitude * 0.78 * math.sin(phase * 1.13)
            offset_z[region] = amplitude * (0.65 + 0.45 * drive) * math.sin(phase * 0.81)
            rotation[region] = (
                0.014 + 0.105 * energy
            ) * importance * math.sin(phase * 0.91)
            scale[region] = 1.0 + (
                0.010 + 0.075 * energy
            ) * importance * math.sin(phase * 1.27)
        return offset_x, offset_y, offset_z, rotation, scale

    def _artwork_vortex_plane_rect(
        self,
        rect: QRectF,
        source: QRectF,
        radius: float,
    ) -> QRectF:
        """Return the shared plane used by both the cover and newborn dots."""
        plane_size = radius * 1.72
        target = QRectF(
            rect.center().x() - plane_size * 0.5,
            rect.center().y() - plane_size * 0.5,
            plane_size,
            plane_size,
        )
        travel = self._smoothstep01((self._morph - 0.03) / 0.62)
        art_rect = QRectF(
            source.x() + (target.x() - source.x()) * travel,
            source.y() + (target.y() - source.y()) * travel,
            source.width() + (target.width() - source.width()) * travel,
            source.height() + (target.height() - source.height()) * travel,
        )
        return art_rect

    def _draw_artwork_vortex_image(
        self,
        painter: QPainter,
        art_rect: QRectF,
    ) -> None:
        """Draw only the intact cover; animated layers are always dots."""
        artwork = self._artwork_square
        if artwork is None or artwork.isNull():
            return
        handoff = self._style_fx * self._smoothstep01(self._morph / 0.35)
        dot_alpha = self._smoothstep01((self._vortex_dissolve - 0.02) / 0.88)
        full_opacity = handoff * (1.0 - dot_alpha)
        if full_opacity > 0.003:
            painter.save()
            painter.setOpacity(full_opacity)
            painter.drawImage(art_rect, artwork)
            painter.restore()

    def _draw_artwork_vortex(
        self,
        painter: QPainter,
        rect: QRectF,
        levels: AudioLevels,
        img_rect: Optional[QRectF],
    ) -> None:
        """Clean artwork sections crossfade into their matching dot fields."""
        source = self._source_rect(rect, img_rect)
        x, y, z = self._artwork_vortex_geometry()
        radius = min(rect.width(), rect.height()) * 0.46
        perspective = 3.45 / (3.45 - z * 0.82)
        dx = rect.center().x() + x * radius * perspective
        dy = rect.center().y() - y * radius * perspective
        art_rect = self._artwork_vortex_plane_rect(rect, source, radius)
        base_x = art_rect.left() + self._relief_uv[:, 0] * art_rect.width()
        base_y = art_rect.top() + self._relief_uv[:, 1] * art_rect.height()
        # Dots first materialize directly over their source pixels. Only after
        # that clean dissolve do the detected sections begin to separate.
        motion_mix = (
            self._smoothstep01((self._vortex_dissolve - 0.28) / 0.68)
            * self._smoothstep01(self._morph / 0.72)
        )
        px = base_x + (dx - base_x) * motion_mix
        py = base_y + (dy - base_y) * motion_mix
        self._draw_artwork_vortex_image(painter, art_rect)
        dot_alpha = self._smoothstep01((self._vortex_dissolve - 0.02) / 0.88)
        fx_alpha = (
            self._style_fx
            * self._smoothstep01(self._morph / 0.16)
            * dot_alpha
        )
        accent = float(np.clip(
            0.22 + 0.30 * self._music_drive[2] + 0.24 * self._music_drive[3]
            + 0.46 * self._sm_vortex_energy,
            0.0,
            1.0,
        ))
        visibility = self._artwork_vortex_visibility()
        self._draw_relief_cloud(
            painter,
            px,
            py,
            z,
            radius,
            fx_alpha,
            accent,
            visibility,
        )

    def _artwork_vortex_visibility(self) -> np.ndarray:
        """Dissolve foreground then background without beat-driven popping."""
        arrival = 0.04 + 0.70 * self._relief_reveal_threshold
        background = self._relief_region == self._relief_background_region
        arrival = arrival + background.astype(np.float64) * 0.08
        visibility = np.clip(
            (self._vortex_dissolve - arrival) / 0.20,
            0.0,
            1.0,
        )
        return visibility * visibility * (3.0 - 2.0 * visibility)

    def _draw_artwork_relief(
        self,
        painter: QPainter,
        rect: QRectF,
        levels: AudioLevels,
        img_rect: Optional[QRectF],
    ) -> None:
        """High-resolution 3D point relief built from roughly 9,000 art samples."""
        source = self._source_rect(rect, img_rect)
        uv = self._relief_uv
        x = self._relief_x
        y = self._relief_y
        vocal = levels.vocal if levels.source == "full-file" else 0.0
        component = self._music_drive[self._relief_cls % len(self._music_drive)]
        depth = (
            (0.07 + 0.22 * vocal + 0.13 * self._sm_motion)
            * np.sin(x * 6.2 + y * 4.4 - self._flow_phase * 1.6)
            + 0.07 * levels.spectral_flux * np.sin(x * 13.0 - y * 8.0 + self._flow_phase)
        )
        depth += 0.105 * component * np.sin(
            self._relief_phase + self._flow_phase * (1.2 + component)
        )
        depth += (
            0.11 * self._sm_climax + 0.16 * self._sm_drop
        ) * np.sin(self._relief_phase + self._flow_phase * 2.0)
        expansion = np.clip(
            0.90 + 0.20 * self._sm_flow - 0.06 * self._sm_anticipation
            + 0.20 * self._sm_climax + 0.28 * self._sm_drop,
            0.68,
            1.42,
        )
        x = x * expansion
        y = y * expansion

        view_amount = self._smoothstep01((self._morph - 0.16) / 0.68)
        yaw = (
            0.13 * math.sin(self._flow_phase * 0.38) + 0.26 * self._sm_shift
            + self._view_yaw * view_amount
        )
        tilt = 0.07 + 0.18 * self._sm_flow + self._view_pitch * view_amount
        ca, sa = math.cos(yaw), math.sin(yaw)
        ct, st = math.cos(tilt), math.sin(tilt)
        xr = x * ca + depth * sa
        zr = -x * sa + depth * ca
        yr = y * ct - zr * st
        zr = y * st + zr * ct

        radius = min(rect.width(), rect.height()) * 0.52
        perspective = 3.4 / (3.4 - zr * 0.85)
        dx = rect.center().x() + xr * radius * perspective
        dy = rect.center().y() - yr * radius * perspective
        sx = source.left() + uv[:, 0] * source.width()
        sy = source.top() + uv[:, 1] * source.height()
        amount = np.clip(
            self._morph * 1.14 - self._relief_stagger * 0.14,
            0.0,
            1.0,
        )
        amount = amount * amount * (3.0 - 2.0 * amount)
        px = sx + (dx - sx) * amount
        py = sy + (dy - sy) * amount

        fx_alpha = self._style_fx * self._smoothstep01(self._morph / 0.18)
        accent = float(np.clip(
            0.20 + 0.26 * levels.vocal + 0.24 * levels.spectral_flux
            + 0.34 * self._sm_climax + 0.42 * self._sm_drop,
            0.0,
            1.0,
        ))
        self._draw_relief_cloud(
            painter,
            px,
            py,
            zr,
            radius,
            fx_alpha,
            accent,
            highlights_enabled=False,
        )

    # ------------------------------ render

    def render(
        self,
        painter: QPainter,
        rect: QRectF,
        now: float,
        levels: AudioLevels,
        img_rect: Optional[QRectF] = None,
    ) -> None:
        w, h = rect.width(), rect.height()
        if w <= 10 or h <= 10:
            return
        dt = 0.0 if self._last_t is None else max(0.0, min(0.1, now - self._last_t))
        self._last_t = now
        self._style_fx = min(1.0, self._style_fx + dt / 0.45)

        if dt > 0.0 and not self._view_dragging:
            self._view_yaw += self._view_yaw_velocity * dt
            next_pitch = self._view_pitch + self._view_pitch_velocity * dt
            clipped_pitch = float(np.clip(next_pitch, -1.22, 1.22))
            if clipped_pitch != next_pitch:
                self._view_pitch_velocity = 0.0
            self._view_pitch = clipped_pitch
            drag_decay = math.exp(-dt * 2.45)
            self._view_yaw_velocity *= drag_decay
            self._view_pitch_velocity *= drag_decay
            if abs(self._view_yaw_velocity) < 0.004:
                self._view_yaw_velocity = 0.0
            if abs(self._view_pitch_velocity) < 0.004:
                self._view_pitch_velocity = 0.0

        # Morph progress toward the current mode.
        tgt = 1.0 if self._mode_sphere else 0.0
        if self._morph != tgt and dt > 0:
            step = dt / self._MORPH_SEC
            self._morph = min(1.0, self._morph + step) if tgt > self._morph else max(0.0, self._morph - step)
        if self._morph <= 0.0:
            return  # fully artwork: nothing to draw

        # Palette crossfades stay global and clean even for the denser relief.
        if dt > 0:
            k = 1.0 - math.exp(-dt / 0.5)
            diff = self._col_tgt - self._col_cur
            if np.abs(diff).max() < 1.0:
                self._col_cur[:] = self._col_tgt
            else:
                self._col_cur += diff * k
            relief_diff = self._relief_col_tgt - self._relief_col_cur
            if np.abs(relief_diff).max() < 1.0:
                self._relief_col_cur[:] = self._relief_col_tgt
            else:
                self._relief_col_cur += relief_diff * k

        if levels.ok and not levels.silent:
            bass, rms, pulse, high = levels.bass, levels.rms, levels.beat, levels.high
            mid = levels.mid
        else:
            bass = rms = pulse = high = mid = 0.0

        ks = 1.0 - math.exp(-dt / 0.09) if dt else 0.0
        self._sm_bass += (bass - self._sm_bass) * ks
        self._sm_rms += (rms - self._sm_rms) * ks

        # "Dance" energy gates all motion beyond rotation: quiet music barely
        # shimmers, a strong beat makes the dots jump. Eases up fast, settles
        # down slowly so pauses wind down gracefully instead of freezing.
        dance_t = min(1.0, 0.45 * rms + 0.35 * mid + 0.65 * pulse)
        kd = 1.0 - math.exp(-dt / (0.15 if dance_t > self._dance else 0.55)) if dt else 0.0
        self._dance += (dance_t - self._dance) * kd
        if self._dance < 0.004:
            self._dance = 0.0
        dance = self._dance

        # A completed file provides context on both sides of this timestamp.
        # Those cues govern sustained speed, reversal, expansion, and high
        # points. Live mode falls back to the current spectrum and beat.
        full = levels.source == "full-file"
        if full:
            motion_t = levels.music_motion
            flow_t = levels.energy_flow
            shift_t = levels.spectral_shift
            climax_t = max(levels.climax, levels.drop)
            intensity_t = levels.track_intensity
            buildup_t = levels.buildup
            anticipation_t = levels.anticipation
            drop_t = levels.drop
        else:
            motion_t = min(1.0, 0.42 * rms + 0.28 * mid + 0.22 * high + 0.35 * pulse)
            flow_t = 0.0
            shift_t = (high - bass) * 0.12
            climax_t = min(1.0, 0.72 * pulse + 0.22 * rms)
            intensity_t = motion_t
            buildup_t = 0.0
            anticipation_t = 0.0
            drop_t = pulse * 0.55

        def ease_state(current: float, target: float, tau: float) -> float:
            amount = 1.0 - math.exp(-dt / tau) if dt else 0.0
            return current + (target - current) * amount

        self._sm_motion = ease_state(self._sm_motion, motion_t, 0.30)
        self._sm_flow = ease_state(self._sm_flow, flow_t, 0.52)
        self._sm_shift = ease_state(self._sm_shift, shift_t, 0.58)
        self._sm_intensity = ease_state(self._sm_intensity, intensity_t, 0.62)
        self._sm_buildup = ease_state(self._sm_buildup, buildup_t, 0.48)
        self._sm_anticipation = ease_state(self._sm_anticipation, anticipation_t, 0.34)
        self._sm_drop = ease_state(
            self._sm_drop,
            drop_t,
            0.065 if drop_t > self._sm_drop else 0.82,
        )
        self._sm_climax = ease_state(
            self._sm_climax,
            climax_t,
            0.10 if climax_t > self._sm_climax else 0.62,
        )
        vortex_energy_t = float(np.clip(
            0.34 * self._sm_intensity + 0.24 * self._sm_motion
            + 0.16 * self._sm_buildup + 0.18 * self._sm_anticipation
            + 0.34 * self._sm_climax + 0.28 * self._sm_drop,
            0.0,
            1.0,
        ))
        vortex_tau = 0.38 if vortex_energy_t > self._sm_vortex_energy else 0.95
        self._sm_vortex_energy = ease_state(
            self._sm_vortex_energy,
            vortex_energy_t,
            vortex_tau,
        )
        if self._style == 8:
            # Calm sections resolve to the clean cover; strong sections become
            # entirely dots. Long attack/release constants keep the handoff
            # continuous even when a full-file section label changes at once.
            dissolve_target = self._smoothstep01(
                (self._sm_vortex_energy - 0.16) / 0.58
            )
            dissolve_tau = 0.82 if dissolve_target > self._vortex_dissolve else 1.45
            self._vortex_dissolve = ease_state(
                self._vortex_dissolve,
                dissolve_target,
                dissolve_tau,
            )
            if abs(self._vortex_dissolve - dissolve_target) < 0.001:
                self._vortex_dissolve = dissolve_target
        channel_target = self._music_channels(levels)
        if dt > 0:
            channel_tau = np.where(channel_target > self._music_drive, 0.075, 0.36)
            channel_amount = 1.0 - np.exp(-dt / channel_tau)
            self._music_drive += (channel_target - self._music_drive) * channel_amount
        direction = float(np.clip(
            0.40 + 1.00 * self._sm_flow + 0.40 * self._sm_buildup
            + 0.58 * self._sm_shift,
            -1.15,
            1.55,
        ))
        # Even the calmest passage retains a slow continuous drift. Builds
        # smoothly wind the system up; detected jumps briefly release a much
        # faster shared clock without discontinuously changing phase.
        motion_speed = (
            0.16 + 1.20 * self._sm_motion + 0.66 * self._sm_intensity
            + 0.92 * self._sm_anticipation + 1.85 * self._sm_drop
            + 0.18 * rms
        )
        self._flow_phase += dt * motion_speed * direction
        self._ang += dt * (
            0.11 + 0.62 * self._sm_motion + 0.38 * self._sm_intensity
            + 0.55 * self._sm_anticipation + 1.20 * self._sm_drop
            + 0.16 * pulse
        ) * direction

        custom_style = self._style in (2, 4, 7, 9)
        if self._style == 7:
            self._update_wave_history(levels, dt)
            self._draw_crt_wavefield(painter, rect, levels, img_rect)
        elif self._style == 2:
            self._draw_chroma_ribbons(painter, rect, levels, img_rect)
        elif self._style == 4:
            self._draw_liquid_orbit(painter, rect, levels, img_rect)
        elif self._style == 5:
            self._draw_harmonic_knot(painter, rect, levels, img_rect)
        elif self._style == 8:
            self._draw_artwork_vortex(painter, rect, levels, img_rect)
            return
        elif self._style == 9:
            self._draw_artwork_relief(painter, rect, levels, img_rect)
            return

        # The custom line/polygon modes use the original dots only while the
        # artwork is dissolving. At steady state there is no 4,000-dot sort or
        # draw cost, which is especially important for the CRT wavefield.
        particle_alpha = 1.0
        if custom_style:
            particle_alpha = 1.0 - self._smoothstep01((self._morph - 0.08) / 0.56)
            if particle_alpha <= 0.003:
                return

        view_amount = self._smoothstep01((self._morph - 0.16) / 0.68)
        if self._style == 7:
            yaw = 0.06 * math.sin(self._flow_phase * 0.18) + self._view_yaw * view_amount
            tilt = 0.68 + self._view_pitch * view_amount
        else:
            yaw = self._ang + self._view_yaw * view_amount
            tilt = (
                0.32 + 0.08 * self._sm_shift + 0.05 * math.sin(now * 0.11)
                + self._view_pitch * view_amount
            )

        ca, sa = math.cos(yaw), math.sin(yaw)
        ct, st = math.cos(tilt), math.sin(tilt)
        m = (np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
             @ np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]]))

        jit = self._cls_jitter[self._cls]
        burst = self._cls_burst[self._cls] * self._burst_i
        component = self._music_drive[self._feature_lane]
        disp = (
            self._shell
            + jit * np.sin(now * self._jit_speed + self._jit_phase) * 6.0 * dance
            + 0.085 * component * np.sin(
                self._jit_phase + self._flow_phase * (1.8 + self._feature_lane * 0.12)
            )
            + (
                0.10 * pulse + 0.05 * high + 0.11 * self._sm_climax
                + 0.20 * self._sm_drop + 0.08 * component
            ) * burst
        )
        target_pts = self._target_geometry(now, levels)
        shape_k = 1.0 - math.exp(-dt / 0.32) if dt else 0.0
        self._shape_pts += (target_pts - self._shape_pts) * shape_k
        pts = (self._shape_pts * disp[:, None]) @ m.T

        R = 0.5 * min(w, h) * (
            (0.70 if self._style == 1 else (0.72 if self._style == 7 else 0.80))
            + 0.085 * self._sm_bass + 0.05 * pulse + 0.03 * levels.energy_ahead
            + 0.05 * self._sm_flow + 0.05 * self._sm_climax
            - 0.025 * self._sm_anticipation + 0.11 * self._sm_drop
        )
        if self._style == 1 and full:
            sphere_scale = np.clip(
                1.0 - 0.36 * self._sm_anticipation
                - 0.20 * max(0.0, -self._sm_flow)
                + 0.16 * self._sm_drop + 0.04 * self._sm_climax,
                0.38,
                1.18,
            )
            R *= float(sphere_scale)
        z = pts[:, 2]
        persp = 3.2 / (3.2 - z * 0.9)
        cx, cy = rect.center().x(), rect.center().y()
        sx = cx + pts[:, 0] * R * persp
        sy = cy - pts[:, 1] * R * persp

        # Blend from artwork home positions while morphing.
        mph = self._morph
        if mph < 1.0:
            src = self._source_rect(rect, img_rect)
            ix = src.left() + self._img_uv[:, 0] * src.width()
            iy = src.top() + self._img_uv[:, 1] * src.height()
            s_amt = 0.35
            mi = np.clip((mph * (1.0 + s_amt) - self._stagger * s_amt), 0.0, 1.0)
            mi = mi * mi * (3.0 - 2.0 * mi)  # smoothstep per dot
            sx = ix + (sx - ix) * mi
            sy = iy + (sy - iy) * mi

        dnorm = (z + 1.0) * 0.5
        depth_band = np.digitize(dnorm, (0.36, 0.68))
        flash = 1.0 + 0.85 * pulse + (
            0.40 * levels.vocal + 0.42 * levels.section_change
            + 0.28 * self._sm_climax + 0.55 * self._sm_drop
            if full else 0.0
        )
        base_px = max(1.0 if self._style == 7 else 1.5, R * (0.0045 if self._style == 7 else 0.0088))

        group = (self._cls * 3 + depth_band) * 2 + self._size_sub
        order = np.argsort(group, kind="stable")
        g_sorted = group[order]
        n_groups = self.N_CLASSES * 6
        bounds = np.searchsorted(g_sorted, np.arange(n_groups + 1))

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        for gidx in range(n_groups):
            lo, hi = bounds[gidx], bounds[gidx + 1]
            if hi <= lo:
                continue
            idx = order[lo:hi]
            cls = gidx // 6
            band = (gidx // 2) % 3
            big = gidx % 2

            rgb = self._col_cur[cls]
            col = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            # Depth shading applies only as we become a sphere.
            band_a = 0.92 + (self._DEPTH_ALPHA[band] - 0.92) * mph
            a = (
                band_a * self._cls_alpha[cls] * (1.0 if big else 0.82)
                * flash * particle_alpha
            )
            col.setAlpha(int(np.clip(255.0 * a, 0.0, 255.0)))
            pen = QPen(col)
            pen.setCapStyle(Qt.RoundCap)
            size_k = self._cls_size[cls] * (1.45 if big else 0.95) * (0.78 + 0.30 * band)
            pen.setWidthF(base_px * size_k)
            painter.setPen(pen)
            painter.drawPoints(np_to_qpolygonf(sx[idx], sy[idx]))
        painter.restore()


class AsyncSphereVisualizer:
    """Latest-frame-only worker for foreground animation rendering.

    QWidget painting remains on Qt's main thread, but the expensive particle,
    mesh, sorting, and offscreen QImage painting run here. Requests never pile
    up: if the UI submits faster than the worker can render, only the newest
    frame survives. Audio capture has a higher macOS QoS class than this
    ordinary worker thread.
    """

    STYLE_NAMES = SphereVisualizer.STYLE_NAMES

    def __init__(self, n_dots: int = VIS_SPHERE_DOTS) -> None:
        self._n_dots = int(n_dots)
        self._commands: "queue.SimpleQueue[tuple[str, tuple]]" = queue.SimpleQueue()
        self._requests: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=1)
        self._latest_lock = threading.Lock()
        self._latest_image: Optional[QImage] = None
        self._latest_morph = 0.0
        self._latest_bg = QColor(5, 5, 9)
        self._style = 1
        self._mode = False
        self._dragging = False
        self._view_yaw = 0.0
        self._view_pitch = 0.0
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            name="visual-render-worker",
            daemon=True,
        )
        self._thread.start()

    def _command(self, name: str, *args) -> None:
        if not self._stopped:
            self._commands.put((name, args))

    def set_mode(self, sphere: bool) -> None:
        self._mode = bool(sphere)
        self._command("set_mode", self._mode)

    def set_style(self, style: int) -> None:
        self._style = max(1, min(9, int(style)))
        self._command("set_style", self._style)

    def style(self) -> int:
        return self._style

    def style_name(self) -> str:
        return self.STYLE_NAMES[self._style]

    def set_artwork(self, img: Optional[QImage]) -> None:
        owned = None if img is None or img.isNull() else img.copy()
        self._command("set_artwork", owned)

    def morph_value(self) -> float:
        with self._latest_lock:
            return float(self._latest_morph)

    def bg_color(self) -> QColor:
        with self._latest_lock:
            return QColor(self._latest_bg)

    def begin_drag(self) -> None:
        self._dragging = True
        self._command("begin_drag")

    def drag_view(self, dx: float, dy: float, dt: float) -> None:
        self._view_yaw += float(dx) * 0.0082
        self._view_pitch = float(np.clip(
            self._view_pitch + float(dy) * 0.0072,
            -1.22,
            1.22,
        ))
        self._command("drag_view", float(dx), float(dy), float(dt))

    def end_drag(self) -> None:
        self._dragging = False
        self._command("end_drag")

    def is_dragging(self) -> bool:
        return self._dragging

    def render(
        self,
        painter: QPainter,
        rect: QRectF,
        now: float,
        levels: AudioLevels,
        img_rect: Optional[QRectF] = None,
    ) -> None:
        if self._stopped or rect.width() <= 10 or rect.height() <= 10:
            return
        device = painter.device()
        dpr = float(device.devicePixelRatioF()) if device is not None else 1.0
        request = (
            float(rect.width()),
            float(rect.height()),
            max(1.0, dpr),
            float(now),
            levels,
            (
                None
                if img_rect is None
                else QRectF(
                    img_rect.left() - rect.left(),
                    img_rect.top() - rect.top(),
                    img_rect.width(),
                    img_rect.height(),
                )
            ),
        )
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                pass
            try:
                self._requests.put_nowait(request)
            except queue.Full:
                pass

        with self._latest_lock:
            image = self._latest_image
        if image is not None and not image.isNull():
            painter.save()
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawImage(rect, image)
            painter.restore()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                pass
            try:
                self._requests.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=3.0)

    def _drain_commands(self, visualizer: SphereVisualizer) -> None:
        while True:
            try:
                name, args = self._commands.get_nowait()
            except queue.Empty:
                return
            getattr(visualizer, name)(*args)

    def _run(self) -> None:
        visualizer = SphereVisualizer(self._n_dots)
        while True:
            try:
                request = self._requests.get(timeout=0.1)
            except queue.Empty:
                self._drain_commands(visualizer)
                if self._stopped:
                    break
                continue
            if request is None:
                break
            self._drain_commands(visualizer)
            width, height, dpr, now, levels, local_img_rect = request
            max_logical = max(1.0, width, height)
            render_dpr = max(1.0, min(
                dpr,
                float(VIS_FOREGROUND_RENDER_MAX_PX) / max_logical,
            ))
            pixel_w = max(1, int(math.ceil(width * render_dpr)))
            pixel_h = max(1, int(math.ceil(height * render_dpr)))
            image = QImage(pixel_w, pixel_h, QImage.Format_ARGB32_Premultiplied)
            image.setDevicePixelRatio(render_dpr)
            image.fill(Qt.transparent)
            worker_painter = QPainter(image)
            worker_painter.setRenderHint(QPainter.Antialiasing, True)
            visualizer.render(
                worker_painter,
                QRectF(0.0, 0.0, width, height),
                now,
                levels,
                local_img_rect,
            )
            worker_painter.end()
            with self._latest_lock:
                self._latest_image = image
                self._latest_morph = visualizer.morph_value()
                self._latest_bg = visualizer.bg_color()
