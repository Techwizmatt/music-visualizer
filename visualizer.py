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
import random
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
    Particle globe in the artwork's own colors, weighted by how common each
    color actually is: the most common color fills the sphere as small, calm
    dots; rarer colors get fewer but bigger, livelier dots that pop with the
    beat. Class colors crossfade smoothly when the track changes.

    Also owns the artwork<->sphere morph: every dot has a "home" position
    sampled from the album art (matched by color), so toggling the mode
    dissolves the artwork into the globe and back.

    All math is numpy-vectorized; dots are drawn in (class, depth, size)
    buckets with round-cap pens — thousands of dots at 60fps.
    """

    N_CLASSES = 6
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

        # Fixed class membership (colors change per artwork, membership never
        # does — that keeps color transitions perfectly clean).
        shares = np.array([0.50, 0.20, 0.12, 0.08, 0.06, 0.04])
        self._cls = rng.choice(self.N_CLASSES, size=n, p=shares)

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

        # Morph state: 0 = artwork, 1 = sphere.
        self._morph = 0.0
        self._mode_sphere = False
        self._img_uv = np.column_stack([rng.random(n), rng.random(n)])

        self._ang = 0.0
        self._last_t: Optional[float] = None
        self._sm_bass = 0.0
        self._sm_rms = 0.0
        self._dance = 0.0
        self._bg = QColor(5, 5, 9)
        self.set_artwork(None)

    # ------------------------------ mode / morph

    def set_mode(self, sphere: bool) -> None:
        self._mode_sphere = bool(sphere)

    def morph_value(self) -> float:
        return self._morph

    # ------------------------------ palette / artwork

    def set_artwork(self, img: Optional[QImage]) -> None:
        pal = weighted_palette(img, self.N_CLASSES)
        for c in range(self.N_CLASSES):
            col = pal[c][0]
            self._col_tgt[c] = (col.red(), col.green(), col.blue())
        if not self._have_colors:
            self._col_cur[:] = self._col_tgt
            self._have_colors = True

        avg = pal[0][0]
        self._bg = QColor(
            int(4 + avg.red() * 0.030),
            int(4 + avg.green() * 0.030),
            int(8 + avg.blue() * 0.045),
        )
        self._assign_home_positions(img)

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

    def bg_color(self) -> QColor:
        return QColor(self._bg)

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

        # Morph progress toward the current mode.
        tgt = 1.0 if self._mode_sphere else 0.0
        if self._morph != tgt and dt > 0:
            step = dt / self._MORPH_SEC
            self._morph = min(1.0, self._morph + step) if tgt > self._morph else max(0.0, self._morph - step)
        if self._morph <= 0.0:
            return  # fully artwork: nothing to draw

        # Class color crossfade (clean, global, cheap).
        if dt > 0:
            k = 1.0 - math.exp(-dt / 0.5)
            diff = self._col_tgt - self._col_cur
            if np.abs(diff).max() < 1.0:
                self._col_cur[:] = self._col_tgt
            else:
                self._col_cur += diff * k

        if levels.ok and not levels.silent:
            # `beat` is the predicted, latency-compensated pulse — dots land
            # ON the beat instead of trailing it.
            bass, rms, pulse, high = levels.bass, levels.rms, levels.beat, levels.high
            mid = levels.mid
        else:
            # No music: the globe just spins, perfectly rigid and calm.
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

        self._ang += dt * (0.22 + 0.50 * self._sm_rms + 0.55 * pulse)
        tilt = 0.32 + 0.06 * math.sin(now * 0.11)

        ca, sa = math.cos(self._ang), math.sin(self._ang)
        ct, st = math.cos(tilt), math.sin(tilt)
        m = (np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
             @ np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]]))

        jit = self._cls_jitter[self._cls]
        burst = self._cls_burst[self._cls] * self._burst_i
        disp = (
            self._shell
            + jit * np.sin(now * self._jit_speed + self._jit_phase) * 6.0 * dance
            + (0.10 * pulse + 0.05 * high) * burst
        )
        pts = (self._base * disp[:, None]) @ m.T

        # energy_ahead: cached profile knows a loud section is coming — the
        # globe pre-swells slightly into drops on repeat plays.
        R = 0.5 * min(w, h) * (
            0.80 + 0.085 * self._sm_bass + 0.05 * pulse + 0.03 * levels.energy_ahead
        )
        z = pts[:, 2]
        persp = 3.2 / (3.2 - z * 0.9)
        cx, cy = rect.center().x(), rect.center().y()
        sx = cx + pts[:, 0] * R * persp
        sy = cy - pts[:, 1] * R * persp

        # Blend from artwork home positions while morphing.
        mph = self._morph
        if mph < 1.0:
            src = img_rect if img_rect is not None else rect
            ix = src.left() + self._img_uv[:, 0] * src.width()
            iy = src.top() + self._img_uv[:, 1] * src.height()
            s_amt = 0.35
            mi = np.clip((mph * (1.0 + s_amt) - self._stagger * s_amt), 0.0, 1.0)
            mi = mi * mi * (3.0 - 2.0 * mi)  # smoothstep per dot
            sx = ix + (sx - ix) * mi
            sy = iy + (sy - iy) * mi

        dnorm = (z + 1.0) * 0.5
        depth_band = np.digitize(dnorm, (0.36, 0.68))

        flash = 1.0 + 0.85 * pulse
        base_px = max(1.5, R * 0.0088)  # finer grain now that dots are denser

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
            a = band_a * self._cls_alpha[cls] * (1.0 if big else 0.82) * flash
            col.setAlpha(int(np.clip(255.0 * a, 0.0, 255.0)))
            pen = QPen(col)
            pen.setCapStyle(Qt.RoundCap)
            size_k = self._cls_size[cls] * (1.45 if big else 0.95) * (0.78 + 0.30 * band)
            pen.setWidthF(base_px * size_k)
            painter.setPen(pen)
            painter.drawPoints(np_to_qpolygonf(sx[idx], sy[idx]))
        painter.restore()
