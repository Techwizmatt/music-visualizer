from __future__ import annotations

import base64
import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from PySide6.QtGui import QImage

from settings import KEEP_LAST_GOOD_SECONDS


def _require_cmd(cmd: str) -> str:
    path = shutil.which(cmd)
    if not path:
        raise RuntimeError(f'"{cmd}" not found on PATH. Install with: brew install media-control')
    return path


def _coerce_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _coerce_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "1", "playing"}:
            return True
        if s in {"false", "no", "0", "paused", "stopped"}:
            return False
    return None


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return str(v)
    except Exception:
        return None


def _parse_iso_epoch(v: Any) -> Optional[float]:
    if not v or not isinstance(v, str):
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


@dataclass(frozen=True)
class NowPlaying:
    title: Optional[str]
    artist: Optional[str]
    album: Optional[str]
    genre: Optional[str]
    track_number: Optional[int]
    total_track_count: Optional[int]
    duration_seconds: Optional[float]
    position_seconds: Optional[float]   # position at `position_epoch` (NOT "now")
    position_epoch: Optional[float]     # wall-clock time the position refers to
    playback_rate: Optional[float]
    is_playing: Optional[bool]
    artwork_bytes: Optional[bytes]
    raw: Dict[str, Any]

    def position_now(self, now_epoch: Optional[float] = None) -> Optional[float]:
        """
        Current playback position, extrapolated from the reported
        (elapsedTime, timestamp) pair. MediaRemote only refreshes elapsedTime
        on events (play/pause/seek/track change), so the raw value can be
        arbitrarily stale — the timestamp delta is essential.
        """
        if self.position_seconds is None:
            return None
        pos = float(self.position_seconds)
        if self.is_playing and self.position_epoch is not None:
            now_epoch = time.time() if now_epoch is None else now_epoch
            rate = self.playback_rate if (self.playback_rate or 0) > 0 else 1.0
            pos += max(0.0, now_epoch - self.position_epoch) * rate
        if self.duration_seconds is not None and self.duration_seconds > 0:
            pos = min(pos, float(self.duration_seconds))
        return max(0.0, pos)


def _extract_artwork_bytes(payload: Dict[str, Any]) -> Optional[bytes]:
    v = payload.get("artworkData")
    if not v or not isinstance(v, str):
        return None
    s = v.strip()
    if "," in s and s.lower().startswith("data:"):
        s = s.split(",", 1)[1].strip()
    missing = (-len(s)) % 4
    if missing:
        s += "=" * missing
    try:
        return base64.b64decode(s, validate=False)
    except Exception:
        return None


def _unwrap_media_control(payload: Dict[str, Any]) -> Dict[str, Any]:
    inner = payload.get("payload")
    if isinstance(inner, dict) and inner:
        return inner
    return payload


def _parse_media_control_payload(payload: Dict[str, Any]) -> NowPlaying:
    title = payload.get("title") or payload.get("name")
    artist = payload.get("artist")
    album = payload.get("album")
    genre = payload.get("genre")

    track_number = _as_int(payload.get("trackNumber"))
    total_track_count = _as_int(payload.get("totalTrackCount"))

    duration = payload.get("durationSeconds") or payload.get("duration") or payload.get("trackDuration")

    position = payload.get("elapsedTime")
    if position is None:
        position = (
            payload.get("positionSeconds")
            or payload.get("elapsed")
            or payload.get("position")
            or payload.get("playbackTime")
        )
    position_epoch = _parse_iso_epoch(payload.get("timestamp"))

    rate = _coerce_float(payload.get("playbackRate"))

    playing = payload.get("playing")
    if playing is None:
        playing = payload.get("isPlaying")
    playing = _coerce_bool(playing)
    if playing is None and rate is not None:
        playing = bool(rate)

    artwork_bytes = _extract_artwork_bytes(payload)

    return NowPlaying(
        title=_as_str(title),
        artist=_as_str(artist),
        album=_as_str(album),
        genre=_as_str(genre),
        track_number=track_number,
        total_track_count=total_track_count,
        duration_seconds=_coerce_float(duration),
        position_seconds=_coerce_float(position),
        position_epoch=position_epoch,
        playback_rate=rate,
        is_playing=playing,
        artwork_bytes=artwork_bytes,
        raw=payload,
    )


def get_now_playing_once() -> NowPlaying:
    media_control = _require_cmd("media-control")
    p = subprocess.run(
        [media_control, "get"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"media-control get failed: {p.stderr.strip()}")

    try:
        payload_any: Any = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"media-control get returned non-JSON output: {e}")

    if not isinstance(payload_any, dict):
        raise RuntimeError(f"Unexpected JSON shape from media-control get: {type(payload_any)}")

    payload = _unwrap_media_control(payload_any)
    if not isinstance(payload, dict):
        raise RuntimeError("Unwrapped payload is not a dict")

    return _parse_media_control_payload(payload)


# ---------- streaming (push updates, incl. artwork) ----------


class MediaControlStream:
    """
    Persistent `media-control stream` reader.

    `media-control get` frequently omits artworkData (MediaRemote publishes
    artwork asynchronously), so polling alone can miss it forever. The stream
    pushes full payloads + diffs, including artwork whenever it materializes.
    Payloads are merged into a running dict and forwarded to the callback as
    parsed NowPlaying objects. Auto-restarts if the process dies.
    """

    _IDENTITY_KEYS = ("uniqueIdentifier", "contentItemIdentifier", "title")

    def __init__(self, on_update) -> None:
        self._on_update = on_update
        self._stop = False
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._merged: Dict[str, Any] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="media-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop:
            try:
                exe = _require_cmd("media-control")
                self._proc = subprocess.Popen(
                    [exe, "stream"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
                assert self._proc.stdout is not None
                self._merged = {}
                for line in self._proc.stdout:
                    if self._stop:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(msg, dict):
                        continue
                    payload = msg.get("payload")
                    if not isinstance(payload, dict):
                        continue

                    if msg.get("diff") is True:
                        # A diff that changes track identity but carries no new
                        # artwork means the retained artwork is stale — drop it.
                        if "artworkData" not in payload and any(
                            k in payload for k in self._IDENTITY_KEYS
                        ):
                            self._merged.pop("artworkData", None)
                        self._merged.update(payload)
                        # None values mean "field removed".
                        self._merged = {k: v for k, v in self._merged.items() if v is not None}
                    else:
                        self._merged = dict(payload)

                    if not self._merged:
                        continue
                    try:
                        self._on_update(_parse_media_control_payload(self._merged))
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                proc, self._proc = self._proc, None
                if proc is not None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            if not self._stop:
                time.sleep(2.0)


# ---------- playback commands (fire and forget) ----------

_COMMANDS = {
    "play", "pause", "toggle-play-pause", "next-track", "previous-track",
    "toggle-shuffle", "toggle-repeat",
}


def send_command(cmd: str) -> None:
    if cmd not in _COMMANDS:
        return
    try:
        exe = _require_cmd("media-control")
        subprocess.Popen([exe, cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def seek(position_sec: float) -> None:
    try:
        exe = _require_cmd("media-control")
        subprocess.Popen(
            [exe, "seek", f"{max(0.0, float(position_sec)):.2f}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _track_key(np_: NowPlaying) -> str:
    uid = _as_str(np_.raw.get("uniqueIdentifier"))
    return "|".join(
        [
            uid or "",
            np_.title or "",
            np_.artist or "",
            np_.album or "",
            str(np_.duration_seconds or ""),
        ]
    )


class NowPlayingState:
    """
    Thread-safe-ish holder for the latest poll result. Position is anchored to
    a monotonic clock and refreshed from each poll's extrapolated position, so
    the UI can render at 60fps between 250ms polls without stutter.
    """

    def __init__(self) -> None:
        self._np: Optional[NowPlaying] = None
        self._np_last_good_mono: float = 0.0

        self._last_track_key: Optional[str] = None

        self._pos_anchor: float = 0.0
        self._pos_anchor_mono: float = time.monotonic()
        self._is_playing: Optional[bool] = None

        self._last_error: Optional[str] = None

        self._artwork_image: Optional[QImage] = None
        self._artwork_sig: Optional[int] = None
        self._fallback_art: Optional[QImage] = None
        self._fallback_art_key: Optional[str] = None

        # Optimistic play/pause flip while a toggle command is in flight.
        self._optimistic_until_mono: float = 0.0
        self._optimistic_playing: Optional[bool] = None

    # ----- internal -----

    def _estimated_pos(self, now_m: float) -> float:
        pos = self._pos_anchor
        if self._is_playing is True:
            pos += max(0.0, now_m - self._pos_anchor_mono)
        if self._np is not None and self._np.duration_seconds:
            pos = min(pos, float(self._np.duration_seconds))
        return max(0.0, pos)

    # ----- called from poll worker thread -----

    def update_from_poll(self, incoming: NowPlaying) -> None:
        now_m = time.monotonic()

        is_empty = (
            incoming.title is None
            and incoming.artist is None
            and incoming.album is None
            and incoming.duration_seconds is None
            and incoming.position_seconds is None
            and incoming.is_playing is None
            and not incoming.raw
        )
        if is_empty and self._np is not None and (now_m - self._np_last_good_mono) <= KEEP_LAST_GOOD_SECONDS:
            return

        key = _track_key(incoming)
        track_changed = bool(key.strip("|")) and (self._last_track_key != key)

        self._np = incoming
        self._np_last_good_mono = now_m
        self._last_error = None

        # Artwork: adopt whenever bytes are present and new. MediaRemote often
        # omits artwork from early polls, so keep accepting it late — never
        # clear an image just because one payload lacked bytes.
        if track_changed:
            self._artwork_image = None
            self._artwork_sig = None
        if incoming.artwork_bytes:
            sig = len(incoming.artwork_bytes)
            if self._artwork_image is None or sig != self._artwork_sig:
                img = QImage.fromData(incoming.artwork_bytes)
                if not img.isNull():
                    self._artwork_image = img
                    self._artwork_sig = sig

        reported = incoming.position_now()
        playing = incoming.is_playing

        if playing is not None and playing == self._optimistic_playing:
            self._optimistic_playing = None
            self._optimistic_until_mono = 0.0

        if track_changed:
            self._last_track_key = key
            self._pos_anchor = reported if reported is not None else 0.0
            self._pos_anchor_mono = now_m
            self._is_playing = playing
            return

        state_flipped = playing is not None and playing != self._is_playing
        if reported is not None:
            est = self._estimated_pos(now_m)
            err = reported - est
            if state_flipped or abs(err) > 1.5:
                # Real discontinuity (seek/pause/track event): snap.
                self._pos_anchor = reported
                self._pos_anchor_mono = now_m
            elif abs(err) > 0.15:
                # MediaRemote timestamps are whole-second quantized, so small
                # disagreements are noise — slew toward the report instead of
                # snapping, or the active lyric line visibly twitches.
                self._pos_anchor = est + err * 0.18
                self._pos_anchor_mono = now_m
        if playing is not None:
            self._is_playing = playing

    def set_error(self, msg: str) -> None:
        self._last_error = msg

    # ----- called from UI -----

    def note_optimistic_playing(self, playing: bool) -> None:
        """Make the play/pause button feel instant while the poll catches up."""
        self._optimistic_playing = playing
        self._optimistic_until_mono = time.monotonic() + 1.5
        if self._is_playing is not None and self._is_playing != playing:
            now_m = time.monotonic()
            self._pos_anchor = self._estimated_pos(now_m)
            self._pos_anchor_mono = now_m
            self._is_playing = playing

    def effective_playing(self) -> Optional[bool]:
        if self._optimistic_playing is not None and time.monotonic() < self._optimistic_until_mono:
            return self._optimistic_playing
        return self._is_playing

    def set_fallback_artwork(self, key: str, img: QImage) -> None:
        """High-res artwork fetched from iTunes; used until MediaRemote's own
        artwork (if any) arrives."""
        if key == self._last_track_key:
            self._fallback_art = img
            self._fallback_art_key = key

    def snapshot(self) -> Tuple[Optional[NowPlaying], float, Optional[QImage], Optional[str], str]:
        np_ = self._np
        err = self._last_error
        key = self._last_track_key or ""
        art = self._artwork_image
        fb = self._fallback_art if self._fallback_art_key == key else None
        if art is None:
            art = fb
        elif fb is not None and fb.width() > art.width() * 1.25:
            art = fb  # the iTunes fetch is meaningfully sharper — prefer it
        if np_ is None:
            return None, 0.0, art, err, key
        pos = self._estimated_pos(time.monotonic())
        return np_, float(pos), art, err, key
