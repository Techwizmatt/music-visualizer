from __future__ import annotations

"""
High-quality album artwork fallback via the free, keyless iTunes Search API.

MediaRemote (media-control) usually supplies artwork, but not always. When a
track has been artwork-less for a moment, this fetcher searches iTunes,
matches the result by title/artist/duration, and upgrades Apple's 100x100
thumbnail URL to a high-resolution rendition (the mzstatic image server
scales to any requested size — 1200x1200 verified, 600x600 as fallback).
"""

import threading
import time
import urllib.parse
import urllib.request
import json
from typing import Dict, Optional

from PySide6.QtGui import QImage

from lyrics_providers import score_candidate, Track
from now_playing import NowPlayingState
from settings import LYRICS_USER_AGENT

_SIZES = ("1200x1200bb", "600x600bb")
_RETRY_SEC = 20.0
_CACHE_MAX = 24


class ArtworkFetcher:
    def __init__(self, state: NowPlayingState) -> None:
        self._state = state
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._attempt_key: str = ""
        self._attempt_t: float = 0.0
        self._cache: Dict[str, QImage] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="artwork-fetcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------ internals

    def _run(self) -> None:
        while not self._stop:
            time.sleep(0.7)
            try:
                self._tick()
            except Exception:
                pass

    def _tick(self) -> None:
        np_, _pos, art, _err, key = self._state.snapshot()
        if np_ is None or not key:
            return
        # Fetch when there's no artwork at all, or only a low-res rendition
        # (MediaRemote often hands out small covers; iTunes has 1200px ones).
        if art is not None and art.width() >= 700:
            return
        if not np_.title or not np_.artist:
            return

        sig = f"{np_.title}|||{np_.artist}|||{np_.album or ''}"
        cached = self._cache.get(sig)
        if cached is not None:
            self._state.set_fallback_artwork(key, cached)
            return

        now = time.monotonic()
        if key == self._attempt_key and (now - self._attempt_t) < _RETRY_SEC:
            return
        self._attempt_key = key
        self._attempt_t = now

        img = self._fetch(np_.title, np_.artist, np_.album or "", np_.duration_seconds or 0.0)
        if img is not None and not img.isNull():
            if len(self._cache) >= _CACHE_MAX:
                self._cache.pop(next(iter(self._cache)))
            self._cache[sig] = img
            self._state.set_fallback_artwork(key, img)

    def _http(self, url: str, timeout: float = 8.0) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": LYRICS_USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _fetch(self, title: str, artist: str, album: str, duration: float) -> Optional[QImage]:
        term = urllib.parse.quote_plus(f"{artist} {title}")
        url = f"https://itunes.apple.com/search?term={term}&entity=song&limit=8"
        try:
            data = json.loads(self._http(url).decode("utf-8", errors="replace"))
        except Exception:
            return None

        track = Track(title=title, artist=artist, album=album, duration_sec=int(round(duration)))
        best_url, best_score = None, 0.0
        for r in data.get("results", []) or []:
            if not isinstance(r, dict):
                continue
            dur_ms = float(r.get("trackTimeMillis") or 0)
            score = score_candidate(
                track,
                str(r.get("trackName") or ""),
                str(r.get("artistName") or ""),
                (dur_ms / 1000.0) if dur_ms > 0 else None,
            )
            if score > best_score and r.get("artworkUrl100"):
                best_score = score
                best_url = str(r["artworkUrl100"])
        if not best_url:
            return None

        for size in _SIZES:
            try:
                raw = self._http(best_url.replace("100x100bb", size), timeout=10.0)
                img = QImage.fromData(raw)
                if not img.isNull():
                    return img
            except Exception:
                continue
        return None
