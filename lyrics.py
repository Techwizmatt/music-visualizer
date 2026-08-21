from __future__ import annotations

"""
LyricsManager: multi-source synced lyrics with caching and retries.

Fetch strategy per track:
  stage 1: LRCLIB /api/get (exact match)  -> LRCLIB /api/search (fuzzy, scored)
  stage 2: NetEase + Kugou raced in a small thread pool (only if stage 1
           produced no synced lyrics)

Outcomes:
  synced        -> cached, shown
  plain only    -> cached (UI stays in centered mode), no retries
  instrumental  -> cached, no retries
  nothing       -> retried on a backoff schedule, then negative-cached
"""

import concurrent.futures
import json
import os
import queue
import tempfile
import time
from dataclasses import dataclass, field
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

from lyrics_providers import (
    LyricsResult,
    Track,
    fetch_kugou,
    fetch_lrclib_get,
    fetch_lrclib_search,
    fetch_netease,
    fetch_qq,
)
from settings import (
    LYRICS_CACHE_MAX_ENTRIES,
    LYRICS_CACHE_PATH,
    LYRICS_MISS_TTL_SEC,
    LYRICS_RETRY_SCHEDULE_SEC,
)


def make_sig(title: str, artist: str, album: str, duration_sec: int) -> str:
    return f"{title}|||{artist}|||{album}|||{duration_sec}"


@dataclass
class LyricsState:
    status: str = "idle"            # idle|cache|searching|retry|ok|plain|none|instrumental
    status_detail: str = ""
    has_synced: bool = False
    instrumental: bool = False
    source: str = ""
    lines: List[Tuple[float, str]] = field(default_factory=list)
    plain: str = ""
    last_track_sig: str = ""
    attempt: int = 0
    next_retry_t: float = 0.0
    fetching: bool = False

    @property
    def resolved(self) -> bool:
        """True when we have a definitive answer for this track."""
        return self.status in ("ok", "cache", "plain", "none", "instrumental")


class LyricsManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._state = LyricsState()
        self._q: "queue.Queue[Optional[Track]]" = queue.Queue(maxsize=8)
        self._stop = False
        self._thread: Optional[Thread] = None
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="lyrics-src"
        )

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        t = Thread(target=self._run, name="lyrics-fetcher", daemon=True)
        self._thread = t
        t.start()

    def stop(self) -> None:
        self._stop = True
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------ public API

    def request_for_track(self, title: str, artist: str, album: str, duration_sec: float) -> None:
        title = (title or "").strip()
        artist = (artist or "").strip()
        album = (album or "").strip()
        dur_i = int(round(max(0.0, float(duration_sec or 0.0))))
        if not title or not artist or dur_i <= 0:
            with self._lock:
                self._state = LyricsState(status="idle")
            return

        sig = make_sig(title, artist, album, dur_i)
        with self._lock:
            if sig == self._state.last_track_sig:
                return  # already handled (any state incl. retry/backoff)
            self._state = LyricsState(status="searching", last_track_sig=sig, fetching=True)

        track = Track(title=title, artist=artist, album=album, duration_sec=dur_i)

        # Cache first (positive, plain, instrumental, or fresh negative).
        cached = self._load_cached(sig)
        if cached is not None:
            with self._lock:
                if self._state.last_track_sig == sig:
                    self._state = cached
            if cached.resolved:
                return

        self._enqueue(track)

    def snapshot(self) -> LyricsState:
        with self._lock:
            s = self._state
            return LyricsState(
                status=s.status,
                status_detail=s.status_detail,
                has_synced=s.has_synced,
                instrumental=s.instrumental,
                source=s.source,
                lines=list(s.lines),
                plain=s.plain,
                last_track_sig=s.last_track_sig,
                attempt=s.attempt,
                next_retry_t=s.next_retry_t,
                fetching=s.fetching,
            )

    # ------------------------------------------------------------ cache

    def _read_cache(self) -> Dict[str, Any]:
        try:
            with open(LYRICS_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _write_cache(self, cache: Dict[str, Any]) -> None:
        if len(cache) > LYRICS_CACHE_MAX_ENTRIES:
            items = sorted(
                cache.items(), key=lambda kv: float((kv[1] or {}).get("saved_at", 0.0))
            )
            for k, _ in items[: len(cache) - LYRICS_CACHE_MAX_ENTRIES]:
                cache.pop(k, None)
        try:
            d = os.path.dirname(os.path.abspath(LYRICS_CACHE_PATH)) or "."
            fd, tmp = tempfile.mkstemp(prefix=".lyrics_cache_", dir=d)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=1)
            os.replace(tmp, LYRICS_CACHE_PATH)
        except Exception:
            pass

    def _load_cached(self, sig: str) -> Optional[LyricsState]:
        entry = self._read_cache().get(sig)
        if not isinstance(entry, dict):
            return None

        if entry.get("miss"):
            saved = float(entry.get("saved_at") or 0.0)
            if (time.time() - saved) < LYRICS_MISS_TTL_SEC:
                return LyricsState(status="none", last_track_sig=sig)
            return None  # stale negative entry -> refetch

        if entry.get("instrumental"):
            return LyricsState(status="instrumental", instrumental=True,
                               source=str(entry.get("source", "cache")), last_track_sig=sig)

        raw_lines = entry.get("lines")
        lines: List[Tuple[float, str]] = []
        if isinstance(raw_lines, list):
            for item in raw_lines:
                if (
                    isinstance(item, list)
                    and len(item) == 2
                    and isinstance(item[0], (int, float))
                    and isinstance(item[1], str)
                ):
                    lines.append((float(item[0]), item[1]))
        plain = str(entry.get("plain") or "")

        if lines:
            return LyricsState(
                status="cache", has_synced=True, source=str(entry.get("source", "cache")),
                lines=lines, last_track_sig=sig,
            )
        if plain.strip():
            return LyricsState(
                status="plain", source=str(entry.get("source", "cache")),
                plain=plain, last_track_sig=sig,
            )
        return None

    def _save_result(self, sig: str, res: LyricsResult) -> None:
        cache = self._read_cache()
        entry: Dict[str, Any] = {"source": res.source, "saved_at": time.time()}
        if res.instrumental:
            entry["instrumental"] = True
        if res.lines:
            entry["lines"] = [[float(t), str(s)] for (t, s) in res.lines]
        if res.plain.strip():
            entry["plain"] = res.plain
        cache[sig] = entry
        self._write_cache(cache)

    def _save_miss(self, sig: str) -> None:
        cache = self._read_cache()
        cache[sig] = {"miss": True, "saved_at": time.time()}
        self._write_cache(cache)

    # ------------------------------------------------------------ fetching

    def _enqueue(self, track: Track) -> None:
        try:
            while self._q.full():
                self._q.get_nowait()
        except Exception:
            pass
        try:
            self._q.put_nowait(track)
        except queue.Full:
            pass

    def _sig_of(self, track: Track) -> str:
        return make_sig(track.title, track.artist, track.album, track.duration_sec)

    def _still_current(self, sig: str) -> bool:
        with self._lock:
            return self._state.last_track_sig == sig

    def _set_status(self, sig: str, **kw: Any) -> None:
        with self._lock:
            if self._state.last_track_sig != sig:
                return
            for k, v in kw.items():
                setattr(self._state, k, v)

    def _fetch_once(self, track: Track, sig: str) -> Optional[LyricsResult]:
        """One full round across all providers. Returns best result or None."""
        best_plain: Optional[LyricsResult] = None

        # Stage 1: LRCLIB
        self._set_status(sig, status="searching", status_detail="lrclib", fetching=True)
        for fn in (fetch_lrclib_get, fetch_lrclib_search):
            if self._stop or not self._still_current(sig):
                return None
            try:
                res = fn(track)
            except Exception:
                res = None
            if res is None:
                continue
            if res.instrumental or res.has_synced:
                return res
            if res.plain.strip() and best_plain is None:
                best_plain = res

        # Stage 2: QQ + NetEase + Kugou raced
        self._set_status(sig, status="searching", status_detail="qq+netease+kugou", fetching=True)
        futures = {
            self._pool.submit(fetch_qq, track): "qq",
            self._pool.submit(fetch_netease, track): "netease",
            self._pool.submit(fetch_kugou, track): "kugou",
        }
        pref = {"qq": 3, "netease": 2, "kugou": 1}
        candidates: List[LyricsResult] = []
        try:
            for fut in concurrent.futures.as_completed(futures, timeout=14.0):
                try:
                    res = fut.result()
                except Exception:
                    res = None
                if res is not None and res.has_synced:
                    candidates.append(res)
        except concurrent.futures.TimeoutError:
            pass

        if candidates:
            candidates.sort(
                key=lambda r: (round(r.match_score, 2), pref.get(r.source, 0)), reverse=True
            )
            return candidates[0]
        return best_plain

    def _run(self) -> None:
        pending_retry: Optional[Track] = None
        retry_at = 0.0

        while not self._stop:
            track: Optional[Track] = None
            try:
                track = self._q.get(timeout=0.25)
            except queue.Empty:
                if pending_retry is not None and time.monotonic() >= retry_at:
                    track = pending_retry
                    pending_retry = None
                else:
                    continue

            if self._stop or track is None:
                break

            sig = self._sig_of(track)
            if not self._still_current(sig):
                continue

            # Wait out a scheduled backoff for this same track.
            with self._lock:
                attempt = self._state.attempt if self._state.last_track_sig == sig else 0

            result = self._fetch_once(track, sig)

            if self._stop or not self._still_current(sig):
                continue

            if result is not None and (result.has_synced or result.instrumental or result.plain.strip()):
                if result.instrumental:
                    new_state = LyricsState(
                        status="instrumental", instrumental=True, source=result.source,
                        last_track_sig=sig,
                    )
                elif result.has_synced:
                    new_state = LyricsState(
                        status="ok", has_synced=True, source=result.source,
                        lines=result.lines, plain=result.plain, last_track_sig=sig,
                    )
                else:
                    new_state = LyricsState(
                        status="plain", source=result.source, plain=result.plain,
                        last_track_sig=sig,
                    )
                with self._lock:
                    if self._state.last_track_sig == sig:
                        self._state = new_state
                self._save_result(sig, result)
                continue

            # Nothing found: schedule a retry or give up.
            if attempt < len(LYRICS_RETRY_SCHEDULE_SEC):
                delay = float(LYRICS_RETRY_SCHEDULE_SEC[attempt])
                self._set_status(
                    sig,
                    status="retry",
                    status_detail="",
                    attempt=attempt + 1,
                    next_retry_t=time.time() + delay,
                    fetching=False,
                )
                pending_retry = track
                retry_at = time.monotonic() + delay
            else:
                self._set_status(sig, status="none", fetching=False, next_retry_t=0.0)
                self._save_miss(sig)
