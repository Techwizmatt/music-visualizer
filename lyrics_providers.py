from __future__ import annotations

"""
Synced-lyrics providers. All free, no API keys. (Endpoints verified 2026-08.)

Chain (see LyricsManager): LRCLIB exact -> LRCLIB fuzzy search -> QQ Music +
NetEase + Kugou raced in parallel. Every provider returns a LyricsResult; the
manager scores them and keeps the best synced result.

Notes from provider verification:
  * LRCLIB /api/get matches duration server-side (±2s); /api/search returns
    full records but needs client-side duration filtering.
  * QQ needs only a Referer header; NetEase durations are in ms; Kugou lyric
    downloads are base64. All three censor profanity and prepend timed credit
    lines ("作词 : …", "Title - Artist") that we strip.
"""

import base64
import difflib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from settings import (
    LRCLIB_BASE_URL,
    LYRICS_DURATION_TOLERANCE_SEC,
    LYRICS_PROVIDER_TIMEOUT_SEC,
    LYRICS_USER_AGENT,
)

# ---------------------------------------------------------------- data types


@dataclass(frozen=True)
class Track:
    title: str
    artist: str
    album: str
    duration_sec: int


@dataclass
class LyricsResult:
    source: str
    lines: List[Tuple[float, str]] = field(default_factory=list)  # synced
    plain: str = ""
    instrumental: bool = False
    match_score: float = 0.0

    @property
    def has_synced(self) -> bool:
        return len(self.lines) >= 2


# ---------------------------------------------------------------- LRC parsing

_LRC_TIME_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
_LRC_WORD_TAG_RE = re.compile(r"<\d+:\d+(?:\.\d+)?>")


def parse_lrc(text: str) -> List[Tuple[float, str]]:
    """Parse standard or enhanced LRC into sorted (seconds, text) lines."""
    out: List[Tuple[float, str]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        times = _LRC_TIME_RE.findall(line)
        if not times:
            continue
        body = _LRC_TIME_RE.sub("", line)
        body = _LRC_WORD_TAG_RE.sub("", body).strip()
        if not body:
            continue
        for mm, ss in times:
            try:
                out.append((int(mm) * 60 + float(ss), body))
            except Exception:
                continue
    out.sort(key=lambda x: x[0])
    dedup: List[Tuple[float, str]] = []
    last = None
    for t, txt in out:
        key = (round(t, 3), txt)
        if key != last:
            dedup.append((t, txt))
            last = key
    return dedup


_CJK_META_RE = re.compile(
    r"^(作词|作曲|编曲|制作人|producer|lyricist|composer|arranger|混音|吉他|贝斯|鼓|键盘|录音|母带|发行|监制|和声|词|曲"
    r"|written by|composed by|lyrics by|arranged by|produced by|mixed by|mastered by)\s*[:：]",
    re.IGNORECASE,
)
_TITLE_DASH_ARTIST_RE = re.compile(r"^.{1,60}\s+-\s+.{1,60}$")


def strip_metadata_lines(lines: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    """Chinese services prepend timed credit lines ("作词 : …", "Written by：…",
    and a "Title - Artist" line near t=0) — drop them."""
    cleaned = []
    for i, (t, s) in enumerate(lines):
        txt = s.strip()
        if _CJK_META_RE.match(txt):
            continue
        if i < 2 and t < 1.5 and _TITLE_DASH_ARTIST_RE.match(txt):
            continue
        cleaned.append((t, s))
    return cleaned if len(cleaned) >= 2 else lines


# ---------------------------------------------------------------- matching

_PAREN_RE = re.compile(r"\s*[\(\[\{][^)\]\}]*[\)\]\}]")
_FEAT_RE = re.compile(r"\s+(feat\.?|ft\.?|featuring)\s+.*$", re.IGNORECASE)
_NONWORD_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = _FEAT_RE.sub("", s)
    s = _PAREN_RE.sub("", s)
    s = _NONWORD_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _similar(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92
    return difflib.SequenceMatcher(None, a, b).ratio()


def score_candidate(
    track: Track,
    cand_title: str,
    cand_artist: str,
    cand_duration_sec: Optional[float],
) -> float:
    """
    0 => reject. Otherwise higher is better (max ~1.0).
    Duration is the strongest signal when available.
    """
    ts = _similar(track.title, cand_title)
    as_ = _similar(track.artist, cand_artist)
    if ts < 0.55 or as_ < 0.45:
        return 0.0
    score = 0.55 * ts + 0.30 * as_
    if cand_duration_sec is not None and cand_duration_sec > 0:
        dd = abs(float(cand_duration_sec) - float(track.duration_sec))
        if dd > LYRICS_DURATION_TOLERANCE_SEC:
            return 0.0
        score += 0.15 * (1.0 - dd / LYRICS_DURATION_TOLERANCE_SEC)
    return score


# ---------------------------------------------------------------- HTTP

def _http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = LYRICS_PROVIDER_TIMEOUT_SEC,
) -> Tuple[int, bytes]:
    hdrs = {"User-Agent": LYRICS_USER_AGENT, "Accept": "*/*"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, method="GET", headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(getattr(resp, "status", 200)), resp.read()
    except urllib.error.HTTPError as e:
        try:
            return int(e.code or 0), e.read()
        except Exception:
            return int(e.code or 0), b""
    except Exception:
        return 0, b""


def _http_json(url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[int, dict]:
    code, body = _http_get(url, headers)
    try:
        data = json.loads(body.decode("utf-8", errors="replace") or "{}")
        return code, data if isinstance(data, (dict, list)) else {}
    except Exception:
        return code, {}


# ---------------------------------------------------------------- LRCLIB

def _lrclib_record_to_result(rec: dict, track: Track, tag: str) -> Optional[LyricsResult]:
    if not isinstance(rec, dict):
        return None
    if rec.get("instrumental"):
        return LyricsResult(source=f"lrclib:{tag}", instrumental=True, match_score=1.0)
    synced = str(rec.get("syncedLyrics") or "")
    plain = str(rec.get("plainLyrics") or "")
    lines = parse_lrc(synced) if synced.strip() else []
    if not lines and not plain.strip():
        return None
    return LyricsResult(source=f"lrclib:{tag}", lines=lines, plain=plain, match_score=1.0)


def fetch_lrclib_get(track: Track) -> Optional[LyricsResult]:
    qp = urllib.parse.urlencode(
        {
            "track_name": track.title,
            "artist_name": track.artist,
            "album_name": track.album,
            "duration": str(int(track.duration_sec)),
        }
    )
    code, data = _http_json(f"{LRCLIB_BASE_URL}/api/get?{qp}")
    if code != 200 or not isinstance(data, dict):
        return None
    return _lrclib_record_to_result(data, track, "get")


def fetch_lrclib_search(track: Track) -> Optional[LyricsResult]:
    """Fuzzy fallback: /api/search has no server-side duration filter, so we
    score candidates ourselves and take the best."""
    attempts = [
        {"track_name": track.title, "artist_name": track.artist},
        {"q": f"{track.artist} {track.title}"},
    ]
    best: Optional[LyricsResult] = None
    best_score = 0.0
    for params in attempts:
        code, data = _http_json(f"{LRCLIB_BASE_URL}/api/search?{urllib.parse.urlencode(params)}")
        if code != 200 or not isinstance(data, list):
            continue
        for rec in data[:20]:
            if not isinstance(rec, dict):
                continue
            score = score_candidate(
                track,
                str(rec.get("trackName") or ""),
                str(rec.get("artistName") or ""),
                float(rec.get("duration") or 0) or None,
            )
            if score <= 0:
                continue
            res = _lrclib_record_to_result(rec, track, "search")
            if res is None:
                continue
            # Prefer synced over plain regardless of small score differences.
            rank = (1 if res.has_synced else 0, score)
            if best is None or rank > (1 if best.has_synced else 0, best_score):
                res.match_score = score
                best, best_score = res, score
        if best is not None and best.has_synced:
            break
    return best


# ---------------------------------------------------------------- QQ Music

_MOZILLA_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_QQ_SEARCH_HEADERS = {"Referer": "https://y.qq.com", "User-Agent": _MOZILLA_UA}
_QQ_LYRIC_HEADERS = {
    "Referer": "https://y.qq.com/portal/player.html",
    "User-Agent": _MOZILLA_UA,
}


def fetch_qq(track: Track) -> Optional[LyricsResult]:
    q = urllib.parse.urlencode(
        {
            "w": f"{track.artist} {track.title}",
            "format": "json",
            "p": "1",
            "n": "8",
            "new_json": "1",
        }
    )
    code, data = _http_json(
        f"https://c.y.qq.com/soso/fcgi-bin/client_search_cp?{q}", _QQ_SEARCH_HEADERS
    )
    if code != 200 or not isinstance(data, dict):
        return None
    songs = (((data.get("data") or {}).get("song")) or {}).get("list") or []
    best_mid, best_score = None, 0.0
    for s in songs:
        if not isinstance(s, dict):
            continue
        singers = ", ".join(
            str(x.get("name") or "") for x in (s.get("singer") or []) if isinstance(x, dict)
        )
        name = str(s.get("name") or s.get("songname") or "")
        interval = float(s.get("interval") or 0)
        score = score_candidate(track, name, singers, interval if interval > 0 else None)
        if score > best_score:
            best_score = score
            best_mid = str(s.get("mid") or s.get("songmid") or "")
    if not best_mid:
        return None

    q2 = urllib.parse.urlencode({"songmid": best_mid, "format": "json", "nobase64": "1"})
    code2, data2 = _http_json(
        f"https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg?{q2}", _QQ_LYRIC_HEADERS
    )
    if code2 != 200 or not isinstance(data2, dict) or data2.get("retcode") != 0:
        return None
    lrc = str(data2.get("lyric") or "")
    if not lrc.strip():
        return None
    lines = strip_metadata_lines(parse_lrc(lrc))
    if not lines:
        return None
    return LyricsResult(source="qq", lines=lines, match_score=best_score)


# ---------------------------------------------------------------- NetEase

_NETEASE_HEADERS = {
    "Referer": "https://music.163.com/",
    # NMTID/os cookies guard against the intermittent {"code":460,"msg":"Cheating"} response.
    "Cookie": "NMTID=00Ok3oJ9d6c1b8e2f4a6c8e0d2b4f6a8; os=pc; appver=8.10.05",
    "User-Agent": _MOZILLA_UA,
}


def fetch_netease(track: Track) -> Optional[LyricsResult]:
    q = urllib.parse.urlencode(
        {"s": f"{track.artist} {track.title}", "type": "1", "limit": "8", "offset": "0"}
    )
    code, data = _http_json(f"https://music.163.com/api/search/get?{q}", _NETEASE_HEADERS)
    if code != 200 or not isinstance(data, dict):
        return None
    songs = ((data.get("result") or {}).get("songs")) or []
    best_id, best_score = None, 0.0
    for s in songs:
        if not isinstance(s, dict):
            continue
        artists = ", ".join(
            str(a.get("name") or "") for a in (s.get("artists") or []) if isinstance(a, dict)
        )
        dur_ms = float(s.get("duration") or 0)
        score = score_candidate(
            track, str(s.get("name") or ""), artists, (dur_ms / 1000.0) if dur_ms > 0 else None
        )
        if score > best_score:
            best_score, best_id = score, s.get("id")
    if not best_id:
        return None

    q2 = urllib.parse.urlencode({"id": str(best_id), "lv": "-1", "kv": "-1", "tv": "-1"})
    code2, data2 = _http_json(f"https://music.163.com/api/song/lyric?{q2}", _NETEASE_HEADERS)
    if code2 != 200 or not isinstance(data2, dict):
        return None
    lrc = str(((data2.get("lrc") or {}).get("lyric")) or "")
    if not lrc.strip():
        return None
    lines = strip_metadata_lines(parse_lrc(lrc))
    if not lines:
        return None
    return LyricsResult(source="netease", lines=lines, match_score=best_score)


# ---------------------------------------------------------------- Kugou

def fetch_kugou(track: Track) -> Optional[LyricsResult]:
    kw = urllib.parse.quote(f"{track.artist} - {track.title}")
    code, data = _http_json(
        f"http://mobilecdn.kugou.com/api/v3/search/song?format=json&keyword={kw}&page=1&pagesize=8"
    )
    if code != 200 or not isinstance(data, dict):
        return None
    infos = ((data.get("data") or {}).get("info")) or []
    best, best_score = None, 0.0
    for s in infos:
        if not isinstance(s, dict):
            continue
        score = score_candidate(
            track,
            str(s.get("songname") or ""),
            str(s.get("singername") or ""),
            float(s.get("duration") or 0) or None,
        )
        if score > best_score:
            best_score, best = score, s
    if not best:
        return None

    q = urllib.parse.urlencode(
        {
            "ver": "1",
            "man": "yes",
            "client": "mobi",
            "keyword": f"{track.artist} - {track.title}",
            "duration": str(int(float(best.get("duration") or track.duration_sec) * 1000)),
            "hash": str(best.get("hash") or ""),
        }
    )
    code2, data2 = _http_json(f"http://krcs.kugou.com/search?{q}")
    cands = (data2.get("candidates") or []) if isinstance(data2, dict) else []
    if code2 != 200 or not cands:
        return None
    cand = cands[0]
    q3 = urllib.parse.urlencode(
        {
            "ver": "1",
            "client": "pc",
            "id": str(cand.get("id") or ""),
            "accesskey": str(cand.get("accesskey") or ""),
            "fmt": "lrc",
            "charset": "utf8",
        }
    )
    code3, data3 = _http_json(f"http://lyrics.kugou.com/download?{q3}")
    if code3 != 200 or not isinstance(data3, dict):
        return None
    content_b64 = str(data3.get("content") or "")
    if not content_b64:
        return None
    try:
        lrc = base64.b64decode(content_b64).decode("utf-8", errors="replace")
    except Exception:
        return None
    lines = strip_metadata_lines(parse_lrc(lrc))
    if not lines:
        return None
    return LyricsResult(source="kugou", lines=lines, match_score=best_score)
