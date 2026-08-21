from __future__ import annotations

# ============================================================
# MusicVisualizer settings
# ============================================================

# ---------- Window ----------
START_FULLSCREEN: bool = False

# ---------- Now Playing (requires `brew install media-control`) ----------
NOW_PLAYING_ENABLED: bool = True
NOW_PLAYING_POLL_MS: int = 250          # 100-250ms is a practical range
KEEP_LAST_GOOD_SECONDS: float = 2.0     # keep last good payload briefly if tool returns empty

# ---------- Lyrics ----------
LYRICS_ENABLED: bool = True

# Providers are tried in stages:
#   stage 1: LRCLIB exact match, then LRCLIB fuzzy search
#   stage 2 (only if stage 1 had no synced lyrics): NetEase + Kugou in parallel
LYRICS_PROVIDER_TIMEOUT_SEC: float = 6.0    # per provider HTTP timeout
LYRICS_USER_AGENT: str = "MusicVisualizer/1.0 (https://github.com/techwizmatt/music-visualizer)"

LRCLIB_BASE_URL: str = "https://lrclib.net"

# Retry backoff after a full round of providers fails (seconds between attempts).
# After the last entry the fetcher gives up until the track is played again.
LYRICS_RETRY_SCHEDULE_SEC: tuple = (3.0, 6.0, 12.0, 24.0, 48.0)

# Duration matching: a candidate whose duration differs by more than this is rejected.
LYRICS_DURATION_TOLERANCE_SEC: float = 3.5

# Negative cache: don't re-hit the network for a known lyricless track for this long.
LYRICS_MISS_TTL_SEC: float = 6 * 3600.0

LYRICS_CACHE_PATH: str = "lyrics_cache.json"
LYRICS_CACHE_MAX_ENTRIES: int = 500

# ---------- Audio capture / analysis (BlackHole loopback) ----------
AUDIO_ENABLED: bool = True
AUDIO_DEVICE_SUBSTRING: str = "blackhole"   # case-insensitive match on input device name
AUDIO_PREFERRED_SAMPLERATE: int = 48000     # falls back to the device default if refused
AUDIO_BLOCK_SIZE: int = 1024
AUDIO_FFT_SIZE: int = 4096                  # ~85ms @ 48k; good bass resolution
AUDIO_RECONNECT_SEC: float = 5.0            # retry cadence if the device vanishes
AUDIO_SILENCE_HINT_SEC: float = 6.0         # playing but silent for this long -> routing hint

# ---------- Visualizer ----------
VISUALIZER_ENABLED: bool = True
VIS_RENDER_MAX_W: int = 220     # offscreen render width; upscaled = free blur, keep small
VIS_BLOB_COUNT: int = 6
VIS_IDLE_MOTION: float = 0.35   # 0..1 how lively the background is with no audio signal
VIS_AUDIO_GAIN: float = 1.0     # overall audio responsiveness multiplier
VIS_BG_BRIGHTNESS: float = 1.0  # artwork-color wash intensity (0.5 subtle .. 1.5 vivid)

# Particle sphere mode (toggle with S): a rotating globe of dots in artwork
# colors that pulses and flashes with the beat, shown in place of the artwork.
VIS_SPHERE_DOTS: int = 4000
VIS_DEFAULT_SPHERE: bool = False

# ---------- Beat prediction ----------
BEAT_ENABLED: bool = True
BEAT_MIN_BPM: float = 60.0
BEAT_MAX_BPM: float = 190.0
# Visuals evaluate the predicted beat clock this far in the FUTURE, canceling
# capture+analysis+render latency so pulses land with the speakers, not after.
BEAT_PREDICT_LOOKAHEAD_SEC: float = 0.07

# Per-track analysis "signature" (never audio): BPM, beat grid aligned to the
# track timeline, and a 1s-resolution energy profile. Replays lock instantly.
ANALYSIS_CACHE_PATH: str = "analysis_cache.json"
ANALYSIS_CACHE_MAX_ENTRIES: int = 500

# ---------- Debug ----------
DEBUG_PANEL_DEFAULT: bool = False   # toggled with D

# ---------- UI ----------
DEFAULT_SHOW_INFO: bool = True   # toggled with I
UI_FPS: int = 60

# Fonts scale with window height; these are multipliers of window height.
FONT_SCALE_LYRICS: float = 0.040        # active lyric line
FONT_SCALE_LYRICS_MIN_PX: int = 22
FONT_SCALE_TITLE: float = 0.0165
FONT_SCALE_SUB: float = 0.0150
FONT_SCALE_HUD: float = 0.0115

# Lyrics panel geometry (fractions of window size, split layout)
LYRICS_PANEL_RIGHT_MARGIN_FRAC: float = 0.055
LYRICS_FOCUS_FRAC: float = 0.38         # active line sits at this fraction of panel height
