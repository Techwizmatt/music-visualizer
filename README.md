# Music Visualizer

A fullscreen "now playing" desktop app for macOS: live synced lyrics, album
artwork, and an audio-reactive visualizer, styled after Apple Music's
fullscreen player.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PySide6](https://img.shields.io/badge/UI-PySide6%20(Qt)-green)
![macOS](https://img.shields.io/badge/platform-macOS-lightgrey)

## Features

- **Now playing info** — title, artist, album, artwork, progress, and
  transport controls (play/pause and next/previous), read live
  from macOS's system media state via the [`media-control`](https://github.com/ungive/media-control)
  CLI. Works with Apple Music/iTunes and most other media apps.
- **Synced lyrics** — fetched automatically and free, with no API keys,
  from multiple public sources with fallback and retry. Results are cached
  locally so repeat plays are instant.
- **High-resolution artwork** — falls back to the iTunes Search API when
  the system doesn't provide it, or provides only a small thumbnail.
- **Audio-reactive visualizer** — captures system audio via a loopback
  device (e.g. [BlackHole](https://github.com/ExistentialAudio/BlackHole)),
  runs live FFT analysis, and drives a permanent soft color wash pulled from
  the album artwork. Nine optional artwork-colored particle, mesh, and
  waveform geometries replace the artwork and react to the audio.
- **Beat tracking** — a real-time onset/tempo/phase tracker estimates BPM
  and predicts the beat slightly ahead of time, so visual pulses stay in
  sync with the audio instead of lagging behind it. A small on-screen debug
  panel can show BPM, confidence, and the live onset waveform.
- **Full-track recording and replay analysis** — captures each BlackHole track
  as float32 stereo at 48 kHz, aligned to the exact position and duration
  reported by macOS, then finishes it as a tagged 320 kbps MP3. First playback
  drives the selected animation from the live analyzer. Once the complete
  track file exists, a background process analyzes the whole sample and repeat
  playback uses its dense visual profile at the current media timestamp.
- **Smooth, animated UI** — the layout glides between a centered "no
  lyrics" view and a split artwork/lyrics view, with eased scrolling and
  crossfaded line highlights.

## Requirements

- macOS (uses macOS-specific APIs for system media info and audio routing)
- Python 3.11+
- [`media-control`](https://github.com/ungive/media-control) — reads
  now-playing info from the system
- [`ffmpeg`](https://ffmpeg.org/) — creates and decodes high-quality MP3 recordings
- A loopback audio device such as
  [BlackHole](https://github.com/ExistentialAudio/BlackHole) — only needed
  for the audio-reactive visualizer; everything else works without it

## Setup

Install the system dependencies:

```bash
brew install media-control
brew install blackhole-2ch   # optional, for the audio visualizer
brew install ffmpeg          # required for recorded MP3 files
```

Install the Python dependencies (a virtual environment is recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
python main.py
```

## Audio routing for the visualizer

The visualizer only sees audio that's actually routed through the loopback
device. To hear system audio while still hearing it yourself:

1. Open **Audio MIDI Setup** and create a **Multi-Output Device** that
   includes both your normal output (e.g. speakers/headphones) and
   **BlackHole 2ch**.
2. Set your Mac's sound output to that **Multi-Output Device**
   (via Control Center or System Settings → Sound → Output).

The app looks for an input device whose name contains "BlackHole" — if
none is found, the app still runs, just without the reactive visualizer
(the debug panel and on-screen hints will say why).

## Recorded audio and full-track visuals

Completed recordings are written under `audio_recordings/`. Their readable
filenames contain artist, title, album, reported duration, and a short hash of
that exact now-playing identity. The reported duration fixes the WAV length;
the reported position places every audio block on that timeline. A track
change closes the previous partial recording before the new track starts.
The block timestamp comes from PortAudio's capture clock, so input-buffer
latency is removed when samples are placed against the now-playing position.

Capture uses float32 stereo at 48 kHz—the highest sample rate supported by MP3.
The working WAV lives under the hidden `audio_recordings/.work/` directory and
is removed after completion. The finished file is a 320 kbps CBR MP3, roughly
12 MB for five minutes. Before lossy encoding, the complete float recording is
scaled with one fixed gain to a -4 dBFS codec-safe peak. This preserves its dynamics and
prevents a quiet BlackHole input from wasting MP3 resolution; the original peak
and applied gain are written into the ID3 metadata. Its other tags contain the
now-playing title, artist, album, genre, track number, reported duration, source
identifiers, and artwork. No separate manifest is created. Partial plays can
continue within the running app, but are not treated as complete until at least
99.5% of the timeline has audio.

CoreAudio's real-time callback performs only one owned float copy and a
nonblocking queue operation. A high-priority capture worker handles the live
analysis ring and timeline handoff, a separate writer handles buffered disk I/O
and MP3 creation, and whole-file analysis runs in a lower-priority process.
Foreground animation frames are rendered into an offscreen image on their own
latest-frame-only worker, so slow visual frames are discarded rather than
queued and can never delay recorded audio.

When that exact MP3 filename exists, a background process analyzes the entire
audio clip into memory for the current playback. It extracts beat, 24 spectrum
bands, a signed waveform trace, vocal presence, brightness, transient density,
stereo width, and section boundaries. Surrounding frames are compared at the
current now-playing time to derive relative slow sections, sustained builds
and releases, a three-second anticipation ramp before large jumps, spectral
direction, and softened chorus/drop impact strength. These long-form signals
change animation speed, direction, tension, expansion, and implosion without
resetting animation phase. No derived analysis is saved:
there is no global analysis cache and no per-track visual sidecar. While that
in-memory pass is running—or when a song has never been heard—the selected mode
keeps using live BlackHole analysis. The palette and particle-image morph are
always generated at runtime from the current artwork supplied by now-playing.
Artwork and visual output are not stored separately from the recording.

The foreground modes are: `1` particle sphere, `2` chroma ribbons, `3` waveform
ring, `4` liquid orbit, `5` harmonic knot, `6` double helix, `7` CRT wavefield,
`8` segmented artwork vortex, and `9` artwork relief. Bass, mids, vocal
presence, snare-like high-frequency transients, highs, stereo width, buildup,
and chorus/drop strength drive separate lines or particle groups in every
mode. These are musical cues, not source-separated stems. Modes 2, 4, 5, 7,
and 8 transform their own lines, rings, and detected artwork regions directly
out of the artwork. Mode 8 detects a simple background and disconnected center
subjects once per cover. The intact cover is its only raster layer: calm
passages show that clean image, then foreground and background dissolve into
their matching dots before those dot groups separate. Strong sections resolve
the entire cover as dots, with long attack/release easing when the music moves
back toward the image. Its camera is clamped to 15 degrees so the artwork stays
centered and never collapses into a narrow, edge-on strip.
Mode 7 uses a phase-continuous spectral surface blended with the real signed
trace, plus animation-local phosphor glow and scanlines. Live audio shows its
accumulating history; full-file playback shows three seconds of earlier traces,
the current playhead, and three seconds of upcoming traces. Mode 9 uses a
separate roughly 9,000-point artwork grid for higher-resolution reconstruction.

The recording status in the debug panel identifies the exact handoff stage:

- `waiting-for-position` — now-playing has not supplied a fresh playback time.
- `waiting-for-audio` — the playback time is ready, but no BlackHole audio block
  has reached the recorder yet.
- `paused` — now-playing says playback is paused.
- `recording` — audio is being placed on the reported track timeline; the
  percentage shows timeline coverage.
- `encoding-mp3` — the complete working recording is becoming the final MP3.
- `analyzing-file` — the finished MP3 is being analyzed in memory.
- `file-ready` — full-file, timestamp-synced animation data is active.

The separate audio status reports `no-device` when BlackHole cannot be found
and `silent` when its stream is open but contains no audible signal. The old
ambiguous `checking` state is no longer used.

## Keyboard & mouse

| Input | Action |
| --- | --- |
| `Space` | play / pause |
| `←` / `→` | previous / next track |
| `F` / double-click | toggle fullscreen |
| `Esc` | exit fullscreen |
| `1`–`9` | choose an artwork-colored foreground animation (`7` is the CRT wavefield) |
| `L` | hide / show lyrics |
| `I` | show / hide track information and controls; artwork/animation remains |
| `V` | toggle the selected foreground animation; the background wash stays on |
| drag the animation | orbit or tilt it without interrupting its music-driven motion |
| `D` | show the beat-tracking debug panel |
| `Q` | quit |
| click / drag progress bar | seek |
| on-screen buttons | previous, play/pause, next |

## Configuration

All tunable behavior lives in [`settings.py`](settings.py) — polling
intervals, which features are enabled, cache locations, visualizer knobs,
beat-tracking parameters, and UI sizing.

## Project layout

| File | Role |
| --- | --- |
| `main.py` | app entry point and service wiring |
| `now_playing.py` | system now-playing polling, position tracking, playback commands |
| `lyrics.py` | lyrics manager: caching, multi-source fetch, retry logic |
| `lyrics_providers.py` | individual lyrics source implementations |
| `artwork_fetcher.py` | high-resolution artwork fallback |
| `audio.py` | audio capture, live/full-file selection, beat-engine integration |
| `recording.py` | timeline capture, tagged MP3 encoding, and asynchronous whole-file analysis |
| `beat.py` | live onset detection and tempo/phase tracking |
| `visualizer.py` | color palette extraction, always-on wash, nine foreground geometries |
| `ui.py` | rendering, animation, and input handling |
| `settings.py` | user-configurable settings |

## Data & caching

The app writes a small lyrics cache plus tagged high-quality MP3 recordings. Derived
audio analysis stays in memory and is not saved. Runtime data is excluded from
version control via `.gitignore`.

## License

No license has been chosen yet — all rights reserved by default. Add a
`LICENSE` file if you want to permit reuse.
