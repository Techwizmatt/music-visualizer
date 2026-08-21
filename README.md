# Music Visualizer

A fullscreen "now playing" desktop app for macOS: live synced lyrics, album
artwork, and an audio-reactive visualizer, styled after Apple Music's
fullscreen player.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PySide6](https://img.shields.io/badge/UI-PySide6%20(Qt)-green)
![macOS](https://img.shields.io/badge/platform-macOS-lightgrey)

## Features

- **Now playing info** — title, artist, album, artwork, progress, and
  transport controls (play/pause, next/prev, shuffle, repeat), read live
  from macOS's system media state via the [`media-control`](https://github.com/ungive/media-control)
  CLI. Works with Apple Music/iTunes and most other media apps.
- **Synced lyrics** — fetched automatically and free, with no API keys,
  from multiple public sources with fallback and retry. Results are cached
  locally so repeat plays are instant.
- **High-resolution artwork** — falls back to the iTunes Search API when
  the system doesn't provide it, or provides only a small thumbnail.
- **Audio-reactive visualizer** — captures system audio via a loopback
  device (e.g. [BlackHole](https://github.com/ExistentialAudio/BlackHole)),
  runs live FFT analysis, and drives a soft color wash pulled from the
  album artwork. An optional particle-sphere mode replaces the artwork with
  a rotating globe of dots in the artwork's colors that reacts to the beat.
- **Beat tracking** — a real-time onset/tempo/phase tracker estimates BPM
  and predicts the beat slightly ahead of time, so visual pulses stay in
  sync with the audio instead of lagging behind it. A small on-screen debug
  panel can show BPM, confidence, and the live onset waveform.
- **Smooth, animated UI** — the layout glides between a centered "no
  lyrics" view and a split artwork/lyrics view, with eased scrolling and
  crossfaded line highlights.

## Requirements

- macOS (uses macOS-specific APIs for system media info and audio routing)
- Python 3.11+
- [`media-control`](https://github.com/ungive/media-control) — reads
  now-playing info from the system
- A loopback audio device such as
  [BlackHole](https://github.com/ExistentialAudio/BlackHole) — only needed
  for the audio-reactive visualizer; everything else works without it

## Setup

Install the system dependencies:

```bash
brew install media-control
brew install blackhole-2ch   # optional, for the audio visualizer
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

## Keyboard & mouse

| Input | Action |
| --- | --- |
| `Space` | play / pause |
| `←` / `→` | previous / next track |
| `F` / double-click | toggle fullscreen |
| `Esc` | exit fullscreen |
| `S` | toggle the particle-sphere visualizer |
| `L` | hide / show lyrics |
| `I` | show / hide the info column |
| `V` | toggle the background visualizer |
| `D` | show the beat-tracking debug panel |
| `Q` | quit |
| click / drag progress bar | seek |
| on-screen buttons | shuffle, prev, play/pause, next, repeat |

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
| `audio.py` | audio capture, FFT analysis, beat-engine integration |
| `beat.py` | onset detection, tempo/phase tracking, per-track analysis cache |
| `visualizer.py` | color palette extraction, background wash, particle sphere |
| `ui.py` | rendering, animation, and input handling |
| `settings.py` | user-configurable settings |

## Data & caching

The app writes small local cache files (lyrics, per-track tempo/analysis
data) to speed up repeat plays. These are generated automatically, contain
no audio, and are excluded from version control via `.gitignore`.

## License

No license has been chosen yet — all rights reserved by default. Add a
`LICENSE` file if you want to permit reuse.
