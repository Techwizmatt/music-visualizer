from __future__ import annotations

import sys
import threading

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from settings import (
    AUDIO_ENABLED,
    LYRICS_ENABLED,
    NOW_PLAYING_ENABLED,
    NOW_PLAYING_POLL_MS,
    START_FULLSCREEN,
    VISUALIZER_ENABLED,
)
from artwork_fetcher import ArtworkFetcher
from audio import AudioAnalyzer
from lyrics import LyricsManager
from now_playing import (
    MediaControlStream,
    NowPlayingState,
    get_now_playing_once,
    _require_cmd,
)
from ui import LyricsInfoWidget, MainWindow
from visualizer import BackgroundVisualizer


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Music Visualizer")

    np_state: NowPlayingState | None = None
    np_stream: MediaControlStream | None = None
    if NOW_PLAYING_ENABLED:
        np_state = NowPlayingState()
        try:
            _require_cmd("media-control")
        except Exception as e:
            np_state.set_error(str(e))
            QMessageBox.critical(None, "Now Playing Error", str(e))

        poll_busy = threading.Event()

        def poll_now_playing_async() -> None:
            if np_state is None or poll_busy.is_set():
                return

            def worker() -> None:
                poll_busy.set()
                try:
                    np_state.update_from_poll(get_now_playing_once())
                except Exception as exc:
                    np_state.set_error(str(exc))
                finally:
                    poll_busy.clear()

            threading.Thread(target=worker, daemon=True).start()

        np_timer = QTimer()
        np_timer.timeout.connect(poll_now_playing_async)
        np_timer.start(max(100, int(NOW_PLAYING_POLL_MS)))
        poll_now_playing_async()

        # Push stream: instant track changes and (crucially) album artwork,
        # which `media-control get` often omits.
        np_stream = MediaControlStream(np_state.update_from_poll)
        np_stream.start()

        # High-res artwork fallback from the iTunes Search API for tracks
        # where MediaRemote never delivers artwork.
        art_fetcher = ArtworkFetcher(np_state)
        art_fetcher.start()

    lyrics_mgr: LyricsManager | None = None
    if LYRICS_ENABLED:
        lyrics_mgr = LyricsManager()
        lyrics_mgr.start()

    audio: AudioAnalyzer | None = None
    if AUDIO_ENABLED:
        audio = AudioAnalyzer()
        audio.start()

    vis = BackgroundVisualizer() if VISUALIZER_ENABLED else None

    widget = LyricsInfoWidget(np_state, lyrics_mgr, audio, vis)
    window = MainWindow(widget)

    if START_FULLSCREEN:
        window.showFullScreen()
    else:
        window.show()

    def on_about_to_quit() -> None:
        if lyrics_mgr is not None:
            lyrics_mgr.stop()
        if audio is not None:
            audio.stop()
        if np_stream is not None:
            np_stream.stop()

    app.aboutToQuit.connect(on_about_to_quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
