"""Adaptive Object Tracking rebuild package."""

__version__ = "0.1.0"


def main(argv=None) -> int:
    """Load the native entry point lazily so package imports stay Qt-light."""
    from aot.app import main as app_main

    return app_main(argv)
