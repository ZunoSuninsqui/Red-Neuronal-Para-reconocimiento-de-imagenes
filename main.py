"""Entry point for the digit recognition application."""
from __future__ import annotations

from ui import DigitRecognizerApp


def main() -> None:
    app = DigitRecognizerApp()
    app.run()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
