"""Generate the deterministic CC0 Gate 2 offline-video fixture."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


WIDTH = 640
HEIGHT = 480
FPS = 5.0
FRAME_COUNT = 3


def generate_fixture(output: Path) -> str:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        FPS,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("MJPG fixture writer could not be opened")
    try:
        colors = ((30, 40, 220), (40, 200, 50), (220, 80, 30))
        for index, color in enumerate(colors):
            frame = np.full((HEIGHT, WIDTH, 3), color, dtype=np.uint8)
            cv2.rectangle(
                frame,
                (60 + index * 100, 100),
                (260 + index * 100, 380),
                (255, 255, 255),
                thickness=-1,
            )
            cv2.putText(
                frame,
                f"GATE 2 FRAME {index + 1}",
                (120, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(output))
    decoded = 0
    try:
        while True:
            succeeded, frame = capture.read()
            if not succeeded:
                break
            if frame is None or frame.shape != (HEIGHT, WIDTH, 3):
                raise RuntimeError("Generated fixture contains an invalid frame")
            decoded += 1
    finally:
        capture.release()
    if decoded != FRAME_COUNT:
        raise RuntimeError(f"Generated fixture decoded {decoded} frames; expected {FRAME_COUNT}")

    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == "__main__":
    destination = Path("tests/fixtures/media/gate2-three-frame.avi")
    print(generate_fixture(destination))
