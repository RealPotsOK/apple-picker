"""Show live YOLO detections from a webcam in an OpenCV window."""

from __future__ import annotations

import argparse
import platform
import time

import cv2
import torch
from ultralytics import YOLO


WINDOW_NAME = "Apple Picker - YOLO Webcam"
APPLE_CLASS_ID = 47
APPLE_CLASS_NAME = "apple"


def best_device() -> str:
    """Use Apple Silicon's GPU when PyTorch can access it."""
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_coreml_model(model_path: str) -> bool:
    return model_path.lower().endswith((".mlmodel", ".mlpackage"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a webcam and draw YOLO detection boxes in real time."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0, normally the MacBook camera)",
    )
    parser.add_argument(
        "--model",
        default="yolo11s.mlpackage",
        help="YOLO weights path/name (default: Core ML yolo11s.mlpackage)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence from 0 to 1 (default: 0.35)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for PyTorch models; Core ML chooses Apple hardware automatically",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # AVFoundation starts Mac cameras more reliably than OpenCV's generic backend.
    if platform.system() == "Darwin":
        camera = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(index)
    else:
        camera = cv2.VideoCapture(index)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def run(args: argparse.Namespace) -> None:
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")

    coreml = is_coreml_model(args.model)
    device = None if coreml else (args.device or best_device())
    backend_name = "Core ML" if coreml else device
    if coreml and args.device:
        print(f"Ignoring --device={args.device}: Core ML selects Apple hardware.")
    print(f"Loading {args.model} with {backend_name}...")
    model = YOLO(args.model, task="detect")
    camera = open_camera(args.camera, args.width, args.height)
    if not camera.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. On macOS, allow your terminal "
            "camera access in System Settings > Privacy & Security > Camera."
        )

    print("Camera running with persistent ByteTrack IDs.")
    print("Press Q or Esc in the video window to quit.")
    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")

            track_options = {
                "persist": True,
                "tracker": "bytetrack.yaml",
                "conf": args.confidence,
                "classes": [APPLE_CLASS_ID],
                "verbose": False,
            }
            if device is not None:
                track_options["device"] = device
            result = model.track(frame, **track_options)[0]
            # Core ML exports may preserve class indices but use generic labels.
            result.names[APPLE_CLASS_ID] = APPLE_CLASS_NAME
            annotated = result.plot()

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}  |  {backend_name.upper()} + APPLE TRACKING  |  Q/Esc: quit",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (50, 255, 50),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
