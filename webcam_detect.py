import argparse
import os
from pathlib import Path
import time

import cv2
from ultralytics import YOLO


def default_model_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "weights" / "best.pt",
        base_dir / "runs" / "TRASH_yolov11s" / "weights" / "best.pt",
        base_dir / "yolo11s.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time trash detection with a webcam.")
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
        help="Path to a YOLO model file. Defaults to weights/best.pt.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index. Use 0 for the default camera.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = YOLO(str(model_path))
    if os.name == "nt":
        camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    else:
        camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    previous_time = time.perf_counter()

    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera")
            break

        start_time = time.perf_counter()
        results = model.predict(frame, conf=args.conf, verbose=False)
        inference_ms = (time.perf_counter() - start_time) * 1000.0

        annotated = results[0].plot()

        current_time = time.perf_counter()
        fps = 1.0 / max(current_time - previous_time, 1e-6)
        previous_time = current_time

        overlay_lines = [
            f"FPS: {fps:.1f}",
            f"Inference: {inference_ms:.1f} ms",
        ]

        y = 30
        for line in overlay_lines:
            cv2.putText(
                annotated,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 32

        cv2.imshow("Trash Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
