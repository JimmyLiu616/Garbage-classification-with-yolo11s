import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO11 model for trash detection.")
    parser.add_argument("--data", type=Path, default=Path("data.yaml"), help="Dataset config path.")
    parser.add_argument("--model", default="yolo11s.pt", help="Base model, for example yolo11n.pt or yolo11s.pt.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--project", default="runs", help="Output directory.")
    parser.add_argument("--name", default="TRASH_yolov11s", help="Experiment name.")
    parser.add_argument("--device", default=None, help="Device to use, for example 0, cpu, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
