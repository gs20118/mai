import argparse
from pathlib import Path

from ultralytics import YOLO


def train_detector(
    data_yaml,
    base_model,
    epochs,
    image_size,
    batch_size,
    device,
    project,
    run_name,
):
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data config not found: {data_yaml}")

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=4,
        project=project,
        name=run_name,
        pretrained=True,
        patience=20,
        cache=False,
    )
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector for craters and UXO objects."
    )
    parser.add_argument(
        "--data",
        default="datasets/detector/data.yaml",
        help="YOLO data.yaml path.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Base YOLO model, for example yolo11n.pt or yolov8n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Training image size. Small UXO objects usually need 1024 or higher.",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device id such as 0, or cpu.",
    )
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="detector_crater_uxo")
    return parser.parse_args()


def main():
    args = parse_args()
    train_detector(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
