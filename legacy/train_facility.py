import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_loaders(data_dir, batch_size, num_workers):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
            transforms.RandomRotation(8),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_model(class_count):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

    return total_loss / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def train_facility_classifier(
    data_dir,
    output,
    epochs,
    batch_size,
    lr,
    weight_decay,
    num_workers,
    device,
):
    train_dataset, _, train_loader, val_loader = build_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_model(class_count=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": train_dataset.classes,
                    "arch": "resnet18",
                },
                output,
            )

    print(f"best_acc={best_acc:.4f}")
    print(f"saved best checkpoint to {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a facility status classifier from cropped images."
    )
    parser.add_argument(
        "--data",
        default="datasets/facility",
        help="Dataset root with train/ and val/ class folders.",
    )
    parser.add_argument(
        "--output",
        default="models/facility_classifier.pt",
        help="Path to save the best checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda, cuda:0, or cpu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_facility_classifier(
        data_dir=args.data,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
