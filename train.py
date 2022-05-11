from __future__ import division, print_function

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from sklearn import metrics
from torchvision import datasets, transforms
from tqdm import tqdm

import setter

torch.manual_seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
cudnn.benchmark = True
plt.ion()  # interactive mode


@dataclass
class Model_features:
    dataset_sizes: int
    classes: list
    model: str
    lr: float
    optimizer: str
    scheduler: str
    num_epochs: int
    batch_size: int


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="select model",
        type=str,
        default="resnet18",
    )
    parser.add_argument(
        "-l", "--lr", help="define learning rate", type=float, default=0.001
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        help="select optimizer",
        type=str,
        default="SGD",
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        help="select scheduler",
        type=str,
        default="StepLR",
    )
    parser.add_argument(
        "-dir",
        "--directory",
        help="select data directory",
        type=Path,
        default="/data/celery/Dataset/eggplant_leaf_face/",
    )
    parser.add_argument("-b", "--batch", help="define batch size", type=int, default=32)
    parser.add_argument(
        "-w",
        "--worker",
        help="define worker size",
        type=int,
        default=8,
    )
    parser.add_argument("-e", "--epoch", help="define num epochs", type=int, default=20)
    args = parser.parse_args()
    return args


def train_model(
    dataloaders,
    model_features,
):
    since = time.time()

    dataset_sizes = model_features.dataset_sizes
    model_name = model_features.model
    classes = model_features.classes
    num_class = len(classes)
    lr = model_features.lr
    optimizer_name = model_features.optimizer
    scheduler_name = model_features.scheduler
    num_epochs = model_features.num_epochs

    model = setter.set_model(model_name, model_features)
    device = torch.device("cuda:0")
    model.to(device)
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = setter.set_optimizer(model, model_features)
    scheduler = setter.set_scheduler(optimizer, model_features)

    best_acc = 0.0
    best_loss = 100

    print("\n-------------------")
    print(f"model: {model_name}")
    print(f"optimizer: {optimizer_name}")
    print(f"lr: {lr}")
    print(f"scheduler: {scheduler_name}")
    print(f"num_class: {num_class}")
    print("-------------------\n")

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                del loss
                torch.cuda.empty_cache()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                train_loss.append(float(epoch_loss))
                train_acc.append(float(epoch_acc))
            elif phase == "test":
                test_loss.append(float(epoch_loss))
                test_acc.append(float(epoch_acc))

            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_path = Path(
                    f"./models/{model_name}_{optimizer_name}_{lr}_{scheduler_name}.pth"
                )

                torch.save(model.module.state_dict(), model_path)
            if phase == "test" and epoch_loss < best_loss:
                best_loss = epoch_loss

        print()

    time_elapsed = time.time() - since

    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best test Acc: {best_acc:4f}")
    print(f"Best test Loss: {best_loss:4f}")

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(train_acc, label="train_acc")
    ax1.plot(test_acc, label="test_acc")
    ax2.plot(train_loss, label="train_loss")
    ax2.plot(test_loss, label="test_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax1.legend()
    ax2.legend()
    plt.savefig(
        f"./results/acc_{model_name}_{optimizer_name}_{lr}_{scheduler_name}.png"
    )
    return model_path


def model_evaluation(dataloaders, model_features, model_path):
    model_name = model_features.model
    lr = model_features.lr
    optimizer_name = model_features.optimizer
    scheduler_name = model_features.scheduler
    classes = model_features.classes

    model = setter.set_model(model_name, model_features)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda:0")
    model.to(device)
    model = nn.DataParallel(model)

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for inputs, labels in tqdm(dataloaders["test"]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for (label, pred) in zip(labels, preds):
                y_true.append(int(label))
                y_pred.append(int(pred))

    report = metrics.classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=3,
        output_dict=True,
    )
    plt.figure()
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, fmt=".3f")
    plt.tight_layout()
    plt.savefig(
        f"./score/score_{model_name}_{optimizer_name}_{lr}_{scheduler_name}.png"
    )

    plt.figure(figsize=(15, 10))
    plt.rcParams["font.size"] = 20
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(data=cm, index=classes, columns=classes)
    sns.heatmap(cm, annot=True, cbar=False, square=True, cmap="Blues", fmt="d")
    plt.tight_layout()
    plt.savefig(
        f"./heatmaps/heatmap_{model_name}_{optimizer_name}_{lr}_{scheduler_name}.png"
    )


def data_transformer():
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(600),
                transforms.RandomResizedCrop(512),
                transforms.RandAugment(4, 5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    return data_transforms


def main():
    args = parse_args()
    data_dir = args.directory
    data_transforms = data_transformer()
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "test"]
    }

    batch_size = args.batch
    num_workers = args.worker
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size,
            num_workers,
        )
        for x in ["train", "test"]
    }

    model_features = Model_features(
        dataset_sizes={x: len(image_datasets[x]) for x in ["train", "test"]},
        classes=image_datasets["train"].classes,
        model=args.model,
        lr=args.lr,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        num_epochs=args.epoch,
        batch_size=batch_size,
    )

    model_path = train_model(dataloaders, model_features)
    model_evaluation(dataloaders, model_features, model_path)


if __name__ == "__main__":
    main()
