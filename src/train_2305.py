import os

import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset_upd_2005 import get_datasets
from model import SegmentationModel
from config import Config
from tqdm import tqdm
from checkpointer_2205 import CheckpointSaver
from metric import MeanIoU
from loss import CrossEntropyDiceLoss
from accelerate import Accelerator
from utils import seed_everything
import matplotlib.pyplot as plt
import numpy as np


def check_asymmetry(dataset, n_samples=3):
    for i in range(n_samples):
        img, mask = dataset[i]
        img = img.permute(1, 2, 0).numpy() if isinstance(img, torch.Tensor) else img

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].set_title("Изображение")
        ax[1].imshow(mask.squeeze(), cmap='gray')
        ax[1].set_title("Маска")

        # Проверка асимметрии (пример)
        contours, _ = cv2.findContours(mask.numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            (x, y), (w, h), angle = cv2.minAreaRect(contours[0])
            ax[1].text(10, 30, f"Соотношение сторон: {max(w, h) / min(w, h):.2f}",
                       color='red', fontsize=12)

        plt.show()


def visualize_sample(dataset, title="Sample", preprocess_flag=False, save_to_disk=False):
    original_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    idx = dataset.indices[0] if hasattr(dataset, 'indices') else 0

    # Получаем сырые данные (без трансформаций)
    image, mask = original_dataset.__getitem__(idx, apply_transform=False)

    fig, axes = plt.subplots(1, 3 if not original_dataset.transform else 4, figsize=(18, 5))

    # 1. Исходное/предобработанное изображение
    axes[0].imshow(image)
    axes[0].set_title(f"1. {'Предобработанное' if preprocess_flag else 'Сырое'} изображение")

    # 2. Маска
    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title("2. Маска")

    # 3. Если есть трансформы - аугментированное изображение
    if original_dataset.transform:
        augmented = original_dataset.transform(image=image, mask=mask)
        aug_img = augmented["image"].permute(1, 2, 0).numpy()
        axes[2].imshow(aug_img)
        axes[2].set_title("3. После аугментаций")

        # 4. Аугментированная маска
        axes[3].imshow(augmented["mask"].squeeze(), cmap='gray')
        axes[3].set_title("4. Маска после аугментаций")

    for ax in axes:
        ax.axis('off')

    if save_to_disk:
        os.makedirs("visualizations", exist_ok=True)
        fig.savefig(f"visualizations/{title.replace(' ', '_')}.png", bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    return fig


def train():
    seed_everything(42)
    accelerator = Accelerator(
        cpu=False,
        mixed_precision="fp16" if Config.DEVICE == "cuda" else "no"
    )
    writer = SummaryWriter(Config.LOGS_DIR)

    # Модель и оптимизатор
    model = SegmentationModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Данные
    train_dataset, val_dataset, test_dataset = get_datasets(
        images_dir=Config.IMAGES_DIR,
        masks_dir=Config.MASKS_DIR,
        val_ratio=0.15,
        test_ratio=0.05,
        preprocess=Config.PREPROCESS_FLAG,
        crop_borders=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    # Loss и метрики
    device = accelerator.device
    class_weights = torch.tensor(Config.CLASS_WEIGHTS, device=device)
    loss_fn = CrossEntropyDiceLoss(weight=class_weights, ignore_index=-1).to(device)
    metric_fn = MeanIoU(classes_num=Config.NUM_CLASSES, ignore_index=-1).to(device)

    # Подготовка для Accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Чекпоинтер с поддержкой loss (NEW)
    checkpointer = CheckpointSaver(
        accelerator=accelerator,
        model=model,
        metric_name="MeanIoU",
        save_dir=Config.CHECKPOINTS_DIR,
        should_minimize=False
    )

    # Визуализация до обучения
    visualize_sample(train_dataset, "Train Sample", preprocess_flag=Config.PREPROCESS_FLAG, save_to_disk=True)
    visualize_sample(val_dataset, "Val Sample", preprocess_flag=Config.PREPROCESS_FLAG)

    # Цикл обучения
    best_iou = 0.0  # NEW: отслеживаем лучший IoU
    best_loss = float('inf')  # NEW: отслеживаем лучший loss

    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_train_loss = 0.0

        # Train phase
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            try:
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, masks.long())
                accelerator.backward(loss)
                optimizer.step()
                epoch_train_loss += loss.item()
            except Exception as batch_error:
                print(f"\nBatch error: {batch_error}")
                continue

        # Val phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                val_loss += loss_fn(outputs, masks.long()).item()
                metric_fn.update(outputs, masks.long())

            val_iou = metric_fn.compute().item()
            val_loss /= len(val_loader)  # NEW: средний loss на валидации
            epoch_train_loss /= len(train_loader)  # NEW: средний train loss

            # Обновляем лучшие метрики (NEW)
            if val_iou > best_iou:
                best_iou = val_iou
            if val_loss < best_loss:
                best_loss = val_loss

            # Логирование в TensorBoard (NEW: добавил loss)
            writer.add_scalars("Loss", {
                "train": epoch_train_loss,
                "val": val_loss
            }, epoch)

            writer.add_scalar("IoU/val", val_iou, epoch)
            writer.add_scalar("Best_IoU", best_iou, epoch)  # NEW
            writer.add_scalar("Best_Loss", best_loss, epoch)  # NEW
            writer.add_scalar("Loss_train", epoch_train_loss, epoch)
            writer.add_scalar("Loss_val", val_loss, epoch)

            '''
            # Визуализация каждые 5 эпох
            if epoch % 5 == 0:
                fig = visualize_sample(
                    val_dataset,
                    f"Val Sample Epoch {epoch}",
                    preprocess_flag=Config.PREPROCESS_FLAG
                )
                writer.add_figure("Validation Samples", fig, epoch)
                plt.close(fig)
            '''

            metric_fn.reset()
            if epoch == 0:
                writer.add_graph(model, images)

        # Сохранение чекпоинта (NEW: передаем val_loss)
        checkpointer.save(metric_val=val_iou, loss_val=val_loss, epoch=epoch)

        # Вывод в консоль (NEW: добавил loss)
        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Best IoU: {best_iou:.4f} | "
            f"Best Loss: {best_loss:.4f}"
        )

    writer.close()
    print("Training completed!")
    return accelerator.unwrap_model(model)

def evaluate(model, test_loader):
    model.eval()
    device = next(model.parameters()).device
    metric_fn = MeanIoU(classes_num=Config.NUM_CLASSES, ignore_index=-1).to(device)
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            metric_fn.update(outputs, masks.long())
    return metric_fn.compute().item()


if __name__ == "__main__":
    print("actual version--train")
    train()