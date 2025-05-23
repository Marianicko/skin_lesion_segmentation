import os
import glob
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

# Проверяем окружение (Colab или нет)
try:
    from google.colab import drive

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def check_asymmetry(dataset, n_samples=3):
    for i in range(n_samples):
        img, mask = dataset[i]
        img = img.permute(1, 2, 0).numpy() if isinstance(img, torch.Tensor) else img

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].set_title("Изображение")
        ax[1].imshow(mask.squeeze(), cmap='gray')
        ax[1].set_title("Маска")

        contours, _ = cv2.findContours(mask.numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            (x, y), (w, h), angle = cv2.minAreaRect(contours[0])
            ax[1].text(10, 30, f"Соотношение сторон: {max(w, h) / min(w, h):.2f}",
                       color='red', fontsize=12)
        plt.show()


def visualize_sample(dataset, title="Sample", preprocess_flag=False, save_to_disk=False):
    original_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    idx = dataset.indices[0] if hasattr(dataset, 'indices') else 0
    image, mask = original_dataset.__getitem__(idx, apply_transform=False)

    fig, axes = plt.subplots(1, 3 if not original_dataset.transform else 4, figsize=(18, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"1. {'Предобработанное' if preprocess_flag else 'Сырое'} изображение")
    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title("2. Маска")

    if original_dataset.transform:
        augmented = original_dataset.transform(image=image, mask=mask)
        axes[2].imshow(augmented["image"].permute(1, 2, 0).numpy())
        axes[2].set_title("3. После аугментаций")
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

    # Настройка путей для чекпоинтов
    if IN_COLAB:
        drive.mount('/content/drive')
        CHECKPOINTS_DIR = '/content/drive/MyDrive/colab_checkpoints'
        print("✓ Google Drive подключен")
    else:
        CHECKPOINTS_DIR = './local_checkpoints'
        print("→ Локальное сохранение чекпоинтов")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    accelerator = Accelerator(
        cpu=False,
        mixed_precision="fp16" if Config.DEVICE == "cuda" else "no"
    )
    writer = SummaryWriter(Config.LOGS_DIR)

    # Инициализация модели
    model = SegmentationModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Загрузка данных
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

    # Подготовка для Accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Инициализация чекпоинтера
    checkpointer = CheckpointSaver(
        accelerator=accelerator,
        model=model,
        metric_name="MeanIoU",
        save_dir=CHECKPOINTS_DIR,
        should_minimize=False
    )

    # Попытка загрузить последний чекпоинт
    start_epoch = 0
    checkpoint_files = sorted(glob.glob(f"{CHECKPOINTS_DIR}/model_e*.pt"))
    if checkpoint_files:
        last_checkpoint = checkpoint_files[-1]
        checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Загружен чекпоинт: {last_checkpoint}, эпоха {start_epoch}")

    # Цикл обучения
    best_iou = 0.0
    best_loss = float('inf')

    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        epoch_train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            try:
                optimizer.zero_grad()
                outputs = model(images)
                loss = CrossEntropyDiceLoss()(outputs, masks.long())
                accelerator.backward(loss)
                optimizer.step()
                epoch_train_loss += loss.item()
            except Exception as e:
                print(f"Ошибка в батче: {e}")
                continue

        # Валидация
        model.eval()
        val_loss = 0.0
        metric_fn = MeanIoU(classes_num=Config.NUM_CLASSES, ignore_index=-1).to(accelerator.device)

        with torch.no_grad():
            for images, masks in val_loader:
                outputs = model(images)
                val_loss += CrossEntropyDiceLoss()(outputs, masks.long()).item()
                metric_fn.update(outputs, masks.long())

        val_iou = metric_fn.compute().item()
        val_loss /= len(val_loader)
        epoch_train_loss /= len(train_loader)

        # Обновление лучших метрик
        best_iou = max(best_iou, val_iou)
        best_loss = min(best_loss, val_loss)

        # Логирование
        writer.add_scalars("Loss", {"train": epoch_train_loss, "val": val_loss}, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)
        writer.add_scalar("Best_IoU", best_iou, epoch)
        writer.add_scalar("Best_Loss", best_loss, epoch)

        # Сохранение чекпоинта
        checkpointer.save(metric_val=val_iou, loss_val=val_loss, epoch=epoch)

        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Best IoU: {best_iou:.4f}"
        )

    writer.close()
    print("Обучение завершено!")
    return accelerator.unwrap_model(model)


if __name__ == "__main__":
    trained_model = train()