import os

import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset_upd_2005 import get_datasets
from model import SegmentationModel
from config import Config
from tqdm import tqdm
from checkpointer import CheckpointSaver
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
    # Инициализация (оставить без изменений)
    seed_everything(42)
    accelerator = Accelerator(
        cpu=False,
        mixed_precision="fp16" if Config.DEVICE == "cuda" else "no"
    )
    writer = SummaryWriter(Config.LOGS_DIR)

    # Модель и оптимизатор (оставить без изменений)
    model = SegmentationModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Данные с проверкой размеров
    train_dataset, val_dataset, test_dataset = get_datasets(
        images_dir=Config.IMAGES_DIR,
        masks_dir=Config.MASKS_DIR,
        val_ratio=0.15,
        test_ratio=0.05,
        preprocess=Config.PREPROCESS_FLAG,
        crop_borders=True
    )

    # Проверка размеров
    print("\n=== Проверка данных ===")
    sample_img, sample_mask = train_dataset[0]
    print(f"Размер изображения: {sample_img.shape}")
    print(f"Размер маски: {sample_mask.shape}")
    assert sample_img.shape[-2:] == sample_mask.shape[-2:], "Размеры изображения и маски не совпадают"

    # Визуализация (оставить без изменений)
    print("\nTrain Sample:")
    visualize_sample(train_dataset, "Train Sample", preprocess_flag=Config.PREPROCESS_FLAG, save_to_disk=True)

    # DataLoader с улучшенными параметрами
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

    # Метрики и loss
    device = accelerator.device

    class_weights = torch.tensor([0.3, 0.7], device=device)
    loss_fn = CrossEntropyDiceLoss(weight=class_weights, ignore_index=-1).to(device)
    metric_fn = MeanIoU(classes_num=Config.NUM_CLASSES, ignore_index=-1).to(device)

    # Подготовка для Accelerator (оставить без изменений)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    #metric_fn = metric_fn.to(device)
    # Чекпоинтер (оставить без изменений)
    checkpointer = CheckpointSaver(
        accelerator=accelerator,
        model=model,
        metric_name="MeanIoU",
        save_dir=Config.CHECKPOINTS_DIR,
        should_minimize=False
    )

    # Проверка аугментаций
    sample_img, sample_mask = train_dataset[0]
    #print("Уникальные значения в маске после трансформаций:", torch.unique(sample_mask))
    visualize_sample(train_dataset, "Augmented Sample", preprocess_flag=Config.PREPROCESS_FLAG)

    # Цикл обучения с обработкой ошибок
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0.0

        try:
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                try:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = loss_fn(outputs, masks.long())
                    accelerator.backward(loss)
                    optimizer.step()
                    epoch_loss += loss.item()
                except Exception as batch_error:
                    print(f"\nОшибка в батче: {batch_error}")
                    print(f"Размеры images: {images.shape}")
                    print(f"Размеры masks: {masks.shape}")
                    continue

        except Exception as epoch_error:
            print(f"\nКритическая ошибка в эпохе {epoch}: {epoch_error}")
            raise

        # Валидация и логирование (оставить без изменений)
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
            if epoch % 5 == 0:  # Логируем каждые 5 эпох, чтобы не перегружать
                fig = visualize_sample(val_dataset, f"Val Sample Epoch {epoch}", preprocess_flag=Config.PREPROCESS_FLAG)
                writer.add_figure("Validation Samples", fig, epoch)
                plt.close(fig)

            metric_fn.reset()

        writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)

        # Сохранение модели
        print("Saving checkpoint (val_iou == ", val_iou)
        checkpointer.save(val_iou, epoch)

    writer.close()
    print("Обучение завершено успешно!")

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