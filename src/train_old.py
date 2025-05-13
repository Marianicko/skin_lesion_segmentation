import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset_upd import get_datasets  # Ваша функция для разделения данных
from model import SegmentationModel
from config import Config
from tqdm import tqdm
from checkpointer import CheckpointSaver
from metric import MeanIoU
from loss import CrossEntropyDiceLoss
from accelerate import Accelerator
from utils import seed_everything


def train():
    # Инициализация
    accelerator = Accelerator(cpu=Config.DEVICE == "cpu")
    model = SegmentationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    # Загрузка данных
    train_dataset, val_dataset, test_dataset = get_datasets(
        images_dir="PH2_Dataset/trainx",
        masks_dir="PH2_Dataset/trainy",
        val_ratio=0.15,
        test_ratio=0.05,
        seed=42
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1)  # Для теста

    # Подготовка метрик и loss
    class_weights = torch.tensor([0.3, 0.7]).to(Config.DEVICE)  # Веса для фона и невуса
    loss_fn = CrossEntropyDiceLoss(weight=class_weights, ignore_index=-1)
    metric_fn = MeanIoU(classes_num=Config.NUM_CLASSES, ignore_index=-1).to(Config.DEVICE)

    # Чекпоинтер
    checkpointer = CheckpointSaver(
        accelerator=accelerator,
        model=model,
        metric_name="MeanIoU",
        save_dir=Config.CHECKPOINTS_DIR,
        should_minimize=False  # Максимизируем IoU
    )

    # Подготовка для Accelerate
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Цикл обучения
    for epoch in range(Config.EPOCHS):
        model.train()
        for images, masks in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks.long())
            accelerator.backward(loss)
            optimizer.step()

        # Валидация
        model.eval()
        val_metric = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                outputs = model(images)
                metric_fn.update(outputs, masks.long())
            val_metric = metric_fn.compute().item()
            metric_fn.reset()

        # Сохранение чекпоинта
        checkpointer.save(val_metric, epoch)

    # Тестирование (опционально)
    test_metric = evaluate(model, test_loader, metric_fn)
    print(f"Test MeanIoU: {test_metric:.4f}")


def evaluate(model, loader, metric_fn):
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            outputs = model(images.to(Config.DEVICE))
            metric_fn.update(outputs, masks.long().to(Config.DEVICE))
        return metric_fn.compute().item()