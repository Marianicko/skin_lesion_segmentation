import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, random_split
import numpy as np
from preprocess_spec import DermatologyPreprocessor

IMAGE_SIZE = 512  # или ваш target_size

# Определяем трансформации для train/val
train_transforms = A.Compose([
    A.SmallestMaxSize(max_size=IMAGE_SIZE, p=1.0),
    A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=15,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5
    ),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class PHDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, preprocess=False):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.bmp")))
        self.masks = sorted(glob.glob(os.path.join(masks_dir, "*_lesion.bmp")))
        self.transform = transform
        self.preprocess = preprocess
        self.preprocessor = DermatologyPreprocessor() if preprocess else None
        assert len(self.images) == len(self.masks), "Кол-во изображений и масок не совпадает"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx, apply_transform=True):
        # Загрузка изображения и маски
        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask == 255).astype(np.uint8)  # Преобразует 255 в 1, остальное в 0

        # Предобработка (если включена)
        if self.preprocess:
            image, _ = self.preprocessor(image)
        else:
            image = image.astype(np.float32) / 255.0

        # Применение трансформаций
        if self.transform and apply_transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask


def get_datasets(images_dir, masks_dir, val_ratio=0.2, test_ratio=0.1, seed=42, preprocess=False):
    """Автоматическое разделение на train/val/test с поддержкой предобработки."""
    # Создаем отдельные датасеты для каждой части
    full_dataset = PHDataset(images_dir, masks_dir, transform=None, preprocess=preprocess)

    # Разделяем индексы
    indices = list(range(len(full_dataset)))
    np.random.seed(seed)
    np.random.shuffle(indices)

    val_size = int(len(full_dataset) * val_ratio)
    test_size = int(len(full_dataset) * test_ratio)
    train_size = len(full_dataset) - val_size - test_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Создаем датасеты с правильными трансформациями
    train_dataset = PHDataset(images_dir, masks_dir, transform=train_transforms, preprocess=preprocess)
    val_dataset = PHDataset(images_dir, masks_dir, transform=val_transforms, preprocess=preprocess)
    test_dataset = PHDataset(images_dir, masks_dir, transform=val_transforms, preprocess=preprocess)

    # Фильтруем данные по индексам
    train_dataset.images = [train_dataset.images[i] for i in train_indices]
    train_dataset.masks = [train_dataset.masks[i] for i in train_indices]

    val_dataset.images = [val_dataset.images[i] for i in val_indices]
    val_dataset.masks = [val_dataset.masks[i] for i in val_indices]

    test_dataset.images = [test_dataset.images[i] for i in test_indices]
    test_dataset.masks = [test_dataset.masks[i] for i in test_indices]

    return train_dataset, val_dataset, test_dataset

# Пример использования для отладки
if __name__ == "__main__":
    # Тестовый прогон
    train_ds, val_ds, test_ds = get_datasets(
        images_dir="../PH2_Dataset/trainx",
        masks_dir="../PH2_Dataset/trainy",
        preprocess=True
    )

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")

    # Проверка первого элемента
    img, mask = train_ds[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")  # Должно быть [0, 1]