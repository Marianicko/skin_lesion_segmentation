import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import random_split
import numpy as np
from preprocess_spec_2005 import DermatologyPreprocessor

IMAGE_SIZE = 512

# Определяем трансформации для train/val
train_transforms = A.Compose([
    # 1. Сначала приводим к общему размеру (сохраняя пропорции)
    A.LongestMaxSize(max_size=IMAGE_SIZE, p=1.0, interpolation=cv2.INTER_LINEAR),

    # 2. Добавляем padding до квадрата (чтобы RandomCrop не выходил за границы)
    A.PadIfNeeded(
        min_height=IMAGE_SIZE,
        min_width=IMAGE_SIZE,
        border_mode=cv2.BORDER_CONSTANT,
        position='random'
    ),

    # 3. Теперь безопасный RandomCrop
    A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),

    # 4. Аугментации
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(
        translate_percent=0.1,
        scale=(0.9, 1.1),  # Ограничим масштабирование
        rotate=(-15, 15),
        p=0.5
    ),
    ToTensorV2(),
], is_check_shapes=False)  # Явно отключаем проверку


val_transforms = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
    ToTensorV2(),
])


class PHDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, preprocess=False, crop_borders=True):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.bmp")))
        self.masks = sorted(glob.glob(os.path.join(masks_dir, "*_lesion.bmp")))
        self.transform = transform
        self.preprocess = preprocess
        self.preprocessor = DermatologyPreprocessor(debug=False, crop_borders=crop_borders) if preprocess else None
        assert len(self.images) == len(self.masks), "Кол-во изображений и масок не совпадает"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx, apply_transform=True):
        # Загрузка
        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask == 255).astype(np.uint8) * 1   # 0 & 1

        # Препроцессинг
        if self.preprocess:
            # Сохраняем копии для отладки
            original_image = image.copy()
            original_mask = mask.copy()

            # Препроцессинг (или нормализация)
            if self.preprocess:
                image, mask = self.preprocessor(image, mask)
            else:
                image = image.astype(np.float32) / 255.0

            # Проверка размеров
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError(f"Размеры не совпадают: image {image.shape}, mask {mask.shape}")

        # Аугментации
        if self.transform and apply_transform:
            transformed = self.transform(
                image=image,
                mask=mask.astype(np.float32)  # Для совместимости с albumentations
            )
            image, mask = transformed["image"], transformed["mask"]
            mask = (mask > 0.5).astype(np.uint8)  # Возвращаем бинарность

        return image, mask


def get_datasets(images_dir, masks_dir, val_ratio=0.2, test_ratio=0.1, seed=42, preprocess=False, crop_borders=True):
    """Правильное разделение на train/val/test с использованием Subset"""
    # Создаем полный датасет без трансформаций
    full_dataset = PHDataset(images_dir, masks_dir, transform=None, preprocess=preprocess, crop_borders=crop_borders)

    # Разделяем индексы
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_ratio)
    test_size = int(dataset_size * test_ratio)
    train_size = dataset_size - val_size - test_size

    # Разделяем датасет
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Присваиваем правильные трансформации
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = val_transforms

    return train_dataset, val_dataset, test_dataset

'''
# Пример использования для отладки
if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets(
        images_dir="../PH2_Dataset/trainx",
        masks_dir="../PH2_Dataset/trainy",
        preprocess=True
    )

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")
    img, mask = train_ds[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
'''
