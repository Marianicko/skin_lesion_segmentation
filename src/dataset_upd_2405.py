import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import random_split
import numpy as np
from preprocess_spec_2305 import DermatologyPreprocessor

IMAGE_SIZE = 512
train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),  # Фиксированный размер
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.MedianBlur(blur_limit=3, p=0.2)
    ], p=0.3),  # Общая вероятность применения любого блюра
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    ToTensorV2(),
], is_check_shapes=False)

val_transforms = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
    ToTensorV2(),
])


class PHDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, preprocess=False, crop_borders=True, target_size=(512, 512)):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.bmp")))
        self.masks = sorted(glob.glob(os.path.join(masks_dir, "*_lesion.bmp")))
        self.transform = transform
        self.preprocess = preprocess
        self.preprocessor = DermatologyPreprocessor(debug=False, crop_borders=crop_borders, target_size=target_size) if preprocess else None
        assert len(self.images) == len(self.masks), "Кол-во изображений и масок не совпадает"
        self.target_size = target_size


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx, apply_transform=True):
        # Загрузка изображения и маски
        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask == 255).astype(np.uint8) * 1

        # Препроцессинг (если нужен)
        if self.preprocess:
            image, mask = self.preprocessor(image, mask)
        else:
            image = image.astype(np.float32) / 255.0

        # Проверка размеров
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Размеры не совпадают: image {image.shape}, mask {mask.shape}")

        # Аугментации
        if self.transform and apply_transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

            # Явное копирование и преобразование маски
            if torch.is_tensor(mask):
                mask = mask.clone().detach().long()  # Важно: копируем и меняем тип
                mask = (mask > 0.5).long()

        return image, mask


def get_datasets(images_dir, masks_dir, val_ratio=0.2, test_ratio=0.1, seed=42, preprocess=False, crop_borders=True):
    """Разделение данных на train/val/test с корректным применением трансформаций."""
    # Создаём отдельные датасеты с разными трансформациями
    print("It's okay")
    train_dataset = PHDataset(
        images_dir, masks_dir,
        transform=train_transforms,  # Аугментации для обучения
        preprocess=preprocess,
        crop_borders=crop_borders
    )
    val_dataset = PHDataset(
        images_dir, masks_dir,
        transform=val_transforms,  # Только resize для валидации
        preprocess=preprocess,
        crop_borders=crop_borders
    )
    test_dataset = PHDataset(
        images_dir, masks_dir,
        transform=val_transforms,  # Только resize для теста
        preprocess=preprocess,
        crop_borders=crop_borders
    )

    # Разделяем индексы вручную (без random_split)
    indices = list(range(len(train_dataset)))  # Все датасеты имеют одинаковую длину
    train_size = int((1 - val_ratio - test_ratio) * len(indices))
    val_size = int(val_ratio * len(indices))
    test_size = len(indices) - train_size - val_size

    # Перемешиваем индексы для воспроизводимости
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Создаём Subset с уже назначенными трансформациями
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    return train_subset, val_subset, test_subset


# В конце файла dataset_upd_2005.py, перед последними строками:

def check_subset_behavior(images_dir, masks_dir):
    print("\n=== Testing Subset behavior ===")
    full_dataset = PHDataset(images_dir, masks_dir, preprocess=True)
    subset = torch.utils.data.Subset(full_dataset, indices=[0, 1, 2])

    for i in range(3):
        item = subset[i]
        #print(f"Item {i} types: {type(item[0])}, {type(item[1])}")


def test_dataloader(images_dir, masks_dir):
    print("\n=== Testing DataLoader ===")
    dataset = PHDataset(images_dir, masks_dir,
                        preprocess=True,
                        target_size=(512, 512))  # Фиксированный размер
    loader = DataLoader(dataset, batch_size=2)

    for batch in loader:
        #print(f"Batch image sizes: {batch[0].shape}")
        #print(f"Batch mask sizes: {batch[1].shape}")
        break  # Проверяем только первый батч


def check_types_compatibility(images_dir, masks_dir):
    print("\n=== Testing Type Compatibility ===")
    train_ds, _, _ = get_datasets(images_dir, masks_dir, preprocess=True)
    sample = train_ds[0]

    print(f"Sample types: {type(sample[0])}, {type(sample[1])}")

    if torch.is_tensor(sample[0]):
        print(f"Image tensor dtype: {sample[0].dtype}")
    if torch.is_tensor(sample[1]):
        print(f"Mask tensor dtype: {sample[1].dtype}")

    try:
        if isinstance(sample[1], np.ndarray):
            _ = sample[1].astype(np.float32)
        print("Numpy conversion OK")
    except Exception as e:
        print(f"Numpy conversion failed: {e}")

    try:
        if torch.is_tensor(sample[1]):
            _ = (sample[1] > 0.5).long()
        print("Tensor conversion OK")
    except Exception as e:
        print(f"Tensor conversion failed: {e}")

def test_sizes(images_dir, masks_dir):
    dataset = PHDataset(images_dir, masks_dir, preprocess=True)
    for i in range(3):
        img, mask = dataset[i]
        #print(f"Item {i} sizes - image: {img.shape}, mask: {mask.shape}")
        assert img.shape[:2] == mask.shape[:2], "Size mismatch"



if __name__ == "__main__":
    # Пути к данным (замените на актуальные)
    IMAGES_DIR = "../PH2_Dataset/trainx"
    MASKS_DIR = "../PH2_Dataset/trainy"

    # Вызываем тестовые функции
    check_subset_behavior(IMAGES_DIR, MASKS_DIR)
    test_dataloader(IMAGES_DIR, MASKS_DIR)
    check_types_compatibility(IMAGES_DIR, MASKS_DIR)

    # Оригинальный тестовый код
    train_ds, val_ds, test_ds = get_datasets(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        preprocess=True,
        crop_borders=False
    )
    #print(f"\nOriginal test:")
    #print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")
    img, mask = train_ds[0]
    #print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    #print(f"Mask unique values: {torch.unique(mask)}")
