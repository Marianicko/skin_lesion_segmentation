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
        #print(f"\n--- Processing item {idx} ---")

        # Загрузка
        print("Loading images...")
        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        #print(f"Initial types - image: {type(image)}, mask: {type(mask)}")

        mask = (mask == 255).astype(np.uint8) * 1
        #print(f"After binarization - mask type: {type(mask)}, dtype: {mask.dtype}")

        # Препроцессинг
        if self.preprocess:
            print("Applying preprocessing...")
            image, mask = self.preprocessor(image, mask)
            #print(f"After preprocessing - image type: {type(image)}, mask type: {type(mask)}")
        else:
            image = image.astype(np.float32) / 255.0

        # Проверка размеров
        if isinstance(image, np.ndarray) and isinstance(mask, np.ndarray):
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError(f"Shape mismatch: image {image.shape}, mask {mask.shape}")

        # Аугментации
        if self.transform and apply_transform:
            print("Applying transforms...")
            #print(f"Before transform - image type: {type(image)}, mask type: {type(mask)}")
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
            #print(f"After transform - image type: {type(image)}, mask type: {type(mask)}")

            if torch.is_tensor(mask):
                #print(f"Mask tensor values before binarization: {torch.unique(mask)}")
                mask = (mask > 0.5).long()
                #print(f"Mask tensor values after binarization: {torch.unique(mask)}")

        if self.target_size and not self.transform:
            # Если трансформации нет, делаем resize здесь
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        #print(f"Final types - image: {type(image)}, mask: {type(mask)}")
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

    #print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")
    img, mask = train_ds[0]
    #print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    #print(f"Mask unique values: {torch.unique(mask)}")
'''


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
