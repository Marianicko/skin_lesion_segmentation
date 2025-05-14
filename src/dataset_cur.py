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

train_transforms = A.Compose([
    # 1. Ресайз с сохранением пропорций + обрезка по маске
    A.SmallestMaxSize(max_size=IMAGE_SIZE, p=1.0),
    A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),

    # 2. Геометрические аугментации
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=15,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5
    ),

    # 3. Нормализация (только для изображения)
    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ToTensorV2(),  # Конвертация в тензор

])

val_transforms = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask == 255).astype(np.uint8)  # Преобразует 255 в 1, остальное в 0
        # Преобразуем маску в тензор и проверяем значения
        mask_tensor = torch.from_numpy(mask)
        unique_values = torch.unique(mask_tensor)
        assert torch.all(torch.isin(unique_values, torch.tensor([0, 1]))), "Маска содержит недопустимые значения!"

        if self.preprocess and self.preprocessor:
            image, _ = self.preprocessor(image)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        return image, mask


def get_datasets(images_dir, masks_dir, val_ratio=0.2, test_ratio=0.1, seed=42, preprocess=False):
    """Автоматическое разделение на train/val/test."""
    full_dataset = PHDataset(images_dir, masks_dir, preprocess=preprocess)

    # Вычисляем размеры выборок
    val_size = int(len(full_dataset) * val_ratio)
    test_size = int(len(full_dataset) * test_ratio)
    train_size = len(full_dataset) - val_size - test_size

    # Разделяем
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Применяем трансформы
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = val_transforms  # Для теста используем val_transforms

    return train_dataset, val_dataset, test_dataset

'''
dataset = PHDataset(images_dir="../PH2_Dataset/trainx", masks_dir="../PH2_Dataset/trainy", transform=train_transforms)
img, mask = dataset[0]
plt.imshow(img.permute(1, 2, 0))
plt.show()

dataset = PHDataset(images_dir="../PH2_Dataset/trainx", masks_dir="../PH2_Dataset/trainy", transform=train_transforms)
img, mask = dataset[0]
print("Unique mask values:", torch.unique(mask))  # Должно быть только 0 и 1
plt.imshow(mask.squeeze(), cmap='gray')
plt.show()
'''