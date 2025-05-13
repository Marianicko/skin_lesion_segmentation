import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset_upd import PHDataset
from preprocess_spec import DermatologyPreprocessor
from dataset_upd import train_transforms

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def test_preprocessing():
    global mean, std
    # Пути к данным
    images_dir = "../PH2_Dataset/trainx"
    masks_dir = "../PH2_Dataset/trainy"

    # Инициализация датасета
    dataset = PHDataset(images_dir=images_dir, masks_dir=masks_dir, transform=None)

    # Инициализация препроцессора
    preprocessor = DermatologyPreprocessor()

    for ix in range(4, 5):
        # Загрузка и предобработка одного примера
        image, mask = dataset[ix]  # Берём первый элемент датасета
        processed_image, processed_mask = preprocessor(image, mask)
        processed_image_norm = (processed_image - mean)/std
        # Визуализация исходного и обработанного изображения
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.title("Original Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.title("Processed Image")
        # Конвертация обратно в RGB для отображения (если нужно)
        processed_image_display = (processed_image - processed_image.min()) / (processed_image.max() - processed_image.min())
        plt.imshow(processed_image_display)
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.title("Processed Mask")
        processed_image_norm_display = (processed_image_norm - processed_image_norm.min()) / (processed_image_norm.max() - processed_image_norm.min())
        plt.imshow(processed_image_norm_display)
        plt.axis('off')
        '''
        if processed_mask is not None:
            plt.imshow(processed_mask, cmap='gray')
        plt.axis('off')
        '''

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_preprocessing()