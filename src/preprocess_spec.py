import cv2
import numpy as np
import logging
from typing import Tuple, Optional

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def crop_black_border(image: np.ndarray, threshold: int = 5) -> np.ndarray:
    """Обрезает чёрную рамку вокруг дерматоскопического изображения."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return image[y:y+h, x:x+w]


class HairRemovalLyakhov:
    def __init__(self, kernel_size_r=5, kernel_size_d=3, threshold=40):
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_r, kernel_size_r))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_d, kernel_size_d))
        self.threshold = threshold

    def __call__(self, image):
        # Разложение на каналы и обработка каждого
        channels = cv2.split(image)
        processed_channels = []

        for channel in channels:
            # Морфологическое закрытие
            closed = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, self.kernel_close)

            # Вычитание для выделения волос
            diff = cv2.subtract(closed, channel)

            # Пороговая бинаризация
            _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

            # Дилатация маски
            mask = cv2.dilate(mask, self.kernel_dilate, iterations=1)

            # Inpainting (Telea или Navier-Stokes)
            cleaned = cv2.inpaint(channel, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            processed_channels.append(cleaned)

        # Сборка обратно в RGB
        return cv2.merge(processed_channels)


class HairRemovalSalido:
    def __init__(self, median_kernel=3, bottomhat_kernel=(15, 15), threshold=5):
        self.median_kernel = median_kernel
        self.bottomhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, bottomhat_kernel)
        self.threshold = threshold

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Step 1: Median filtering per channel
        channels = cv2.split(image)
        processed_channels = []

        for channel in channels:
            # Median filter
            median = cv2.medianBlur(channel, self.median_kernel)

            # Bottom-hat transform
            bottomhat = cv2.morphologyEx(median, cv2.MORPH_BLACKHAT, self.bottomhat_kernel)

            # Binary thresholding (5%)
            _, mask = cv2.threshold(bottomhat, self.threshold, 255, cv2.THRESH_BINARY)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # Remove small objects
            mask = self._remove_small_objects(mask, min_size=90)

            # Harmonic inpainting (using Navier-Stokes)
            inpainted = cv2.inpaint(channel, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
            processed_channels.append(inpainted)

        return cv2.merge(processed_channels)

    def _remove_small_objects(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                mask[labels == i] = 0
        return mask


class DermatologyPreprocessor:
    def __init__(self):
        self.hair_removal = HairRemovalSalido()
        self.clahe_params = {'clipLimit': 2.0, 'tileGridSize': (8, 8)}

    def __normalize(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)/255.0
        return image

    def __call__(self, image, mask=None):
        try:
            # Step 0: Обрезаем чёрную рамку
            image = crop_black_border(image)
            
            # 1. Удаление артефактов (до аугментаций!)
            image = self.hair_removal(image)

            # 2. CLAHE (если нужно)
            clahe = cv2.createCLAHE(**self.clahe_params)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            image = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

            return self.__normalize(image), mask
        except Exception as e:
            logger.error(f"Ошибка в предобработке: {e}")
            raise
