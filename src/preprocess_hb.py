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
    return image[y:y + h, x:x + w]


class HairRemovalSalido:
    def __init__(self,
                 median_kernel: int = 3,
                 bottomhat_kernel: Tuple[int, int] = (15, 15),
                 threshold: int = 5,
                 guided_radius: int = 10,
                 guided_eps: float = 0.01):
        self.median_kernel = median_kernel
        self.bottomhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, bottomhat_kernel)
        self.threshold = threshold
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps

    def __call__(self, image: np.ndarray) -> np.ndarray:
        channels = cv2.split(image)
        processed_channels = []

        for channel in channels:
            # Median filter
            median = cv2.medianBlur(channel, self.median_kernel)

            # Bottom-hat transform
            bottomhat = cv2.morphologyEx(median, cv2.MORPH_BLACKHAT, self.bottomhat_kernel)

            # Binary thresholding (Otsu's method)
            _, mask = cv2.threshold(bottomhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = self._remove_small_objects(mask, min_size=90)

            # Telea Inpainting
            inpainted = cv2.inpaint(channel, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # Guided Filter to preserve edges
            if cv2.__version__ >= '4.5.1':
                guided = cv2.ximgproc.guidedFilter(
                    guide=channel,
                    src=inpainted,
                    radius=self.guided_radius,
                    eps=self.guided_eps
                )
            else:
                guided = inpainted  # Fallback if OpenCV < 4.5.1
            processed_channels.append(guided)

        return cv2.merge(processed_channels)

    def _remove_small_objects(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                mask[labels == i] = 0
        return mask


class BubbleRemoval:
    def __init__(self,
                 dog_sigma1: float = 1.0,
                 dog_sigma2: float = 2.0,
                 nlm_h: float = 10.0,
                 nlm_template_size: int = 7,
                 nlm_search_size: int = 21):
        self.dog_sigma1 = dog_sigma1
        self.dog_sigma2 = dog_sigma2
        self.nlm_h = nlm_h
        self.nlm_template_size = nlm_template_size
        self.nlm_search_size = nlm_search_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale for bubble detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Difference of Gaussians (DoG) for bubble detection
        gauss1 = cv2.GaussianBlur(gray, (0, 0), self.dog_sigma1)
        gauss2 = cv2.GaussianBlur(gray, (0, 0), self.dog_sigma2)
        dog = gauss1 - gauss2
        _, bubble_mask = cv2.threshold(np.uint8(np.abs(dog)), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Cleanup mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_OPEN, kernel)

        # Apply Non-Local Means (NLM) only to masked regions
        if cv2.__version__ >= '4.5.1':
            nlm = cv2.fastNlMeansDenoisingColored(image, None, self.nlm_h, self.nlm_h,
                                                 self.nlm_template_size, self.nlm_search_size)
            # Blend original and NLM-filtered image using the mask
            result = np.where(bubble_mask[..., None] > 0, nlm, image)
        else:
            result = image  # Fallback if OpenCV < 4.5.1
        return result


class DermatologyPreprocessor:
    def __init__(self):
        self.hair_removal = HairRemovalSalido()
        self.bubble_removal = BubbleRemoval()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, image, mask=None):
        try:
            # Step 0: Обрезаем чёрную рамку
            image = crop_black_border(image)

            # 1. Удаление артефактов (до аугментаций!)

            image = self.hair_removal(image)
            image = self.bubble_removal(image)

            # 2. CLAHE (если нужно)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe.apply(l)
            image = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

            return image, mask
        except Exception as e:
            logger.error(f"Ошибка в предобработке: {e}")
            raise