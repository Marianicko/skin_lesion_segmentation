import cv2
import numpy as np
import logging
from typing import Tuple, Optional

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DermatologyPreprocessor:
    """
    Pipeline for preprocessing dermatoscopic images
    1. Удаление волос
    2. Подавление пузырьков (считаем, что осн проблема - блики)
    3. CLAHE
    4. Стандартизация

    Параметры класса
    1) target_size: Tuple[int, int] - определяет размер, к которому будет приведено изображение после resize
    2) clahe_clip_limit: float - определяет степен усиления контраста с помощью CLAHE
    3) clahe_grid_size: Tuple[int, int] - размер сетки, на которую делится изображение для CLAHE
    4) hair_kernel: np.ndarray - ядро для морфологических операций при удалении волос
    5) bubble_threshold: int - порог для детекции пузырьков в V-канале HSV
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512),
                 clahe_clip_limit: float = 2.0,
                 clahe_grid_size: Tuple[int, int] = (8, 8)):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                                     tileGridSize=clahe_grid_size)

        # Для удаления волос нужны параметры, вот они: (пока что self_kernel нацелено на длинные "вертикальные" волосы)
        self.hair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 10))
        # Параметры для детекции пузырьков
        self.bubble_threshold = 240

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Основной метод - применяет pipeline к изображению и маске
        :param self:
        :param image:
        :param mask:
        :return:
        """
        try:
            # 1. Ресайз с сохранением пропорций
            # (lkz vfcrb ltkftv ghbyelbntkmysq d cbke njuj? xnj ,eltv bcgjkmpjdfnm ГТуе)
            '''
            image = self._resize(image)
            if mask is not None:
                mask = self._resize(mask, is_mask=True)
                if mask.shape[:2] != self.target_size:
                    mask = cv2.resize(
                        mask,
                        self.target_size,
                        interpolation=cv2.INTER_NEAREST  # Важно для масок!
                    )
            '''
            # 2. Удаление волос
            image_no_hairs = self._remove_hairs(image)

            # 3. Подавление пузырьков
            image_no_artifacts = self._remove_bubbles(image_no_hairs)

            # 4. Улучшение контраста (только для изображения! - маска и так бинарная)
            enhanced_image = self._apply_clahe(image_no_artifacts)

            # 5. Нормализация (стандартизация ImageNet)
            #normalized_image = self._normalize(enhanced_image)

            #return normalized_image, mask
            return enhanced_image, mask

        except Exception as e:
            logger.error(f"Ошибка в предобработке: {str(e)}")
            raise

    def _resize(self, img: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """
        Resize с сохранением пропорций и padding для приведения к target_size
        """
        h, w = img.shape[:2]    # (h, w, d), d -- channels
        scale = min(self.target_size[0] / w, self.target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h),
                             interpolation=cv2.INTER_AREA if is_mask else cv2.INTER_LINEAR)
        # Почему разная интерполяция? - да всё просто: для маски критически важно не допустить
        # размытия границ, поэтому билинейную использовать ну никак нельзя

        # Добавляем padding (если нужно - вдруг при сохраняющем пропорции resize изображение "не впишется" в target_size)
        if not is_mask:
            pad_x = self.target_size[0] - new_w # изображение уменьшаем, но не увеличиваем
            pad_y = self.target_size[1] - new_h
            resized = cv2.copyMakeBorder(resized, 0, pad_y, 0, pad_x,
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return resized

    def _remove_hairs(self, image: np.ndarray) -> np.ndarray:
        """
        Удаление волос через морфологические операции:
        1. Black-hat трансформация для выделения тёмных линейных структур (!)
        (!) - для светлых волос можно рассмотреть вариант алгоритма Ляховых
        2. Пороговая бинаризация
        3. Inpainting для заполнения
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Black-hat трансформация
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.hair_kernel)

        # Бинаризация волос, порог - 10, значение из статьи, подобрано эмпирически
        # Поправим, если плохо будет справляться с тёмными волосами, перекрывающими тёмные родинки
        _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        # Удаление мелких артефактов
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Inpainting (Telea) - почему Telea?
        result = cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        logger.debug(f"Удалено волос: {np.sum(hair_mask > 0)} пикселей")
        return result

    def _remove_bubbles(self, image: np.ndarray) -> np.ndarray:
        """
                Детекция и удаление пузырьков/бликов:
                1. Переход в HSV-пространство (блики имеют близкие к 255 значения в канале V)
                2. Пороговая бинаризация по V-каналу
                3. Inpainting (Взяли навье-стокса)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)

        # Детекция бликов
        _, bubble_mask = cv2.threshold(v, self.bubble_threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow(bubble_mask)

        bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        #cv2.imshow(bubble_mask)

        result = cv2.inpaint(image, bubble_mask, 2, cv2.INPAINT_NS)
        logger.debug(f"Удалено пузырьков: {np.sum(bubble_mask > 0)} (в пикселях)")
        return result

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Применение CLAHE к L-каналу в LAB-пространстве"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # здесь пробую применить bilateral filter, чтобы избежать "засветов", связанных с растяжением гистограммы
        l = cv2.bilateralFilter(l, d=9, sigmaColor=75, sigmaSpace=75)
        # CLAHE только для L-канала
        l_clahe = self.clahe.apply(l)

        merged = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Стандартизация по mean/std ImageNet"""
        image = image.astype(np.float32) / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (image - mean) / std

    def _debug_visualize(self, step_name: str, image: np.ndarray):
        if logger.level == logging.DEBUG:
            cv2.imwrite(f"debug_{step_name}.jpg", image)

