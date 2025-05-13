import cv2

if __name__ == "__main__":
    preprocessor = DermatologyPreprocessor(target_size=(512, 512))

    # Загрузка тестового изображения
    image = cv2.imread("PH2_Dataset/trainx/IMD002.bmp")
    mask = cv2.imread("PH2_Dataset/trainy/IMD002_lesion.bmp", cv2.IMREAD_GRAYSCALE)

    # Применение пайплайна
    processed_img, processed_mask = preprocessor(image, mask)

    # Визуализация
    cv2.imwrite("processed_image.jpg", (processed_img * 255).astype(np.uint8))
    if processed_mask is not None:
        cv2.imwrite("processed_mask.png", processed_mask)
