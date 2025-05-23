import torch


class Config:
    # Данные
    IMAGE_SIZE = 512  # Совпадает с dataset_upd.py
    BATCH_SIZE = 4    # Для маленького датасета PH2 можно 4-8 (в датасете 200 изобр-й)
    NUM_WORKERS = 2   # Параллельная загрузка данных
    NUM_CLASSES = 2   # Фон + невус
    PREPROCESS_FLAG = True  # Включение предобработки

    # Предобработка
    BOTTOMHAT_KERNEL = (9, 9)
    THRESHOLD_HAIR_REMOVAL = 10
    CLAHE_CLIP_LIMIT = 1.0
    CLAHE_TILE_GRID_SIZE = (16, 16)

    # Модель
    IN_CHANNELS = 3   # RGB
    BILINEAR = True   # Билинейный апсемплинг
    CLASS_WEIGHTS = [0.5, 0.5]

    # Обучение
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LR = 1e-4         # Для Adam
    EPOCHS = 50       # Ранняя остановка может прервать раньше
    WEIGHT_DECAY = 1e-5  # L2-регуляризация

    # Пути
    CHECKPOINTS_DIR = "checkpoints"
    LOGS_DIR = "logs"
    IMAGES_DIR = "./PH2_Dataset/trainx"
    MASKS_DIR = "./PH2_Dataset/trainy"





config = Config()