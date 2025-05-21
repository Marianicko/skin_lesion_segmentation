import torch


class Config:
    # Данные
    IMAGE_SIZE = 512  # Совпадает с dataset_upd.py
    BATCH_SIZE = 4    # Для маленького датасета PH2 можно 4-8 (в датасете 200 изобр-й)
    NUM_WORKERS = 2   # Параллельная загрузка данных
    NUM_CLASSES = 2   # Фон + невус

    # Модель
    IN_CHANNELS = 3   # RGB
    BILINEAR = True   # Билинейный апсемплинг

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

    PREPROCESS_FLAG = True

config = Config()