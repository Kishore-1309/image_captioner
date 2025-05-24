import torch

class Config:
    # Paths
    DEFAULT_FEATURE_DIR = "features"
    DEFAULT_CHECKPOINT_DIR = "checkpoints"
    
    # Model
    IMAGE_FEATURE_DIM = 512
    GPT2_HIDDEN_DIM = 768
    NUM_HEADS = 8
    NUM_LAYERS = 2
    DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 5e-5
    MAX_LENGTH = 40
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")