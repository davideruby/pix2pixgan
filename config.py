import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu"
# TRAIN_DIR = "data/train"
# VAL_DIR = "data/val"
ROOT_DIR = "data"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
VIRTUAL_BATCH_SIZE = 81
NUM_WORKERS = 0
CHANNELS_IMG = 3
L1_LAMBDA = 100
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.999
NUM_EPOCHS = 30
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth"
CHECKPOINT_GEN = "gen.pth"
SMOOTH_POSITIVE_LABELS = True
SMOOTH_NEGATIVE_LABELS = True
LOG_WANDB = True
SEED = 42
NGPU = 2
NUM_NODES = 1

transform_training = A.Compose(
    [
        A.Flip(p=0.75),
        A.RandomRotate90(p=0.75),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)

transform_test = A.Compose(
    [
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)
