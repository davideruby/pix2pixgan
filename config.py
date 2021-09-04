import torch

DEVICE = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu"
ROOT_DIR = "dataset/data"
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
NUM_WORKERS = 0
CHANNELS_IMG = 3
L1_LAMBDA = 75
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.999
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth"
CHECKPOINT_GEN = "gen.pth"
SMOOTH_POSITIVE_LABELS = True
SMOOTH_NEGATIVE_LABELS = True
LOG_WANDB = True
HE_NORM = True
SEED = 42
NGPU = 2
NUM_NODES = 1
