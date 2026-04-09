import pathlib

import torch

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 200
FINETUNE_EPOCHS = 50
MAX_LR = 2e-3
FINETUNE_LR = 2e-4
MIXUP_ALPHA = 0.3
CUTMIX_ALPHA = 1.0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = pathlib.Path("/kaggle/input/datasets/huynhthethien/radarcommunsignaldata2026train")
GROUP_ID = "04"
