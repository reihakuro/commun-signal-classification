import random

import numpy as np
import torch


def set_seed(seed=42):
    """Fix all random seeds to ensure fully reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = int(np.clip(cx - cut_w // 2, 0, W))
    y1 = int(np.clip(cy - cut_h // 2, 0, H))
    x2 = int(np.clip(cx + cut_w // 2, 0, W))
    y2 = int(np.clip(cy + cut_h // 2, 0, H))
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(2), x.size(3), lam)
    mixed = x.clone()
    mixed[:, :, x1:x2, y1:y2] = x[idx, :, x1:x2, y1:y2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / float(x.size(2) * x.size(3))
    return mixed, y, y[idx], lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
