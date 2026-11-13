import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)                       
    np.random.seed(seed)                    
    torch.manual_seed(seed)                 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)    


def calculate_psnr(img1, img2):
    mse = torch.mean((img1.float() - img2.float()) ** 2)
    if mse > 0:
        return 20 * torch.log10(torch.tensor(2.0) / torch.sqrt(mse))
    return torch.tensor(float("inf"))


def per_image_psnr(img1, img2):
    mse = torch.mean((img1.float() - img2.float()) ** 2, dim=[1, 2, 3]).clamp_min(1e-12)
    return 20 * torch.log10(2.0 / torch.sqrt(mse))


def robust_bit_loss(logits, targets, hard_fraction=0.20, margin=2.0):
    per_bit = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    mean_loss = per_bit.mean()
    k = max(1, int(per_bit.size(1) * hard_fraction))
    hard_loss = per_bit.topk(k, dim=1).values.mean()
    signed_targets = targets.mul(2.0).sub(1.0)
    margin_loss = F.relu(margin - signed_targets * logits).mean()
    return 0.70 * mean_loss + 0.20 * hard_loss + 0.10 * margin_loss
