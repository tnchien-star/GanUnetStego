import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from . import config
from .coding import rs_encode_bits

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# convert to tensor, and normalize to [-1, 1]
def build_transform(image_size=None):
    image_size = config.IMAGE_SIZE if image_size is None else image_size
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def list_images(root_dir, sort=True):
    paths = []
    for root, _, files in os.walk(root_dir):        
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, filename))

    if not paths:
        raise RuntimeError(
            "No image found in '{}'".format(root_dir)
        )
    return sorted(paths) if sort else paths


def to_tensor(path, transform=None, device=None):
    transform = build_transform() if transform is None else transform
    tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0)
    return tensor if device is None else tensor.to(device)


def to_pil(tensor):
    if tensor.dim() == 4:          
        tensor = tensor[0]
    array = ((tensor.detach().float().cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0)
    array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


class SteganographyDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images=10000, validation=False):
        self.image_paths = list_images(root_dir, sort=True)
        random.shuffle(self.image_paths)

        split = int(len(self.image_paths) * 0.9)
        if validation:
            self.image_paths = self.image_paths[split:][:500]
        else:
            self.image_paths = self.image_paths[:split][:max_images]

        self.transform = build_transform() if transform is None else transform
        print("{}: {} images".format("Val" if validation else "Train", len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        try:
            image = Image.open(self.image_paths[index]).convert("RGB")
            if self.transform:
                image = self.transform(image)

            payload = np.random.randint(0, 2, config.PAYLOAD_BITS).astype(np.float32)
            coded = rs_encode_bits(payload, config.N_SYMBOLS)

            return image, torch.from_numpy(payload), torch.from_numpy(coded)
        except Exception:
            return self.__getitem__((index + 1) % len(self))
