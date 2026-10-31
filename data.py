import glob
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class XrayDataset(Dataset):

    def __init__(self, root, splits, img_size):
        self.files = []
        for split in splits:
            pattern = os.path.join(root, split, "*", "*.jp*g")   # matches .jpg and .jpeg
            self.files += sorted(glob.glob(pattern))
        if len(self.files) == 0:
            raise FileNotFoundError("No images found in " + root + " for " + str(splits))
        self.transform = transforms.Compose([
            transforms.Grayscale(1),                       # 1 channel
            transforms.Resize((img_size, img_size)),       # 128 x 128
            transforms.ToTensor(),                         # pixel values scaled to [0, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index])
        return self.transform(image)


class FakeDataset(Dataset):

    def __init__(self, n, img_size, seed):
        gen = torch.Generator().manual_seed(seed)
        small = torch.rand(n, 1, 16, 16, generator=gen)
        self.images = F.interpolate(small, size=img_size, mode="bilinear")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


def get_datasets(cfg):
    if cfg.SYNTHETIC:
        return FakeDataset(256, cfg.IMG_SIZE, 0), FakeDataset(64, cfg.IMG_SIZE, 1)
    train_set = XrayDataset(cfg.DATA_ROOT, ["train"], cfg.IMG_SIZE)
    val_set = XrayDataset(cfg.DATA_ROOT, ["val", "test"], cfg.IMG_SIZE)   # held-out images
    return train_set, val_set
