
import json

import torch
import torch.nn.functional as F
from ignite.metrics import FID, InceptionScore
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import config as cfg
from data import get_datasets
from vae import VAE

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(cfg.SEED)

# Load the trained model.
model = VAE(cfg.LATENT_DIM, cfg.BASE_CHANNELS).to(device)
model.load_state_dict(torch.load(f"{cfg.OUT_DIR}/vae.pt", map_location=device))
model.eval()

# Real images (the training set) to compare against.
train_set, _ = get_datasets(cfg)
real_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)


def prepare(images):
    """Inception expects 3-channel 299x299 images, Copy the gray channel 3 times and resize."""
    images = images.repeat(1, 3, 1, 1)
    return F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)


fid_metric = FID(device=device)
is_metric = InceptionScore(device=device)

num_batches = cfg.N_GENERATED // cfg.BATCH_SIZE
real_iter = iter(real_loader)
for i in range(num_batches):
    fake = model.generate(cfg.BATCH_SIZE, device)

    try:
        real = next(real_iter)
    except StopIteration:               # ran out of real images: start again
        real_iter = iter(real_loader)
        real = next(real_iter)
    real = real.to(device)

    fake, real = prepare(fake), prepare(real)
    fid_metric.update((fake, real))     # FID compares fake vs real
    is_metric.update(fake)              # IS only looks at the fake images
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/{num_batches} batches done")

results = {"fid": float(fid_metric.compute()),
           "is": float(is_metric.compute()),
           "n_generated": num_batches * cfg.BATCH_SIZE}
print(results)
with open(f"{cfg.OUT_DIR}/metrics.json", "w") as f:
    json.dump(results, f, indent=1)

# A grid of 32 generated images for the report.
torch.manual_seed(cfg.SEED)
save_image(model.generate(32, device), f"{cfg.OUT_DIR}/generated_grid.png", nrow=8)
