import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import config as cfg
from data import get_datasets
from vae import VAE, vae_loss

torch.manual_seed(cfg.SEED)
os.makedirs(cfg.OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------- Data ----------
train_set, val_set = get_datasets(cfg)
train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False)
print("Training images:", len(train_set), " Held-out images:", len(val_set))

# ---------- Model and optimizer ----------
model = VAE(cfg.LATENT_DIM, cfg.BASE_CHANNELS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

# We store the average loss of every epoch so we can plot it later.
history = {"train_loss": [], "train_rec": [], "train_kl": [],
           "val_loss": [], "val_rec": [], "val_kl": []}

for epoch in range(1, cfg.EPOCHS + 1):
    # KL warm-up: beta starts small and reaches BETA after WARMUP_EPOCHS.
    beta = cfg.BETA * min(1.0, epoch / cfg.WARMUP_EPOCHS)

    # ----- Training pass -----
    model.train()
    sum_loss, sum_rec, sum_kl, count = 0.0, 0.0, 0.0, 0
    for images in train_loader:
        images = images.to(device)
        recon, mu, logvar = model(images)
        loss, rec, kl = vae_loss(recon, images, mu, logvar, beta)

        optimizer.zero_grad()     # clear gradients from the previous step
        loss.backward()           # compute gradients
        optimizer.step()          # update the weights

        n = images.size(0)
        sum_loss += loss.item() * n
        sum_rec += rec.item() * n
        sum_kl += kl.item() * n
        count += n
    history["train_loss"].append(sum_loss / count)
    history["train_rec"].append(sum_rec / count)
    history["train_kl"].append(sum_kl / count)

    # ----- Validation pass (no weight updates) -----
    model.eval()
    sum_loss, sum_rec, sum_kl, count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
            recon, mu, logvar = model(images)
            loss, rec, kl = vae_loss(recon, images, mu, logvar, cfg.BETA)   # full beta
            n = images.size(0)
            sum_loss += loss.item() * n
            sum_rec += rec.item() * n
            sum_kl += kl.item() * n
            count += n
    history["val_loss"].append(sum_loss / count)
    history["val_rec"].append(sum_rec / count)
    history["val_kl"].append(sum_kl / count)

    print(f"Epoch {epoch:3d}  train loss {history['train_loss'][-1]:.1f}  "
          f"(recon {history['train_rec'][-1]:.1f}, KL {history['train_kl'][-1]:.1f})  "
          f"val loss {history['val_loss'][-1]:.1f}")

    # Save a grid of generated images every 10 epochs so you can watch progress.
    if epoch % 10 == 0 or epoch == cfg.EPOCHS:
        model.eval()
        save_image(model.generate(64, device), f"{cfg.OUT_DIR}/samples_epoch{epoch}.png", nrow=8)

# ---------- Save everything ----------
torch.save(model.state_dict(), f"{cfg.OUT_DIR}/vae.pt")

settings = {"epochs": cfg.EPOCHS, "batch_size": cfg.BATCH_SIZE, "lr": cfg.LEARNING_RATE,
            "latent_dim": cfg.LATENT_DIM, "base": cfg.BASE_CHANNELS, "beta": cfg.BETA,
            "kl_warmup": cfg.WARMUP_EPOCHS, "img_size": cfg.IMG_SIZE, "seed": cfg.SEED}
with open(f"{cfg.OUT_DIR}/history.json", "w") as f:
    json.dump({"args": settings, "history": history}, f, indent=1)

# ---------- Loss curves ----------
epochs = range(1, cfg.EPOCHS + 1)
fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
panels = [("Total loss (neg. ELBO)", "loss"), ("Reconstruction error", "rec"), ("KL divergence", "kl")]
for ax, (title, key) in zip(axes, panels):
    ax.plot(epochs, history["train_" + key], label="train")
    ax.plot(epochs, history["val_" + key], label="val")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("epoch", fontsize=8)
    ax.tick_params(labelsize=7)
axes[0].legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{cfg.OUT_DIR}/loss_curves.pdf")
plt.savefig(f"{cfg.OUT_DIR}/loss_curves.png", dpi=150)

# ---------- Reconstructions: top row = real images, bottom row = VAE output ----------
model.eval()
originals = next(iter(val_loader))[:8].to(device)
with torch.no_grad():
    recons, _, _ = model(originals)
save_image(torch.cat([originals, recons]), f"{cfg.OUT_DIR}/recon.png", nrow=8)
print("Done. Results are in", cfg.OUT_DIR)
