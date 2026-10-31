import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=256, base=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.base = base

        # ---------- Encoder: image (1x128x128) -> small feature map (8*base x 4 x 4) ----------
        # Each conv uses kernel 4, stride 2, padding 1, which halves the height and width.
        self.enc1 = nn.Conv2d(1,        base,     4, 2, 1)   # 128 -> 64
        self.enc2 = nn.Conv2d(base,     base * 2, 4, 2, 1)   # 64  -> 32
        self.enc3 = nn.Conv2d(base * 2, base * 4, 4, 2, 1)   # 32  -> 16
        self.enc4 = nn.Conv2d(base * 4, base * 8, 4, 2, 1)   # 16  -> 8
        self.enc5 = nn.Conv2d(base * 8, base * 8, 4, 2, 1)   # 8   -> 4
        self.bn1 = nn.BatchNorm2d(base)
        self.bn2 = nn.BatchNorm2d(base * 2)
        self.bn3 = nn.BatchNorm2d(base * 4)
        self.bn4 = nn.BatchNorm2d(base * 8)
        self.bn5 = nn.BatchNorm2d(base * 8)

        flat_size = base * 8 * 4 * 4          # number of values after flattening (4096 for base=32)
        self.fc_mu = nn.Linear(flat_size, latent_dim)       # predicts the mean of z
        self.fc_logvar = nn.Linear(flat_size, latent_dim)   # predicts log(variance) of z

        # ---------- Decoder: z -> image ----------
        self.fc_dec = nn.Linear(latent_dim, flat_size)
        # Each transposed conv doubles the height and width.
        self.dec1 = nn.ConvTranspose2d(base * 8, base * 8, 4, 2, 1)   # 4  -> 8
        self.dec2 = nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1)   # 8  -> 16
        self.dec3 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)   # 16 -> 32
        self.dec4 = nn.ConvTranspose2d(base * 2, base,     4, 2, 1)   # 32 -> 64
        self.dec5 = nn.ConvTranspose2d(base,     1,        4, 2, 1)   # 64 -> 128
        self.dbn1 = nn.BatchNorm2d(base * 8)
        self.dbn2 = nn.BatchNorm2d(base * 4)
        self.dbn3 = nn.BatchNorm2d(base * 2)
        self.dbn4 = nn.BatchNorm2d(base)

    def encode(self, x):
        h = F.leaky_relu(self.bn1(self.enc1(x)), 0.2)
        h = F.leaky_relu(self.bn2(self.enc2(h)), 0.2)
        h = F.leaky_relu(self.bn3(self.enc3(h)), 0.2)
        h = F.leaky_relu(self.bn4(self.enc4(h)), 0.2)
        h = F.leaky_relu(self.bn5(self.enc5(h)), 0.2)
        h = h.flatten(1)                      # (batch, 8*base, 4, 4) -> (batch, flat_size)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # z = mean + std * random noise. This keeps the sampling step differentiable.
        std = torch.exp(0.5 * logvar)         # logvar = log(std^2), so std = exp(logvar / 2)
        noise = torch.randn_like(std)         # random numbers from N(0, 1)
        return mu + std * noise

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, self.base * 8, 4, 4)   # back to a 4x4 feature map
        h = F.relu(self.dbn1(self.dec1(h)))
        h = F.relu(self.dbn2(self.dec2(h)))
        h = F.relu(self.dbn3(self.dec3(h)))
        h = F.relu(self.dbn4(self.dec4(h)))
        return torch.sigmoid(self.dec5(h))    # sigmoid keeps pixel values between 0 and 1

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, n, device):
        """Make n new images from random z ~ N(0, 1). The encoder is not used."""
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            return self.decode(z)


def vae_loss(recon, x, mu, logvar, beta):
    batch_size = x.size(0)
    # 1) Reconstruction error: squared difference summed over all pixels, averaged over images
    recon_loss = F.mse_loss(recon, x, reduction="sum") / batch_size
    # 2) KL divergence between N(mu, std^2) and N(0, 1), summed over latent dims, averaged over images
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss
