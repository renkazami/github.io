import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


@dataclass
class HyperParams:
    """Container for hyperparameters."""

    lr: float = 1e-3
    batch_size: int = 8
    epochs: int = 10
    latent_dim: int = 128
    seq_len: int = 5
    frame_size: int = 64
    lambda_recon: float = 1.0
    lambda_grad: float = 0.01
    beta: float = 1e-4
    model_path: str = "./vae_plasma_model.pt"
    data_path: str = "./plasma_simulation_data.pt"  # path to training data


class PlasmaDataset(Dataset):
    """Dataset generating sequences of frames for VAE training."""

    def __init__(self, data: torch.Tensor, seq_len: int):
        if data.ndim != 3:
            raise ValueError("Data must have shape (num_frames, height, width)")
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int):
        frames = self.data[idx : idx + self.seq_len]  # (seq_len, H, W)
        frames = torch.stack(frames, dim=0)
        x = frames.unsqueeze(0)  # (1, seq_len, H, W)
        target = self.data[idx + self.seq_len // 2].unsqueeze(0)
        return x.float(), target.float()


class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

    def forward(self, x):
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 64, 8, 8)
        return self.deconv(x)


class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(recon, target, mu, logvar, lambda_recon, lambda_grad, beta):
    mse_loss = F.mse_loss(recon, target)

    grad_recon_x = recon[:, :, 1:, :] - recon[:, :, :-1, :]
    grad_target_x = target[:, :, 1:, :] - target[:, :, :-1, :]
    grad_recon_y = recon[:, :, :, 1:] - recon[:, :, :, :-1]
    grad_target_y = target[:, :, :, 1:] - target[:, :, :, :-1]
    grad_loss = F.mse_loss(grad_recon_x, grad_target_x) + F.mse_loss(
        grad_recon_y, grad_target_y
    )

    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    return lambda_recon * mse_loss + lambda_grad * grad_loss + beta * kl_div


def train(model: VAE, dataloader: DataLoader, hp: HyperParams, device: torch.device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    for epoch in range(hp.epochs):
        total_loss = 0.0
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(
                recon, target, mu, logvar, hp.lambda_recon, hp.lambda_grad, hp.beta
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{hp.epochs} -- loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), hp.model_path)
    print(f"Model saved to {hp.model_path}")


def main(hp: HyperParams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(hp.data_path):
        raise FileNotFoundError(f"Data not found: {hp.data_path}")
    data = torch.load(hp.data_path)
    dataset = PlasmaDataset(data, hp.seq_len)
    dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)

    model = VAE(hp.latent_dim).to(device)
    train(model, dataloader, hp, device)


if __name__ == "__main__":
    hyperparams = HyperParams()
    main(hyperparams)
