"""
Conditional Variational Autoencoder (CVAE) for ETF return prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Encoder(nn.Module):
    def __init__(self, target_dim, cond_dim, hidden_layers, latent_dim):
        super().__init__()
        input_dim = target_dim + cond_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(prev_dim, latent_dim)
        self.logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, target, cond):
        x = torch.cat([target, cond], dim=-1)
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_layers, target_dim):
        super().__init__()
        input_dim = latent_dim + cond_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, target_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=-1)
        return self.net(x)


class ConditionalVAE(nn.Module):
    def __init__(self, target_dim, cond_dim, hidden_layers, latent_dim):
        super().__init__()
        self.encoder = Encoder(target_dim, cond_dim, hidden_layers, latent_dim)
        self.decoder = Decoder(latent_dim, cond_dim, hidden_layers, target_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, target, cond):
        mu, logvar = self.encoder(target, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, cond)
        return recon, mu, logvar

    def sample(self, cond, num_samples=1):
        """Sample from the prior and decode with given condition."""
        self.eval()
        with torch.no_grad():
            batch_size = cond.size(0)
            cond_expanded = cond.repeat(num_samples, 1)
            z = torch.randn(num_samples * batch_size, self.encoder.mu.out_features, device=cond.device)
            recon = self.decoder(z, cond_expanded)
            # Reshape to (num_samples, batch_size, target_dim)
            recon = recon.view(num_samples, batch_size, -1)
            return recon


class VAETrainer:
    def __init__(self, target_dim, cond_dim, hidden_layers=None, latent_dim=8,
                 beta=0.5, lr=0.001, seed=42):
        if hidden_layers is None:
            hidden_layers = [128, 64]
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ConditionalVAE(target_dim, cond_dim, hidden_layers, latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.beta = beta

    def fit(self, cond, target, epochs=100, batch_size=128):
        cond = torch.tensor(cond, dtype=torch.float32).to(self.device)
        target = torch.tensor(target, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(cond, target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        best_loss = float('inf')
        best_state = None

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_cond, batch_target in loader:
                batch_cond, batch_target = batch_cond.to(self.device), batch_target.to(self.device)
                recon, mu, logvar = self.model(batch_target, batch_cond)

                recon_loss = nn.functional.mse_loss(recon, batch_target)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_target.size(0)
                loss = recon_loss + self.beta * kl_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(batch_cond)

            avg_loss = total_loss / len(target)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1:3d} | Loss: {avg_loss:.6f}")

        if best_state:
            self.model.load_state_dict(best_state)

    def predict_expected_returns(self, latest_cond, tickers, num_samples=100):
        """Sample from the CVAE and return expected returns per ETF."""
        self.model.eval()
        cond_t = torch.tensor(latest_cond, dtype=torch.float32).unsqueeze(0).to(self.device)
        samples = self.model.sample(cond_t, num_samples)  # (num_samples, 1, n_etfs)
        samples = samples.squeeze(1).cpu().numpy()        # (num_samples, n_etfs)
        expected = samples.mean(axis=0)
        return {t: float(expected[i]) for i, t in enumerate(tickers)}

    def compute_regime_stress(self, cond, target, lookback=63):
        """Compute average KL divergence over recent lookback window."""
        self.model.eval()
        with torch.no_grad():
            cond_t = torch.tensor(cond[-lookback:], dtype=torch.float32).to(self.device)
            target_t = torch.tensor(target[-lookback:], dtype=torch.float32).to(self.device)
            mu, logvar = self.model.encoder(target_t, cond_t)
            # KL divergence per sample
            kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            return float(kl_per_sample.mean().item())
