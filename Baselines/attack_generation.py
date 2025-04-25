import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from imblearn.ensemble import BalancedRandomForestClassifier

def NMSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2) / torch.std(y_true) ** 2

class VAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(4, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)       # mean
        self.fc_logvar = nn.Linear(hidden_dim,latent_dim)   # log-variance

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.ReLU()
        )

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


class Attack_Generation:
    def __init__(self, device, criterion, latent_dim, hidden_dim, lr, trees, supervised = False, y=None):
        self.device = device
        self.model = VAE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(self.device)
        self.model.to(self.device)
        self.criterion = criterion
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.supervised = supervised
        self.y = y
        if supervised:
            self.detector = BalancedRandomForestClassifier(n_estimators=trees,
                                                           random_state=42, sampling_strategy=0.15)
        else:
            self.detector = IsolationForest(n_estimators=trees)

    def train(self, real_data, batch_size=32, num_epochs=1000):
        if self.supervised:
            self.detector.fit(real_data.cpu().numpy(), self.y)
        else:
            self.detector.fit(real_data.cpu().numpy())
        self.train_vae(real_data, batch_size, num_epochs)

    def vae_loss(self, recon_x, x, mu, logvar, epoch, beta=0.001):
        #recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        recon_loss = NMSE(x, recon_x)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, KL: {kl_div.item():.4f}, Recon Loss: {recon_loss.item():.4f}")

        return recon_loss  + beta * kl_div

    def train_vae(self, data,  batch_size, num_epochs=1000):
        data = data.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Sample a random batch
            batch_indices = torch.randint(0, len(data), (batch_size,)).to(self.device)
            inputs = data[batch_indices]

            reconstructed, mu, logvar = self.model(inputs)
            loss = self.vae_loss(reconstructed, inputs, mu, logvar, epoch)

            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    def generate_valid_samples(self, num_samples):
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            samples = self.model.decoder(z).cpu().numpy()
        if self.supervised:
            undetected = self.detector.predict(samples) == 0
        else:
            undetected = self.detector.predict(samples) == 1
        valid_samples = samples[undetected]
        return valid_samples
