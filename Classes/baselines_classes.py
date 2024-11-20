from typing import Literal
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from abc import ABC, abstractmethod
import polars as pl
from sklearn.mixture import GaussianMixture

class DataGenerator(ABC):
    @abstractmethod
    def generate_batch(self, batch_size) -> NDArray[np.float64]:
        """
        Generate a batch of data from the dataset.

        Args:
            batch_size (int): The size of the batch to generate.

        Returns:
            A batch of data from the dataset.
        """


class MultivariateNormalGenerator(DataGenerator):
    def __init__(self, x):
        self.mu = np.mean(x, axis=0)
        if len(x) == 1:
            # With the edge case of a single transaction,
            # we create a second transaction with the same values
            # to compute the covariance matrix
            x = np.stack([x, x])
        self.sigma = np.cov(x, rowvar=False)

    def generate_batch(self, batch_size):
        return np.random.multivariate_normal(self.mu, self.sigma, batch_size)


class UnivariateNormalGenerator(DataGenerator):
    def __init__(self, x):
        self.shape = x.shape
        self.mu = np.mean(x, axis=0)
        self.sigma = np.std(x, axis=0)

    def generate_batch(self, batch_size):
        to_return = np.random.normal(self.mu, self.sigma, (batch_size, self.shape[1]))
        return to_return


class UniformGenerator(DataGenerator):
    def __init__(self, x, quantile=0.01):
        self.shape = x.shape
        lower_quantile = quantile
        upper_quantile = 1 - quantile
        self.min_vals = np.quantile(x, lower_quantile, axis=0)
        self.max_vals = np.quantile(x, upper_quantile, axis=0)

    def generate_batch(self, batch_size):
        to_return = np.random.uniform(self.min_vals, self.max_vals, (batch_size, self.shape[1]))
        return to_return


class GaussianMixtureGenerator(DataGenerator):
    def __init__(self, x):
        self.shape = x.shape
        self.generator = GaussianMixture(n_components=10)
        self.generator.fit(x)

    def generate_batch(self, batch_size):
        to_return = self.generator.sample(batch_size)[0]
        return to_return


class BaselineAgent:
    def __init__(
        self,
        train_X: pl.DataFrame,
        generation_method: Literal["uniform", "multivariate", "univariate", "mixture", "gan"],
        quantile=0.01,
    ):
        self.train_X = train_X
        print(generation_method, train_X.shape)
        match generation_method:
            case "uniform":
                self.generator = UniformGenerator(train_X.to_numpy(), quantile=quantile)
            case "multivariate":
                self.generator = MultivariateNormalGenerator(train_X.to_numpy())
            case "univariate":
                self.generator = UnivariateNormalGenerator(train_X.to_numpy())
            case "mixture":
                self.generator = GaussianMixtureGenerator(train_X.to_numpy())
            case _:
                raise ValueError("Invalid generation method")

    def select_actions_batch(self, batch_size: int) -> np.ndarray:
        to_return = self.generator.generate_batch(batch_size)
        to_return = pl.DataFrame(pd.DataFrame(to_return, columns=self.train_X.columns))
        return to_return

    """
    def save(self, checkpoint_path):
        return

    def load(self, checkpoint_path):
        return
    """


"""
# GAN-based DataGenerator
class GANDataGenerator(DataGenerator):
    def __init__(self, x: np.ndarray, num_epochs=1000, batch_size=1024, z_dim=100, learning_rate=2e-6):
        super().__init__()
        self.x = x
        self.z_dim = z_dim

        # Get the dimensionality of the data
        self.data_dim = x.shape[1]

        # Create the generator and discriminator
        self.device = util.get_device()
        # self.device = torch.device("cuda:1")
        self.generator = Generator(z_dim, self.data_dim).to(self.device)
        self.discriminator = Discriminator(self.data_dim).to(self.device)

        # Loss function
        self.criterion = nn.BCELoss().to(self.device)

        # Optimizers
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        # Train the GAN
        self.train_gan(num_epochs, batch_size)

    def train_gan(self, num_epochs: int, batch_size: int):
        epoch_start = time()
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        x = torch.tensor(self.x, dtype=torch.float32)
        n_batches = max(1, len(self.x) // batch_size)
        for epoch in range(num_epochs):
            for _ in range(n_batches):
                # Train discriminator
                indices = torch.randint(0, len(self.x), (batch_size,))
                real_data = x[indices].to(self.device)
                fake_data = self.generator.forward(torch.randn(batch_size, self.z_dim, device=self.device))

                self.optimizer_d.zero_grad()
                outputs = self.discriminator.forward(real_data)
                loss_real = self.criterion.forward(outputs, real_labels)

                outputs = self.discriminator(fake_data.detach())
                loss_fake = self.criterion(outputs, fake_labels)

                loss_d = loss_real + loss_fake
                loss_d.backward()
                self.optimizer_d.step()

                # Train generator
                self.optimizer_g.zero_grad()
                fake_data = self.generator(torch.randn(batch_size, self.z_dim, device=self.device))
                outputs = self.discriminator(fake_data)
                loss_g = self.criterion.forward(outputs, real_labels)
                loss_g.backward()
                self.optimizer_g.step()

            if epoch % 100 == 0:
                end = time()
                duration = end - epoch_start
                epoch_start = end
                logging.info(
                    f"Epoch [{epoch}/{num_epochs}]  Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f} Duration: {duration:.2f}s"
                )

    def generate_batch(self, batch_size):
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        generated_data = self.generator.forward(z)
        return generated_data.detach().numpy(force=True)
"""

"""
# Define the generator model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x) -> torch.Tensor:
        return self.model(x)


# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

"""
