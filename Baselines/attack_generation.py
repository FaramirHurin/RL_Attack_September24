from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from banksys.banksys import Banksys
from banksys.card import Card
from banksys.terminal import Terminal
from banksys.transaction import Transaction
from agents import Agent

COLUMNS = [
    "payer_id",
    "credit_card",
    "is_online",
    "amount",
    "payee_id",
    "distance",
    "time_seconds",
    "date_time",
    "hour",
    "fraud",
    "run_id",
    "label",
]

VAE_COLUMNS = ["is_online", "amount", "payee_x", "payee_y", "hour"]
VAE_CLIENT_COLUMNS = ["is_online", "amount", "delta_x", "delta_y", "hour"]

OBSERVATION_FEATURES = ["remaining_time", "is_credit", "hour", "day"]
OBSERVATION_FEATURES_CLIENT = ["remaining_time", "is_credit", "hour", "day", "payer_x", "payer_y"]


def NMSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2) / torch.std(y_true) ** 2


def vae_loss(recon_x, x, mu, logvar, epoch, beta=0.005):
    # recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    recon_loss = NMSE(x, recon_x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, KL: {kl_div.item():.4f}, Recon Loss: {recon_loss.item():.4f}")
    return recon_loss + beta * kl_div


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # log-variance

        # Decoder
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

    def encode(self, x):
        h = torch.sigmoid(self.fc1(x))
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
        # Apply sigmoid only to one node
        # reconstructed = reconstructed.clone()
        # reconstructed = torch.sigmoid(reconstructed)
        # reconstructed[:, 0] = torch.sigmoid(reconstructed[:, 0])
        # assert min(reconstructed[:, 0]) >= 0
        # assert max(reconstructed[:, 0]) <= 1
        return reconstructed, mu, logvar


class Attack_Generation:
    def __init__(
        self,
        device,
        criterion,
        latent_dim,
        hidden_dim,
        lr,
        trees,
        training_data,
        y,
        supervised=False,
    ):
        self.device = device
        self.model = VAE(input_dim=training_data.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim).to(self.device)
        self.model.to(self.device)
        self.criterion = criterion
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.supervised = supervised
        self.training_data = training_data
        self.y = y
        if supervised:
            self.detector = BalancedRandomForestClassifier(n_estimators=trees, random_state=42, sampling_strategy=0.15)  # type: ignore
        else:
            self.detector = IsolationForest(n_estimators=trees)

    def train(self, batch_size=32, num_epochs=1000):
        if self.supervised:
            self.detector.fit(self.training_data.cpu().numpy(), self.y)
        else:
            self.detector.fit(self.training_data.cpu().numpy())  # type: ignore
        self._train_vae(self.training_data.to(torch.float), batch_size, num_epochs)

    def _train_vae(self, data, batch_size, num_epochs=1000):
        data = data.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Sample a random batch
            batch_indices = torch.randint(0, len(data), (batch_size,)).to(self.device)
            inputs = data[batch_indices]

            reconstructed, mu, logvar = self.model(inputs)

            loss = vae_loss(reconstructed, inputs, mu, logvar, epoch)

            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    def _generate_batch(self, num_samples: int = 1) -> np.ndarray:
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


class VaeAgent(Agent):
    def __init__(
        self,
        device,
        latent_dim: int,
        hidden_dim: int,
        lr: float,
        trees: int,
        banksys: Banksys,
        terminal_codes: list,
        current_time: datetime,
        batch_size=32,
        num_epochs=1000,
        know_client: bool = False,
        supervised: bool = True,
        quantile: float = 0.9,
    ):
        super().__init__()
        self.device = device
        self.banksys = banksys
        self.current_time = current_time
        self.terminals = terminal_codes
        self.know_client = know_client
        self.supervised = supervised
        self.quantile = quantile

        if self.know_client:
            self.columns = VAE_CLIENT_COLUMNS
        else:
            self.columns = VAE_COLUMNS

        # Preprocess the data
        transactions_df = self._prepare_data()
        q_low = transactions_df["amount"].quantile(0.02)
        q_hi = transactions_df["amount"].quantile(0.98)
        transactions_df = transactions_df[(transactions_df["amount"] < q_hi) & (transactions_df["amount"] > q_low)]
        labels = transactions_df["is_fraud"].values
        transactions_df = transactions_df[self.columns]

        # Normalize the data
        self.scaler = MinMaxScaler()
        self.scaler.fit(transactions_df[self.columns].values)
        normalized_values = self.scaler.transform(transactions_df[self.columns].values)

        self.attack_generator = Attack_Generation(
            device=device,
            criterion=torch.nn.MSELoss(),
            y=labels,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            training_data=torch.tensor(normalized_values).to(device),
            lr=lr,
            trees=trees,
            supervised=supervised,
        )
        self.attack_generator.train(batch_size=batch_size, num_epochs=num_epochs)

    def _prepare_data(self) -> pd.DataFrame:
        """
        Preprocess the data and return a DataFrame with the transactions
        """
        terminals: list[Terminal] = [terminal for terminal in self.banksys.terminals if terminal in self.terminals]

        transactions_df = self.get_trx_from_terminals(terminals, self.current_time)

        if self.know_client:
            customers: list[Card] = self.banksys.cards
            transactions_df = self._trx_and_customers(transactions_df, customers)
        transactions_df["hour"] = transactions_df["timestamp"].dt.hour
        return transactions_df

    def choose_action(self, observation: np.ndarray):
        """
        Choose an action based on the observation given by environment
        :param observation: the observation comprising the remaining time, is_credit, hour, day and
        if self.know_client, payer_x, payer_y
        :return: the action to be taken, comprising
         [is_online, amount, terminal_x, terminal_y, delay_hours, delay_day, payee_x, payee_y]
        """
        # Generate a batch of transactions
        batch = self.attack_generator._generate_batch(3000)
        # Turn it to the original scale and to dataframe
        batch = self.scaler.inverse_transform(batch)
        # Sort batch by second column (amount)
        batch = pd.DataFrame(batch, columns=self.columns)
        batch["is_online"] = batch["is_online"] > 0.5
        batch["amount"] = batch["amount"].round(2)
        if self.know_client:
            # TODO How to pass observation[payer_x]. Possibly -2, -1
            batch["payee_x"] = observation[-2] + batch["delta_x"]
            batch["payee_y"] = observation[-1] + batch["delta_y"]
        else:
            batch["payee_x"] = batch["payee_x"].astype(int)
            batch["payee_y"] = batch["payee_y"].astype(int)
        batch = batch.sort_values(by="amount", ascending=True)
        small_df = batch.iloc[: int(self.quantile * len(batch)), :]
        index = np.random.randint(0, len(small_df))
        trx = small_df.iloc[index, :]

        # If hour > observation.hour, same day and new hour. Otherwise, next day and new hour
        trx = trx.copy()  # Make an explicit copy to avoid SettingWithCopyWarning
        trx["delay_hours"] = int((observation[-2] + trx["hour"]) % 24)
        trx["delay_day"] = 0

        # Drop trx['hour']
        trx = trx.drop("hour")
        # Reorder Series to match the order of the Action class
        trx = trx[["is_online", "amount", "payee_x", "payee_y", "delay_day", "delay_hours"]]
        return trx.to_numpy()

    @staticmethod
    def get_trx_from_terminals(terminals: list[Terminal], current_time: datetime) -> pd.DataFrame:
        """
        Get the transactions from the terminals
        :param terminals: list of terminals
        :param current_time: current time
        :return: DataFrame with the transactions
        """
        transactions: list[Transaction] = []
        for terminal in terminals:
            transactions += terminal.transactions
            # Add terminal coordinates to the transactions
            for transaction in transactions:
                transaction.payee_x = terminal.x
                transaction.payee_y = terminal.y
        transactions = [transaction for transaction in transactions if transaction.timestamp <= current_time]
        transactions_df = pd.DataFrame([transaction.__dict__ for transaction in transactions])
        return transactions_df

    @staticmethod
    def _trx_and_customers(transactionsDF: pd.DataFrame, customers: list[Card]) -> pd.DataFrame:
        """
        Preprocess the transactions and use s.
        :param transactionsDF: DataFrame with the transactions
        :return: DataFrame with the transactions, where we add the customers coordinates
        and the delta_x and delta_y
        """
        # Create a DataFrame with the customers and their coordinates
        card_df = pd.DataFrame([card.__dict__ for card in customers])
        card_df = card_df[["id", "customer_x", "customer_y"]]
        # rename id to card_id
        card_df = card_df.rename(columns={"id": "card_id"})

        # Join transactionsDF and card_df on card_id
        transactionsDF = pd.merge(transactionsDF, card_df, on="card_id", how="left")
        transactionsDF = transactionsDF.rename(columns={"x": "customer_x", "y": "customer_y"})

        transactionsDF["delta_x"] = transactionsDF["payee_x"] - transactionsDF["customer_x"]
        transactionsDF["delta_y"] = transactionsDF["payee_y"] - transactionsDF["customer_y"]

        return transactionsDF

    def update(self, *args, **kwargs):
        return {}
