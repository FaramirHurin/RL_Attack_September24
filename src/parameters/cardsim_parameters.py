from dataclasses import dataclass
import orjson
import polars as pl
from utils import serialize_unknown
import hashlib
import os


@dataclass(eq=True, frozen=True)
class CardSimParameters:
    n_days: int = 365
    start_date: str = "2023-01-01"
    n_payers: int = 10_000
    with_modification: bool = False
    ulb_data: bool = False

    def load_simulation_data(self, cache_dir: str):
        from cardsim import Cardsim

        if self.ulb_data:
            try:
                transactions = pl.read_csv("src/mlgsim/transactions.csv")
                cards = pl.read_csv("src/mlgsim/customer_profiles.csv")
                terminals = pl.read_csv("src/mlgsim/terminal_profiles.csv")
            except FileNotFoundError:
                from mlgsim.generare_dataset import main as generate_dataset
                import subprocess

                generate_dataset()
                # with open(FILENAME) as f:
                #     nb = nbformat.read(f, nbformat.NO_CONVERT)
                # ep = ExecutePreprocessor(timeout=600)
                # output = ep.preprocess(nb)
                # --- Run the Python script to create the dataset---
                # subprocess.run(["python", "src/mlgsim/generate_datasets.py"], check=True)
                # --- Run the Jupyter notebook ---
                subprocess.run(
                    ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "src/mlgsim/prepare_datasets.ipynb"], check=True
                )
                transactions = pl.read_csv("src/mlgsim/transactions.csv")
                cards = pl.read_csv("src/mlgsim/customer_profiles.csv")
                terminals = pl.read_csv("src/mlgsim/terminal_profiles.csv")

            transactions = transactions.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S").alias("timestamp")
            )
            return transactions, cards, terminals

        return Cardsim().load(
            n_days=self.n_days,
            n_payers=self.n_payers,
            start_date=self.start_date,
            cache_dir=self.cache_dir(cache_dir),
            with_modification=self.with_modification,
        )

    def cache_dir(self, cache_dir: str):
        serialized = orjson.dumps(self, default=serialize_unknown)
        hash_digest = hashlib.sha256(serialized).hexdigest()
        return os.path.join(cache_dir, hash_digest)

    @staticmethod
    def paper_params(with_modification: bool, ulb_data: bool = False):
        """
        - n_days: 365 * 2 + 150 + 30
        - n_payers: 20_000
        - start_date: "2023-01-01"
        """
        return CardSimParameters(
            n_days=365 * 2 + 150 + 30,  # 2 years budget + 150 days training + 30 days warmup
            n_payers=20_000,
            start_date="2023-01-01",
            with_modification=with_modification,
            ulb_data=ulb_data,
        )

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "simulation_params.json"), "wb") as f:
            f.write(orjson.dumps(self, default=serialize_unknown, option=orjson.OPT_INDENT_2))
