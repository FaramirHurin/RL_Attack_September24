from dataclasses import dataclass
import polars as pl
import hashlib
import subprocess


@dataclass(eq=True)
class CardSimParameters:
    n_days: int = 365
    start_date: str = "2023-01-01"
    n_payers: int = 10_000

    def get_simulation_data(self, use_cache: bool = True, ulb_data=False):
        from cardsim import Cardsim

        if ulb_data:
            try:
                transactions = pl.read_csv("src/mlgsim/transactions.csv")
                cards = pl.read_csv("src/mlgsim/customer_profiles.csv")
                terminals = pl.read_csv("src/mlgsim/terminal_profiles.csv")
            except FileNotFoundError as e:
                # --- Run the Python script to create the dataset---
                subprocess.run(["python", "src/mlgsim/generate_datasets.py"], check=True)
                # --- Run the Jupyter notebook ---
                subprocess.run([
                    "jupyter", "nbconvert",
                    "--to", "notebook",
                    "--execute",
                    "--inplace",
                    "src/mlgsim/prepare_datasets.ipynb"
                ], check=True)

                transactions = pl.read_csv("src/mlgsim/transactions.csv")
                cards = pl.read_csv("src/mlgsim/customer_profiles.csv")
                terminals = pl.read_csv("src/mlgsim/terminal_profiles.csv")


            transactions = transactions.with_columns(
               pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S").alias("timestamp")
            )


        else:
            simulator = Cardsim()
            transactions, cards, terminals = simulator.simulate(
                n_days=self.n_days,
                n_payers=self.n_payers,
                start_date=self.start_date,
                use_cache=use_cache,
            )
        return transactions, cards, terminals

    @staticmethod
    def paper_params():
        """
        - n_days: 365 * 2 + 150 + 30
        - n_payers: 20_000
        - start_date: "2023-01-01"
        """
        return CardSimParameters(
            n_days=365 * 2 + 150 + 30,  # 2 years budget + 150 days training + 30 days warmup
            n_payers=20_000,
            start_date="2023-01-01",
        )

    def __hash__(self) -> int:
        h = hashlib.sha256(str((self.n_days, self.start_date, self.n_payers)).encode("utf-8")).hexdigest()
        return int(h, 16)
