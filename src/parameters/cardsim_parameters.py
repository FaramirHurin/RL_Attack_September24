from dataclasses import dataclass
import polars as pl


@dataclass(eq=True)
class CardSimParameters:
    n_days: int = 365
    start_date: str = "2023-01-01"
    n_payers: int = 10_000

    def get_simulation_data(self, use_cache: bool = True, ulb_data=False):
        from cardsim import Cardsim

        if ulb_data:
            transactions = pl.read_csv("MLG_Simulator/transactions.csv")
            cards = pl.read_csv("MLG_Simulator/customer_profiles.csv")
            terminals = pl.read_csv("MLG_Simulator/terminal_profiles.csv")
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
