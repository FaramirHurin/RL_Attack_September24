from parameters import Parameters
import dotenv
from datetime import datetime
import os
from banksys import Transaction
import logging

if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    import polars as pl

    df = pl.read_csv("/home/yann/projects/python/RL_Attack_September24/cache/cardsim/payees-10000.csv")
    print(df[0].to_dicts()[0])
    exit()
    p = Parameters()
    b = p.create_banksys()
    trx = Transaction(
        amount=100.0,
        timestamp=datetime(2023, 3, 2, 10),
        terminal_id=25,
        card_id=17,
        is_online=False,
        is_fraud=True,
    )
    b.process_transaction(trx)
