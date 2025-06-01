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
