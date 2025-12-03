from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence


@dataclass
class EnvParameters:
    n_episodes: int
    know_client: bool
    terminal_fract: float
    pool_size: int
    include_weekday: bool
    avg_block_delay: timedelta
    customer_location_is_known: bool
    can_choose_debit_credit: bool

    def __init__(
        self,
        n_episodes: int = 4000,
        know_client: bool = True,
        terminal_fract: float = 0.1,
        pool_size: int = 50,
        include_weekday: bool = True,
        avg_card_block_delay: int | timedelta = timedelta(days=7),
        customer_location_is_known: bool = True,
        aggregation_windows: Sequence[timedelta | float] = (timedelta(hours=1), timedelta(days=1), timedelta(days=7), timedelta(days=30)),
        can_choose_debit_credit: bool = False,
    ):
        self.n_episodes = n_episodes
        self.know_client = know_client
        self.terminal_fract = terminal_fract
        self.pool_size = pool_size
        self.include_weekday = include_weekday
        self.customer_location_is_known = customer_location_is_known
        if isinstance(avg_card_block_delay, int):
            avg_card_block_delay = timedelta(seconds=avg_card_block_delay)
        self.avg_block_delay = avg_card_block_delay
        self.can_choose_debit_credit = can_choose_debit_credit
        self.aggregation_windows = []
        for window in aggregation_windows:
            if isinstance(window, (float, int)):
                window = timedelta(seconds=window)
            self.aggregation_windows.append(window)
