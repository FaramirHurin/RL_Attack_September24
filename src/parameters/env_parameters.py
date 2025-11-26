from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence


@dataclass
class EnvParameters:
    n_episodes: int
    know_client: bool
    terminal_fract: float
    card_pool_size: int
    include_weekday: bool
    avg_card_block_delay: timedelta
    normalize_location: bool
    customer_location_is_known: bool
    can_choose_debit_credit: bool
    scale_amount: float

    def __init__(
        self,
        n_episodes: int = 4000,
        know_client: bool = True,
        terminal_fract: float = 0.1,
        card_pool_size: int = 50,
        include_weekday: bool = True,
        avg_card_block_delay: int | timedelta = 7,
        normalize_location: bool = False,
        customer_location_is_known: bool = True,
        aggregation_windows: Sequence[timedelta | float] = (timedelta(hours=1), timedelta(days=1), timedelta(days=7), timedelta(days=30)),
        can_choose_debit_credit: bool = False,
        scale_amount: float = 1.0,
    ):
        self.n_episodes = n_episodes
        self.know_client = know_client
        self.terminal_fract = terminal_fract
        self.card_pool_size = card_pool_size
        self.include_weekday = include_weekday
        self.scale_amount = scale_amount
        self.customer_location_is_known = customer_location_is_known
        if isinstance(avg_card_block_delay, int):
            avg_card_block_delay = timedelta(seconds=avg_card_block_delay)
        self.avg_card_block_delay = avg_card_block_delay
        self.normalize_location = normalize_location
        self.can_choose_debit_credit = can_choose_debit_credit
        self.aggregation_windows = []
        for window in aggregation_windows:
            if isinstance(window, (float, int)):
                window = timedelta(seconds=window)
            self.aggregation_windows.append(window)
