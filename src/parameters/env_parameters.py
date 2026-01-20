from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class EnvParameters:
    n_episodes: int = 6000
    terminal_fract: float = 0.1
    pool_size: int = 50
    include_weekday: bool = True
    _avg_block_delay: timedelta | float = timedelta(days=7)
    customer_location_is_known: bool = True
    can_choose_debit_credit: bool = False

    @property
    def avg_block_delay(self) -> timedelta:
        if isinstance(self._avg_block_delay, (float, int)):
            return timedelta(seconds=self._avg_block_delay)
        return self._avg_block_delay
