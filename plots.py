from datetime import datetime
from dataclasses import dataclass
from marlenv import Episode
from functools import cached_property
from banksys import Transaction, Card
from environment import Action


@dataclass
class LogItem:
    episode: Episode

    @cached_property
    def t_start(self):
        return datetime.fromisoformat(self.episode.metrics["t_start"])

    @cached_property
    def t_end(self):
        return datetime.fromisoformat(self.episode.metrics["t_end"])

    @cached_property
    def terminal_ids(self) -> list[int]:
        return self.episode.metrics["terminals"]

    @cached_property
    def card_id(self) -> int:
        return self.episode.metrics["card_id"]

    @cached_property
    def transactions(self):
        res = list[Transaction]()
        t = self.t_start
        for i, transition in enumerate(self.episode.transitions()):
            action = Action.from_numpy(transition.action)
            t = t + action.timedelta
            res.append(Transaction(action.amount, t, self.terminal_ids[i], self.card_id, action.is_online, False))
        res[-1].label = True
        return res
