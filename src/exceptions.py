from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from banksys import Transaction


class AttackPeriodExpired(Exception):
    """Exception raised when an attack is finished."""


class InsufficientFundsError(Exception):
    """Exception raised when a card has insufficient funds for an action."""

    def __init__(self, trx: "Transaction"):
        super().__init__(f"Card {trx.card_id} has insufficient funds for the action of amount {trx.amount}.\n{trx}")
        self.trx = trx
