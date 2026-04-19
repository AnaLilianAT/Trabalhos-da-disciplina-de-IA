from __future__ import annotations

from abc import ABC, abstractmethod

from src.game.move import Move
from src.game.state import GameState


class Agent(ABC):
    """Base class for any decision-making player."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def choose_move(self, state: GameState) -> Move:
        """Return a move for the current player in state."""
        raise NotImplementedError
