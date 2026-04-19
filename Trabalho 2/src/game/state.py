from __future__ import annotations

from dataclasses import dataclass, field

from src.game.board import Board
from src.game.config import GameConfig
from src.game.move import Move
from src.utils.constants import BLACK, Player


@dataclass(slots=True)
class GameState:
    """Complete snapshot of the game at a specific turn."""

    board: Board
    current_player: Player
    consecutive_passes: int
    config: GameConfig
    history: list[Move] = field(default_factory=list)

    @classmethod
    def initial(cls, config: GameConfig | None = None) -> "GameState":
        resolved = config or GameConfig()
        return cls(
            board=Board.create_initial(resolved),
            current_player=BLACK,
            consecutive_passes=0,
            config=resolved,
            history=[],
        )

    def clone(self) -> "GameState":
        return GameState(
            board=self.board.copy(),
            current_player=self.current_player,
            consecutive_passes=self.consecutive_passes,
            config=self.config,
            history=list(self.history),
        )

    def with_updates(
        self,
        *,
        board: Board | None = None,
        current_player: Player | None = None,
        consecutive_passes: int | None = None,
        append_move: Move | None = None,
    ) -> "GameState":
        next_history = list(self.history)
        if append_move is not None:
            next_history.append(append_move)

        return GameState(
            board=board if board is not None else self.board,
            current_player=current_player if current_player is not None else self.current_player,
            consecutive_passes=(
                consecutive_passes
                if consecutive_passes is not None
                else self.consecutive_passes
            ),
            config=self.config,
            history=next_history,
        )
