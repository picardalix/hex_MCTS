from hex.hex_board import HexBoard
from hex.player import Player
import random
from typing import Tuple

class RandomAgent:
    """Agent qui joue aléatoirement"""
    
    def get_move(self, board: HexBoard, player: Player) -> Tuple[int, int]:
        """Retourne un coup aléatoire"""
        valid_moves = board.get_valid_moves()
        return random.choice(valid_moves)