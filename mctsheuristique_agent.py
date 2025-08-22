from hex.player import Player
import math, random, time
from typing import Tuple, Optional, List, Dict
from hex.mcts_agent import MCTSAgent

# ===============================
# 2. MCTS avec playouts heuristiques
# ===============================
class MCTSHeuristicAgent(MCTSAgent):
    def _simulate(self, node) -> float:
        board_copy = node.board.copy()
        current_player = Player.PLAYER2 if node.player == Player.PLAYER1 else Player.PLAYER1

        while not board_copy.is_game_over():
            moves = board_copy.get_valid_moves()
            if not moves:
                break
            # Heuristique simple : privilégier centre
            moves.sort(key=lambda m: abs(m[0]-board_copy.size//2) + abs(m[1]-board_copy.size//2))
            # 80% heuristique, 20% aléatoire
            if random.random() < 0.8:
                move = moves[0]
            else:
                move = random.choice(moves)

            board_copy.make_move(move[0], move[1], current_player)
            current_player = Player.PLAYER2 if current_player == Player.PLAYER1 else Player.PLAYER1

        winner = board_copy.get_winner()
        if winner == node.player:
            return 1.0
        elif winner is None:
            return 0.5
        else:
            return 0.0
