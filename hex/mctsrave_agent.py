from hex.game import HexBoard
from hex.player import Player
import math, random, time
from typing import Tuple, Optional, List, Dict
from hex.mctsrave_node import MCTSRAVENode

# ===============================
# 1. MCTS avec RAVE (AMAF)
# ===============================
class MCTSRAVEAgent:
    def __init__(self, exploration_constant: float = math.sqrt(2), max_time: float = 1.0, rave_const: float = 300):
        self.exploration_constant = exploration_constant
        self.max_time = max_time
        self.rave_const = rave_const  # poids qui équilibre MCTS et RAVE

    def get_move(self, board: HexBoard, player: Player) -> Tuple[int, int]:
        root = MCTSRAVENode(board, player)

        start_time = time.time()
        while time.time() - start_time < self.max_time:
            node = self._select(root)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            result, played_moves = self._simulate(node)
            self._backpropagate(node, result, played_moves)

        if not root.children:
            return random.choice(board.get_valid_moves())
        return root.most_visited_child().move

    def _select(self, node):
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_constant, self.rave_const)
        return node

    def _simulate(self, node) -> Tuple[float, List[Tuple[int, int]]]:
        board_copy = node.board.copy()
        current_player = Player.PLAYER2 if node.player == Player.PLAYER1 else Player.PLAYER1
        played_moves = []

        while not board_copy.is_game_over():
            move = random.choice(board_copy.get_valid_moves())
            board_copy.make_move(move[0], move[1], current_player)
            played_moves.append(move)
            current_player = Player.PLAYER2 if current_player == Player.PLAYER1 else Player.PLAYER1

        winner = board_copy.get_winner()
        if winner == node.player:
            return 1.0, played_moves
        elif winner is None:
            return 0.5, played_moves
        else:
            return 0.0, played_moves

    def _backpropagate(self, node, result: float, played_moves: List[Tuple[int, int]]):
        while node is not None:
            node.update(result, played_moves)
            result = 1.0 - result
            node = node.parent
