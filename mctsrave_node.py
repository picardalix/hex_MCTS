
from hex.game import HexBoard
from hex.player import Player
import math, random, time
from typing import Tuple, Optional, List, Dict

class MCTSRAVENode:
    def __init__(self, board: HexBoard, player: Player, move: Optional[Tuple[int, int]] = None, parent=None):
        self.board = board.copy()
        self.player = player
        self.move = move
        self.parent = parent
        self.children: List['MCTSRAVENode'] = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = board.get_valid_moves()
        self.rave_stats: Dict[Tuple[int, int], Tuple[int, int]] = {}  # move -> (wins, visits)

    def is_terminal(self):
        return self.board.is_game_over()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def expand(self):
        move = random.choice(self.untried_moves)
        next_player = Player.PLAYER2 if self.player == Player.PLAYER1 else Player.PLAYER1
        new_board = self.board.copy()
        new_board.make_move(move[0], move[1], next_player)
        child = MCTSRAVENode(new_board, next_player, move, self)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child

    def update(self, result: float, played_moves: List[Tuple[int, int]]):
        self.visits += 1
        self.wins += result
        # Mise à jour RAVE
        for m in played_moves:
            if m not in self.rave_stats:
                self.rave_stats[m] = (0, 0)
            w, v = self.rave_stats[m]
            self.rave_stats[m] = (w + result, v + 1)

    def uct_rave_value(self, child, exploration_constant, rave_const):
        # exploitation classique
        if child.visits == 0:
            q_value = 0
        else:
            q_value = child.wins / child.visits

        # exploitation RAVE (AMAF)
        if child.move in self.rave_stats:
            rave_w, rave_v = self.rave_stats[child.move]
            amaf_value = rave_w / rave_v if rave_v > 0 else 0
        else:
            amaf_value = 0

        beta = rave_const / (child.visits + rave_const)
        combined_value = (1 - beta) * q_value + beta * amaf_value

        # exploration
        if child.visits == 0:
            return float("inf")
        return combined_value + exploration_constant * math.sqrt(math.log(self.visits) / child.visits)

    def best_child(self, exploration_constant, rave_const):
        return max(self.children, key=lambda c: self.uct_rave_value(c, exploration_constant, rave_const))

    def most_visited_child(self):
        return max(self.children, key=lambda c: c.visits)

