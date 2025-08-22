from hex.hex_board import HexBoard
from hex.player import Player
import math
from typing import List, Tuple, Optional, Dict

class MCTSNode:
    """Nœud pour l'arbre MCTS"""
    
    def __init__(self, board: HexBoard, player: Player, move: Optional[Tuple[int, int]] = None, parent=None):
        self.board = board.copy()
        self.player = player
        self.move = move  # Le coup qui a mené à cet état
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = board.get_valid_moves()
    
    def is_fully_expanded(self) -> bool:
        """Vérifie si tous les coups possibles ont été explorés"""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Vérifie si c'est un nœud terminal"""
        return self.board.is_game_over()
    
    def add_child(self, move: Tuple[int, int], player: Player) -> 'MCTSNode':
        """Ajoute un enfant au nœud"""
        new_board = self.board.copy()
        new_board.make_move(move[0], move[1], player)
        
        child = MCTSNode(new_board, player, move, self)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child
    
    def update(self, result: float):
        """Met à jour les statistiques du nœud"""
        self.visits += 1
        self.wins += result
    
    def uct_value(self, exploration_constant: float = math.sqrt(2)) -> float:
        """Calcule la valeur UCT du nœud"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float = math.sqrt(2)) -> 'MCTSNode':
        """Sélectionne le meilleur enfant selon UCT"""
        return max(self.children, key=lambda child: child.uct_value(exploration_constant))
    
    def most_visited_child(self) -> 'MCTSNode':
        """Retourne l'enfant le plus visité"""
        return max(self.children, key=lambda child: child.visits)
