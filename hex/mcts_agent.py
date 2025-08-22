from hex.game import HexBoard
from hex.player import Player
from hex.mcts_node import MCTSNode
import math
import random
import time
from typing import List, Tuple, Optional

class MCTSAgent:
    """Agent utilisant l'algorithme MCTS"""
    
    def __init__(self, exploration_constant: float = math.sqrt(2), max_time: float = 1.0):
        self.exploration_constant = exploration_constant
        self.max_time = max_time
    
    def get_move(self, board: HexBoard, player: Player) -> Tuple[int, int]:
        """Retourne le meilleur coup selon MCTS"""
        root = MCTSNode(board, player)
        
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < self.max_time:
            # 1. Sélection
            node = self._select(root)
            
            # 2. Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)
            
            # 3. Simulation
            result = self._simulate(node)
            
            # 4. Rétropropagation
            self._backpropagate(node, result)
            
            iterations += 1
        
        #print(f"MCTS: {iterations} itérations en {self.max_time:.2f}s")
        
        if not root.children:
            # Si aucun enfant n'a été créé, jouer aléatoirement
            valid_moves = board.get_valid_moves()
            return random.choice(valid_moves)
        
        # Retourner le coup du nœud le plus visité
        best_child = root.most_visited_child()
        #print(f"Coup choisi: {best_child.move}, visites: {best_child.visits}, taux de victoire: {best_child.wins/best_child.visits:.3f}")
        return best_child.move
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Phase de sélection: descendre dans l'arbre"""
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_constant)
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Phase d'expansion: ajouter un nouveau nœud"""
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            # Alterner les joueurs
            next_player = Player.PLAYER2 if node.player == Player.PLAYER1 else Player.PLAYER1
            return node.add_child(move, next_player)
        return node
    
    def _simulate(self, node: MCTSNode) -> float:
        """Phase de simulation: jouer aléatoirement jusqu'à la fin"""
        board_copy = node.board.copy()
        current_player = Player.PLAYER2 if node.player == Player.PLAYER1 else Player.PLAYER1
        
        while not board_copy.is_game_over():
            valid_moves = board_copy.get_valid_moves()
            if not valid_moves:
                break
                
            move = random.choice(valid_moves)
            board_copy.make_move(move[0], move[1], current_player)
            
            # Alterner les joueurs
            current_player = Player.PLAYER2 if current_player == Player.PLAYER1 else Player.PLAYER1
        
        winner = board_copy.get_winner()
        
        # Retourner 1 si le joueur du nœud a gagné, 0 sinon
        if winner == node.player:
            return 1.0
        elif winner is None:  # Match nul (très rare dans Hex)
            return 0.5
        else:
            return 0.0
    
    def _backpropagate(self, node: MCTSNode, result: float):
        """Phase de rétropropagation: remonter le résultat"""
        while node is not None:
            node.update(result)
            # Inverser le résultat pour le parent (joueur opposé)
            result = 1.0 - result
            node = node.parent
