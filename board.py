
from hex.player import Player
from typing import Tuple, Optional, List
import numpy as np

class HexBoard:
    """Classe représentant le plateau de jeu Hex"""
    
    def __init__(self, size: int = 7):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        
    def copy(self):
        """Crée une copie du plateau"""
        new_board = HexBoard(self.size)
        new_board.board = self.board.copy()
        return new_board
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """Vérifie si un coup est valide"""
        return (0 <= row < self.size and 
                0 <= col < self.size and 
                self.board[row][col] == Player.EMPTY.value)
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Retourne la liste des coups valides"""
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == Player.EMPTY.value:
                    moves.append((row, col))
        return moves
    
    def make_move(self, row: int, col: int, player: Player) -> bool:
        """Effectue un coup sur le plateau"""
        if self.is_valid_move(row, col):
            self.board[row][col] = player.value
            return True
        return False
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Retourne les voisins d'une case dans le graphe hexagonal"""
        neighbors = []
        # Les 6 directions possibles dans un graphe hexagonal
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                neighbors.append((new_row, new_col))
        return neighbors
    
    def is_connected_path(self, player: Player) -> bool:
        """Vérifie si un joueur a créé un chemin gagnant"""
        if player == Player.PLAYER1:
            # Player 1 doit connecter haut et bas
            return self._has_path_top_bottom(player)
        else:
            # Player 2 doit connecter gauche et droite
            return self._has_path_left_right(player)
    
    def _has_path_top_bottom(self, player: Player) -> bool:
        """Vérifie s'il existe un chemin du haut vers le bas"""
        visited = set()
        
        # Commencer par toutes les cases du haut appartenant au joueur
        for col in range(self.size):
            if self.board[0][col] == player.value:
                if self._dfs_to_bottom(0, col, player, visited):
                    return True
        return False
    
    def _has_path_left_right(self, player: Player) -> bool:
        """Vérifie s'il existe un chemin de gauche à droite"""
        visited = set()
        
        # Commencer par toutes les cases de gauche appartenant au joueur
        for row in range(self.size):
            if self.board[row][0] == player.value:
                if self._dfs_to_right(row, 0, player, visited):
                    return True
        return False
    
    def _dfs_to_bottom(self, row: int, col: int, player: Player, visited: set) -> bool:
        """DFS pour trouver un chemin vers le bas"""
        if row == self.size - 1:  # Atteint le bas
            return True
        
        visited.add((row, col))
        
        for nr, nc in self.get_neighbors(row, col):
            if (nr, nc) not in visited and self.board[nr][nc] == player.value:
                if self._dfs_to_bottom(nr, nc, player, visited):
                    return True
        return False
    
    def _dfs_to_right(self, row: int, col: int, player: Player, visited: set) -> bool:
        """DFS pour trouver un chemin vers la droite"""
        if col == self.size - 1:  # Atteint la droite
            return True
        
        visited.add((row, col))
        
        for nr, nc in self.get_neighbors(row, col):
            if (nr, nc) not in visited and self.board[nr][nc] == player.value:
                if self._dfs_to_right(nr, nc, player, visited):
                    return True
        return False
    
    def get_winner(self) -> Optional[Player]:
        """Retourne le gagnant s'il y en a un"""
        if self.is_connected_path(Player.PLAYER1):
            return Player.PLAYER1
        elif self.is_connected_path(Player.PLAYER2):
            return Player.PLAYER2
        return None
    
    def is_game_over(self) -> bool:
        """Vérifie si la partie est terminée"""
        return self.get_winner() is not None or len(self.get_valid_moves()) == 0
    
    def display(self):
        """Affiche le plateau"""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print("\n  " + " ".join([str(i) for i in range(self.size)]))
        
        for i in range(self.size):
            # Indentation pour créer l'effet hexagonal
            print(" " * i + str(i) + " ", end="")
            for j in range(self.size):
                print(symbols[self.board[i][j]], end=" ")
            print()
