import random
from hex.player import Player

class GreedyAgent:
    """Agent qui choisit un coup simple mais pas totalement aléatoire"""
    
    def get_move(self, board, player: Player):
        valid_moves = board.get_valid_moves()

        # 1. Prendre le centre si dispo
        center = (board.size // 2, board.size // 2)
        if center in valid_moves:
            return center

        # 2. Essayer de jouer à côté d’un de ses pions déjà placés
        for row, col in valid_moves:
            for nr, nc in board.get_neighbors(row, col):
                if board.board[nr][nc] == player.value:
                    return (row, col)

        # 3. Sinon coup aléatoire
        return random.choice(valid_moves)
