from enum import Enum

class Player(Enum):
    """Énumération pour représenter les joueurs"""
    EMPTY = 0
    PLAYER1 = 1  # Joueur 1 (connecte haut-bas)
    PLAYER2 = 2  # Joueur 2 (connecte gauche-droite)