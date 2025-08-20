from hex.hex_board import HexBoard
from hex.player import Player
import random
import time

class HexGame:
    """Classe principale pour gérer une partie de Hex"""
    
    def __init__(self, size: int = 7, random_start: bool = True):
        self.board = HexBoard(size)
        if random_start:
            self.current_player = random.choice([Player.PLAYER1, Player.PLAYER2])
        else:
            self.current_player = Player.PLAYER1
    
    def play_game(self, agent1, agent2, display: bool = True):
        """Joue une partie complète"""
        agents = {Player.PLAYER1: agent1, Player.PLAYER2: agent2}

        
        if display:
            print(f"Nouvelle partie de Hex {self.board.size}x{self.board.size}")
            print("Joueur 1 (X) doit connecter haut-bas")
            print("Joueur 2 (O) doit connecter gauche-droite")
            print(f"La partie commence avec Joueur {self.current_player.value}")
            self.board.display()
        
        while not self.board.is_game_over():
            agent = agents[self.current_player]
            
            start_time = time.time()
            move = agent.get_move(self.board, self.current_player)
            decision_time = time.time() - start_time
            
            if self.board.make_move(move[0], move[1], self.current_player):
                if display:
                    print(f"\nJoueur {self.current_player.value} joue {move} (temps: {decision_time:.3f}s)")
                    self.board.display()
                
                # Changer de joueur
                self.current_player = Player.PLAYER2 if self.current_player == Player.PLAYER1 else Player.PLAYER1
            else:
                print(f"Coup invalide: {move}")
                break
        
        winner = self.board.get_winner()
        if display:
            if winner:
                print(f"\nJoueur {winner.value} gagne!")
            else:
                print("\nMatch nul!")
        
        return winner