import numpy as np
import random
import math
import time
from enum import Enum
from typing import List, Tuple, Optional, Dict
import copy

class Player(Enum):
    """Énumération pour représenter les joueurs"""
    EMPTY = 0
    PLAYER1 = 1  # Joueur 1 (connecte haut-bas)
    PLAYER2 = 2  # Joueur 2 (connecte gauche-droite)

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
        
        print(f"MCTS: {iterations} itérations en {self.max_time:.2f}s")
        
        if not root.children:
            # Si aucun enfant n'a été créé, jouer aléatoirement
            valid_moves = board.get_valid_moves()
            return random.choice(valid_moves)
        
        # Retourner le coup du nœud le plus visité
        best_child = root.most_visited_child()
        print(f"Coup choisi: {best_child.move}, visites: {best_child.visits}, taux de victoire: {best_child.wins/best_child.visits:.3f}")
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


class RandomAgent:
    """Agent qui joue aléatoirement"""
    
    def get_move(self, board: HexBoard, player: Player) -> Tuple[int, int]:
        """Retourne un coup aléatoire"""
        valid_moves = board.get_valid_moves()
        return random.choice(valid_moves)

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


def compare_agents(num_games: int = 10, board_size: int = 7):
    """Compare les performances des agents"""
    print(f"\nComparaison sur {num_games} parties (plateau {board_size}x{board_size})")
    print("="*60)
    
    # MCTS vs Random
    mcts_agent = MCTSAgent(max_time=1.0)  # 1 seconde par coup
    random_agent = RandomAgent()
    
    mcts_wins = 0
    random_wins = 0
    
    for i in range(num_games):
        print(f"\nPartie {i+1}/{num_games}")
        
        # Alterner qui commence
        if i % 2 == 0:
            game = HexGame(board_size)
            print(f"La partie commence avec {game.current_player}")
            winner = game.play_game(mcts_agent, random_agent, display=False)
            if winner == Player.PLAYER1:
                mcts_wins += 1
            elif winner == Player.PLAYER2:
                random_wins += 1
        else:
            game = HexGame(board_size)
            print(f"La partie commence avec {game.current_player}")
            winner = game.play_game(random_agent, mcts_agent, display=False)
            if winner == Player.PLAYER1:
                random_wins += 1
            elif winner == Player.PLAYER2:
                mcts_wins += 1
        
        print(f"Gagnant: {'MCTS' if winner == (Player.PLAYER1 if i % 2 == 0 else Player.PLAYER2) else 'Random' if winner else 'Nul'}")
    
    print(f"\n{'='*60}")
    print(f"RÉSULTATS FINAUX:")
    print(f"MCTS: {mcts_wins} victoires ({mcts_wins/num_games*100:.1f}%)")
    print(f"Random: {random_wins} victoires ({random_wins/num_games*100:.1f}%)")
    print(f"Match nul: {num_games - mcts_wins - random_wins}")

import matplotlib.pyplot as plt

def plot_winrate_vs_games(num_games=50, board_size=5, max_time=0.5):
    """Trace le taux de victoire cumulatif de MCTS contre Random"""
    mcts_agent = MCTSAgent(max_time=max_time)
    random_agent = RandomAgent()
    
    mcts_wins = []
    random_wins = []
    draws = []
    results = []
    
    for i in range(num_games):
        game = HexGame(board_size)
        if i % 2 == 0:
            winner = game.play_game(mcts_agent, random_agent, display=False)
            results.append(1 if winner == Player.PLAYER1 else 0 if winner == Player.PLAYER2 else 0.5)
        else:
            winner = game.play_game(random_agent, mcts_agent, display=False)
            results.append(1 if winner == Player.PLAYER2 else 0 if winner == Player.PLAYER1 else 0.5)
        
        mcts_wins.append(sum(1 for r in results if r == 1) / len(results) * 100)
        random_wins.append(sum(1 for r in results if r == 0) / len(results) * 100)
        draws.append(sum(1 for r in results if r == 0.5) / len(results) * 100)
    
    plt.plot(range(1, num_games+1), mcts_wins, label="MCTS (%)")
    plt.plot(range(1, num_games+1), random_wins, label="Random (%)")
    plt.plot(range(1, num_games+1), draws, label="Match nul (%)")
    plt.xlabel("Nombre de parties")
    plt.ylabel("Taux de victoire (%)")
    plt.title(f"Taux de victoire cumulatif sur {num_games} parties (plateau {board_size}x{board_size})")
    plt.legend()
    plt.show()


def plot_winrate_vs_time(times=[0.1, 0.2, 0.5, 1.0, 2.0], num_games=20, board_size=5):
    """Trace le taux de victoire de MCTS en fonction du temps de réflexion par coup"""
    winrates = []
    for t in times:
        mcts_agent = MCTSAgent(max_time=t)
        random_agent = RandomAgent()
        wins = 0
        for i in range(num_games):
            game = HexGame(board_size)
            if i % 2 == 0:
                winner = game.play_game(mcts_agent, random_agent, display=False)
                if winner == Player.PLAYER1:
                    wins += 1
            else:
                winner = game.play_game(random_agent, mcts_agent, display=False)
                if winner == Player.PLAYER2:
                    wins += 1
        winrates.append(wins / num_games * 100)
    
    plt.plot(times, winrates, marker="o")
    plt.xlabel("Temps max par coup (s)")
    plt.ylabel("Taux de victoire (%)")
    plt.title(f"Impact du temps de calcul MCTS sur {num_games} parties (plateau {board_size}x{board_size})")
    plt.show()


def plot_simulations_per_move(board_size=5, max_time=0.5):
    """Trace le nombre de simulations effectuées par MCTS à chaque coup dans une partie"""
    class TrackingMCTSAgent(MCTSAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.sim_counts = []
        def get_move(self, board, player):
            root = MCTSNode(board, player)
            start_time = time.time()
            iterations = 0
            while time.time() - start_time < self.max_time:
                node = self._select(root)
                if not node.is_terminal() and not node.is_fully_expanded():
                    node = self._expand(node)
                result = self._simulate(node)
                self._backpropagate(node, result)
                iterations += 1
            self.sim_counts.append(iterations)
            if not root.children:
                return random.choice(board.get_valid_moves())
            return root.most_visited_child().move

    mcts_agent = TrackingMCTSAgent(max_time=max_time)
    random_agent = RandomAgent()
    game = HexGame(board_size)
    game.play_game(mcts_agent, random_agent, display=False)

    plt.plot(range(1, len(mcts_agent.sim_counts)+1), mcts_agent.sim_counts, marker="o")
    plt.xlabel("Numéro du coup (MCTS uniquement)")
    plt.ylabel("Nombre de simulations effectuées")
    plt.title(f"Évolution des simulations MCTS par coup (plateau {board_size}x{board_size})")
    plt.show()


# Exemple d'utilisation
if __name__ == "__main__":
    # Partie de démonstration
    game = HexGame(5)
    mcts_agent = MCTSAgent(max_time=0.5)
    random_agent = RandomAgent()
    winner = game.play_game(mcts_agent, random_agent, display=True)

    # Comparaison des performances
    compare_agents(num_games=5, board_size=5)

    # === Graphiques pour le rapport ===
    plot_winrate_vs_games(num_games=30, board_size=5, max_time=0.5)
    plot_winrate_vs_time(times=[0.1, 0.2, 0.5, 1.0], num_games=10, board_size=5)
    plot_simulations_per_move(board_size=5, max_time=0.5)
