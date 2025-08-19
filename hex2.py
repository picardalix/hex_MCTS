import numpy as np
import random
import math
import time
from enum import Enum
from typing import List, Tuple, Optional, Dict
import copy
import matplotlib.pyplot as plt

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
    
    def __init__(self, board: HexBoard, player_just_moved: Player, move: Optional[Tuple[int, int]] = None, parent=None):
        self.board = board.copy()
        self.player_just_moved = player_just_moved  # joueur qui a joué pour arriver à ce nœud
        self.move = move
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
  
    def add_child(self, move: Tuple[int, int], next_player: Player) -> 'MCTSNode':
        """Ajoute un enfant avec le joueur qui vient de jouer"""
        new_board = self.board.copy()
        new_board.make_move(move[0], move[1], next_player)
        child = MCTSNode(new_board, next_player, move, self)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child
    
    def update(self, result: float):
        """Met à jour les statistiques"""
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
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            next_player = Player.PLAYER2 if node.player_just_moved == Player.PLAYER1 else Player.PLAYER1
            return node.add_child(move, next_player)
        return node
    
    def _simulate(self, node: MCTSNode) -> float:
        board_copy = node.board.copy()
        current_player = Player.PLAYER2 if node.player_just_moved == Player.PLAYER1 else Player.PLAYER1
        
        while not board_copy.is_game_over():
            move = random.choice(board_copy.get_valid_moves())
            board_copy.make_move(move[0], move[1], current_player)
            current_player = Player.PLAYER2 if current_player == Player.PLAYER1 else Player.PLAYER1
        
        winner = board_copy.get_winner()
        if winner == node.player_just_moved:
            return 1.0
        elif winner is None:
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


class GreedyAgent:
    """Agent qui essaie de bloquer l'adversaire ou de progresser vers son objectif"""
    
    def get_move(self, board: HexBoard, player: Player) -> Tuple[int, int]:
        """Stratégie simple : priorité aux coups sur les bords"""
        valid_moves = board.get_valid_moves()
        
        if player == Player.PLAYER1:
            # Joueur 1 veut connecter haut-bas : préférer les colonnes centrales
            center_col = board.size // 2
            priority_moves = [move for move in valid_moves 
                            if abs(move[1] - center_col) <= 1]
        else:
            # Joueur 2 veut connecter gauche-droite : préférer les lignes centrales  
            center_row = board.size // 2
            priority_moves = [move for move in valid_moves 
                            if abs(move[0] - center_row) <= 1]
        
        if priority_moves:
            return random.choice(priority_moves)
        return random.choice(valid_moves)


class WeakMCTSAgent:
    """Agent MCTS avec moins de temps de calcul"""
    
    def __init__(self, max_time: float = 0.1):
        self.mcts_agent = MCTSAgent(max_time=max_time)
    
    def get_move(self, board: HexBoard, player: Player) -> Tuple[int, int]:
        return self.mcts_agent.get_move(board, player)


class HexGame:
    """Classe principale pour gérer une partie de Hex"""
    
    def __init__(self, size: int = 7):
        self.board = HexBoard(size)
        self.current_player = Player.PLAYER1
    
    def play_game(self, agent1, agent2, display: bool = True):
        """Joue une partie complète"""
        agents = {Player.PLAYER1: agent1, Player.PLAYER2: agent2}
        
        if display:
            print(f"Nouvelle partie de Hex {self.board.size}x{self.board.size}")
            print("Joueur 1 (X) doit connecter haut-bas")
            print("Joueur 2 (O) doit connecter gauche-droite")
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
    """Compare les performances de différents agents"""
    print(f"\nComparaisons sur {num_games} parties (plateau {board_size}x{board_size})")
    print("="*80)
    
    agents = {
        "MCTS Fort (1.0s)": MCTSAgent(max_time=1.0),
        "MCTS Faible (0.1s)": WeakMCTSAgent(max_time=0.1), 
        "Greedy": GreedyAgent(),
        "Random": RandomAgent()
    }
    
    matchups = [
        ("MCTS Fort (1.0s)", "Random"),
        ("MCTS Fort (1.0s)", "Greedy"), 
        ("MCTS Fort (1.0s)", "MCTS Faible (0.1s)"),
        ("MCTS Faible (0.1s)", "Greedy"),
        ("MCTS Faible (0.1s)", "Random"),
        ("Greedy", "Random")
    ]
    
    for agent1_name, agent2_name in matchups:
        print(f"\n{agent1_name} vs {agent2_name}")
        print("-" * 50)
        
        agent1 = agents[agent1_name]
        agent2 = agents[agent2_name]
        
        wins1 = wins2 = draws = 0
        
        for i in range(num_games):
            # Alterner qui commence
            if i % 2 == 0:
                game = HexGame(board_size)
                winner = game.play_game(agent1, agent2, display=False)
                if winner == Player.PLAYER1:
                    wins1 += 1
                elif winner == Player.PLAYER2:
                    wins2 += 1
                else:
                    draws += 1
            else:
                game = HexGame(board_size)
                winner = game.play_game(agent2, agent1, display=False)
                if winner == Player.PLAYER1:
                    wins2 += 1
                elif winner == Player.PLAYER2:
                    wins1 += 1
                else:
                    draws += 1
        
        print(f"{agent1_name}: {wins1} victoires ({wins1/num_games*100:.1f}%)")
        print(f"{agent2_name}: {wins2} victoires ({wins2/num_games*100:.1f}%)")
        if draws > 0:
            print(f"Matchs nuls: {draws}")


def detailed_analysis():
    """Analyse détaillée d'une partie MCTS vs Greedy"""
    print("\n" + "="*60)
    print("ANALYSE DÉTAILLÉE: MCTS vs Greedy")
    print("="*60)
    
    game = HexGame(5)
    mcts_agent = MCTSAgent(max_time=0.5)
    greedy_agent = GreedyAgent()
    
    # Jouer une partie avec affichage détaillé
    winner = game.play_game(mcts_agent, greedy_agent, display=True)
    
    if winner:
        winner_name = "MCTS" if winner == Player.PLAYER1 else "Greedy"
        print(f"\n🏆 Victoire de {winner_name}!")
    
    # Analyser quelques positions clés
    print(f"\n📊 OBSERVATIONS:")
    print(f"• Le plateau final montre les stratégies adoptées")
    print(f"• MCTS adapte sa stratégie selon l'évolution du jeu") 
    print(f"• Greedy suit une heuristique simple mais cohérente")



def run_experiments(num_games: int = 50, board_size: int = 5):
    """Lance une série de parties et retourne les résultats"""
    mcts_agent = MCTSAgent(max_time=0.5)
    greedy_agent = GreedyAgent()
    
    mcts_wins, greedy_wins = 0, 0
    results = []  # stocke gagnant par partie (1=MCTS, 0=Greedy)

    for i in range(num_games):
        game = HexGame(board_size)
        # alterner qui commence
        if i % 2 == 0:
            winner = game.play_game(mcts_agent, greedy_agent, display=False)
            if winner == Player.PLAYER1:
                mcts_wins += 1
                results.append(1)
            else:
                greedy_wins += 1
                results.append(0)
        else:
            winner = game.play_game(greedy_agent, mcts_agent, display=False)
            if winner == Player.PLAYER1:
                greedy_wins += 1
                results.append(0)
            else:
                mcts_wins += 1
                results.append(1)

    print(f"\nRésultats sur {num_games} parties (plateau {board_size}x{board_size})")
    print(f"MCTS: {mcts_wins}/{num_games} ({mcts_wins/num_games*100:.1f}%)")
    print(f"Greedy: {greedy_wins}/{num_games} ({greedy_wins/num_games*100:.1f}%)")

    return results

def plot_results(results: List[int], title="Résultats MCTS vs Greedy"):
    """Trace l'évolution du taux de victoire cumulatif"""
    cumulative_winrate = [sum(results[:i+1])/(i+1) for i in range(len(results))]

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(results)+1), cumulative_winrate, marker="o")
    plt.axhline(0.5, color="red", linestyle="--", label="50%")
    plt.xlabel("Nombre de parties")
    plt.ylabel("Taux de victoire cumulatif (MCTS)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":

    # Lancer une série d’expériences
    results = run_experiments(num_games=50, board_size=5)
    # Tracer les résultats
    plot_results(results, title="MCTS (0.5s) vs Greedy - Plateau 5x5")
    # Analyse détaillée d'une partie
    # detailed_analysis()
    
    # # Comparaisons complètes
    # compare_agents(num_games=8, board_size=5)
    
    # # Test sur différentes tailles de plateau
    # print(f"\n{'='*80}")
    # print("IMPACT DE LA TAILLE DU PLATEAU")
    # print("="*80)
    
    # for size in [4, 5, 6]:
    #     print(f"\nPlateau {size}x{size} - MCTS Fort vs Greedy (6 parties)")
        
    #     mcts_wins = greedy_wins = 0
    #     mcts_agent = MCTSAgent(max_time=0.5)
    #     greedy_agent = GreedyAgent()
        
    #     for i in range(6):
    #         if i % 2 == 0:
    #             game = HexGame(size)
    #             winner = game.play_game(mcts_agent, greedy_agent, display=False)
    #             if winner == Player.PLAYER1:
    #                 mcts_wins += 1
    #             else:
    #                 greedy_wins += 1
    #         else:
    #             game = HexGame(size)
    #             winner = game.play_game(greedy_agent, mcts_agent, display=False)
    #             if winner == Player.PLAYER1:
    #                 greedy_wins += 1
    #             else:
    #                 mcts_wins += 1
        
    #     print(f"MCTS: {mcts_wins}/6 ({mcts_wins/6*100:.1f}%) | Greedy: {greedy_wins}/6 ({greedy_wins/6*100:.1f}%)")