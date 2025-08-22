
# Hex + Monte Carlo Tree Search (MCTS)

Ce projet propose une implémentation du **jeu de Hex** et de plusieurs agents d’intelligence artificielle basés sur **Monte Carlo Tree Search (MCTS)** et ses variantes.  
Il permet de simuler des parties, comparer différents agents (MCTS, RAVE, heuristiques, etc.) et analyser leurs performances sur des plateaux de tailles variées.

---

## Fonctionnalités

- Implémentation du jeu de **Hex** avec gestion du plateau et des règles.
- Plusieurs agents disponibles :
  - `RandomAgent` : joue un coup aléatoire.
  - `GreedyAgent` : privilégie des coups simples (centre, proximité de ses pierres).
  - `MCTSAgent` : MCTS classique avec UCT.
  - `MCTSRAVEAgent` : variante MCTS avec Rapid Action Value Estimation.
  - `MCTSHeuristicAgent` : simulations biaisées avec heuristique (proximité du centre).
  - `MCTSProgressiveBiasAgent` : sélection avec biais progressif.
- Comparaison d’agents via des **tournois round-robin** automatisés.
- Paramétrage du temps de réflexion par coup (`max_time`).

---
##Exemple d’utilisation

Exécuter une partie de démonstration entre MCTS et Random sur un plateau $5\times5$ :

from hex.mcts_agent import MCTSAgent
from hex.random_agent import RandomAgent
from hex.game import HexGame

if __name__ == "__main__":
    game = HexGame(5)
    mcts_agent = MCTSAgent(max_time=0.5)
    random_agent = RandomAgent()
    winner = game.play_game(mcts_agent, random_agent, display=True)
    print("Winner:", winner)
