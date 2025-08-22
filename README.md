
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
```python
from hex.mcts_agent import MCTSAgent
from hex.random_agent import RandomAgent
from hex.game import HexGame

if __name__ == "__main__":
    game = HexGame(5)
    mcts_agent = MCTSAgent(max_time=0.5)
    random_agent = RandomAgent()
    winner = game.play_game(mcts_agent, random_agent, display=True)
    print("Winner:", winner) ```

---
##Comparaison d’agents

Vous pouvez lancer un tournoi round-robin entre plusieurs agents :
```python
from hex.mcts_agent import MCTSAgent
from hex.random_agent import RandomAgent
from hex.greedy_agent import GreedyAgent
from hex.experience import compare_agents

agents = {
    "MCTS(0.1s)": MCTSAgent(max_time=0.1),
    "MCTS(0.5s)": MCTSAgent(max_time=0.5),
    "Greedy": GreedyAgent(),
    "Random": RandomAgent(),
}

compare_agents(agents, num_games=10, board_sizes=[3, 5, 7, 11], display=False)```

---
##Variantes testées

MCTS standard : baseline.
RAVE : propagation AMAF, efficace sur petits plateaux.
Heuristic playouts : simulations biaisées par la distance au centre.
Progressive Bias : sélection influencée par une heuristique, qui s’estompe avec les visites.

---
##Résultats principaux

MCTS bat systématiquement les agents Random et Greedy.
Les variantes heuristiques (Heuristic, Progressive Bias) offrent un gain, surtout avec peu de temps de calcul.
RAVE fonctionne bien sur de petits plateaux ($3\times 3$) mais perd en efficacité sur les plus grands.

---
##Améliorations possibles

Playouts heuristiques plus poussés (détection de ponts, chaînes, etc.).
MCTS parallèle pour augmenter le nombre de simulations.
Allocation adaptative du temps de réflexion selon la phase de jeu.
