
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
