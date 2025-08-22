from hex.mcts_agent import MCTSAgent
from hex.random_agent import RandomAgent
from hex.greedy_agent import GreedyAgent
from hex.experience import compare_agents
from hex.game import HexGame
from hex.mctsrave_agent import MCTSRAVEAgent
from hex.mctsheuristique_agent import MCTSHeuristicAgent
from hex.mctsprog_agent import MCTSProgressiveBiasAgent

# Exemple d'utilisations
if __name__ == "__main__":
    # Partie de démonstration
    game = HexGame(5)
    mcts_agent = MCTSAgent(max_time=0.5)
    random_agent = RandomAgent()
    winner = game.play_game(mcts_agent, random_agent, display=True)

    agents0 = {
        "MCTS(0.1s)": MCTSAgent(max_time=0.1),
        "MCTS(0.5s)": MCTSAgent(max_time=0.5),
        "Greedy": GreedyAgent(),
        "Random": RandomAgent(),
     }

    compare_agents(agents0, num_games=10, board_sizes=[3,5,7,11], display=False)

    agents1 = {
    "MCTS(0.1s)": MCTSAgent(max_time=0.1),
    "RAVE(0.1s)": MCTSRAVEAgent(max_time=0.1),
    "Heuristic(0.1s)": MCTSHeuristicAgent(max_time=0.1),
    "ProgBias(0.1s)": MCTSProgressiveBiasAgent(max_time=0.1),
    "Random": RandomAgent(),
    "Greedy" : GreedyAgent()
}

    compare_agents(agents1, num_games=50, board_sizes=[3, 5, 7], display=False)

    agents2 = {
        "MCTS(0.5s)": MCTSAgent(max_time=0.5),
        "RAVE(0.5s)": MCTSRAVEAgent(max_time=0.5),
        "Heuristic(0.5s)": MCTSHeuristicAgent(max_time=0.5),
        "ProgBias(0.5s)": MCTSProgressiveBiasAgent(max_time=0.5),
        "Greedy" : GreedyAgent()
    }

    compare_agents(agents2, num_games=50, board_sizes=[3, 5, 7], display=False)
