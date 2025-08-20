from hex.mcts_agent import MCTSAgent
from hex.random_agent import RandomAgent
from hex.greedy_agent import GreedyAgent
from hex.experience import compare_agents
from hex.game import HexGame

# Exemple d'utilisation
if __name__ == "__main__":
    # Partie de démonstration
    game = HexGame(5)
    mcts_agent = MCTSAgent(max_time=0.5)
    random_agent = RandomAgent()
    winner = game.play_game(mcts_agent, random_agent, display=True)

    agents = {
       "MCTS(0.1s)": MCTSAgent(max_time=0.1),
        "MCTS(0.5s)": MCTSAgent(max_time=0.5),
        "Greedy": GreedyAgent(),
        "Random": RandomAgent(),
    }

    compare_agents(agents, num_games=10, board_sizes=[3,5,7,11], display=False)