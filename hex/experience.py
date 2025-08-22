from hex.player import Player
from hex.game import HexGame
from hex.mcts_agent import MCTSAgent
from hex.random_agent import RandomAgent
import time
import matplotlib.pyplot as plt
import pandas as pd


def compare_agents(agents: dict, num_games: int = 20, board_sizes=[3,5,7,11], display=False):
    """
    Compare plusieurs agents en round-robin.

    agents : dict {nom: instance_agent}
    num_games : nombre de parties par confrontation
    """

    for board_size in board_sizes:
        print(f"\n=== Tournoi round-robin pour plateau {board_size}x{board_size} ===")

        results = {name: {opponent: 0 for opponent in agents} for name in agents}
        total_games = {name: {opponent: 0 for opponent in agents} for name in agents}

        names = list(agents.keys())

        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i == j:
                    continue
                agent1, agent2 = agents[name1], agents[name2]

                for g in range(num_games):
                    # --- alterner les rôles ---
                    if g % 2 == 0:
                        first, second = agent1, agent2
                        first_name, second_name = name1, name2
                    else:
                        first, second = agent2, agent1
                        first_name, second_name = name2, name1

                    game = HexGame(board_size)

                    # --- déroulement de la partie ---
                    while not game.board.is_game_over():
                        current_agent = first if game.current_player == Player.PLAYER1 else second
                        current_name = first_name if game.current_player == Player.PLAYER1 else second_name

                        start_time = time.time()
                        move = current_agent.get_move(game.board, game.current_player)
                        duration = time.time() - start_time

                        game.board.make_move(move[0], move[1], game.current_player)
                        game.current_player = Player.PLAYER2 if game.current_player == Player.PLAYER1 else Player.PLAYER1

                    # --- déterminer le vainqueur ---
                    winner = game.board.get_winner()
                    if winner == Player.PLAYER1:
                        results[first_name][second_name] += 1
                    else:
                        results[second_name][first_name] += 1

                    total_games[first_name][second_name] += 1
                    total_games[second_name][first_name] += 1

        # --- Construire tableau résultats ---
        df = pd.DataFrame(index=agents.keys(), columns=agents.keys())
        for name1 in agents:
            for name2 in agents:
                if name1 == name2:
                    df.loc[name1, name2] = "-"
                else:
                    wins = results[name1][name2]
                    games = total_games[name1][name2]
                    df.loc[name1, name2] = f"{wins}/{games} ({wins/games*100:.1f}%)"

        print("\n=== Résultats round-robin ===")
        print(df)

        # --- Score moyen par agent ---
        avg_scores = {name: (sum(results[name].values()) / max(1, sum(total_games[name].values()))) * 100
                    for name in agents}

        # --- Graphiques ---
        fig, ax = plt.subplots(figsize=(12, 5))

        # Taux de victoire
        ax.bar(avg_scores.keys(), avg_scores.values())
        ax.set_ylabel("Taux de victoire moyen (%)")
        ax.set_title(f"Taux de victoire ({num_games} parties par match)")

        plt.suptitle(f"Tournoi round-robin (plateau {board_size}x{board_size})")
        plt.show()



