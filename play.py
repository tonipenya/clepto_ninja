import copy

import pyspiel

from cleptoninja import register_game as register_cleptoninja_game
from player import GreedyPlayer, HumanInTheLoopPlayer, Player, RandomPlayer


def run_game(players: list[Player]):
    register_cleptoninja_game()
    game = pyspiel.load_game("clepto_ninja")
    state = game.new_initial_state()

    while not state.is_terminal():
        player_index = state.current_player()
        player = players[player_index]
        action = player.action(state, player_index)
        state.apply_action(action)

    return state


def run_match(players: list[Player], winning_score: int = 40):
    scoreboard = {p: 0 for p in players}

    while max(scoreboard.values()) < winning_score:
        payouts = run_game(players).returns()

        # update scoreboard
        for player, payout in zip(players, payouts):
            scoreboard[player] += int(payout)

        # rotate players
        players = players[1:] + players[:1]

        print_scoreboard(scoreboard)


def print_scoreboard(scoreboard):
    print("------------ SCOREBOARD ------------")
    for player in sorted(scoreboard, key=scoreboard.get, reverse=True):
        print(f"| {str(player):<29} {str(scoreboard[player]):>2} |")
    print("------------------------------------")
    print()


if __name__ == "__main__":
    players = [
        RandomPlayer(),
        GreedyPlayer(),
        HumanInTheLoopPlayer(),
        GreedyPlayer(),
    ]

    state = run_match(players)
