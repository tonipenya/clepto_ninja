import os

import pyspiel

from cleptoninja import Phase, encode_bid, encode_offer, register_game as register_cleptoninja_game
from player import ActorCriticPlayer, GreedyPlayer, Player, RandomPlayer


class HumanInTheLoopPlayer(Player):
    def action(self, state, player_id):
        prompt = {
            Phase.OFFER: "Place offer: ",
            Phase.BID: f"Place bid for auction {state._round}: ",
        }[state._phase]

        action_encoder = {
            Phase.OFFER: encode_offer,
            Phase.BID: encode_bid,
        }[state._phase]

        print()
        print(f"You are player: {player_id}")
        action = -1
        while action not in state._legal_actions():
            try:
                cards = input(prompt)
                first_card, second_card = [int(c) for c in cards]
                action = action_encoder(first_card, second_card, state.max_card)
            except ValueError as e:
                print("Invalid action!!!", e)

        return action


def run_game(players: list[Player]):
    register_cleptoninja_game()
    game = pyspiel.load_game("clepto_ninja")
    state = game.new_initial_state()

    while not state.is_terminal():
        os.system("cls" if os.name == "nt" else "clear")
        player_index = state.current_player()
        print(
            state._string_representation(
                player=player_index, hide_private_information=True
            )
        )
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
        input("Game ended. Press [Enter] to continue")


def print_scoreboard(scoreboard):
    print("------------ SCOREBOARD ------------")
    for player in sorted(scoreboard, key=scoreboard.get, reverse=True):
        print(f"| {str(player):<29} {str(scoreboard[player]):>2} |")
    print("------------------------------------")
    print()


if __name__ == "__main__":
    players = [
        RandomPlayer(),
        HumanInTheLoopPlayer(),
        GreedyPlayer(),
        ActorCriticPlayer.load("best_actor_critic_player.pt"),
    ]

    state = run_match(players)
