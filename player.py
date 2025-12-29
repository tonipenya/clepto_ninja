import random
from abc import abstractmethod

import pyspiel

from cleptoninja import Phase, decode_bid, decode_offer, encode_bid, encode_offer


class Player:
    player_id: int

    def __init__(self, player_id):
        self.player_id = player_id

    @property
    def name(self):
        return type(self).__name__

    @abstractmethod
    def action(self, state: pyspiel.State) -> int: ...


class PolicyPlayer(Player):
    def __init__(self, player_id, policy):
        self.player_id = player_id
        self.policy = policy

    @property
    def name(self):
        return super().name + f"(seat:{self.player_id})"

    def action(self, state):
        action_probs = self.policy.action_probabilities(state, self.player_id)
        actions, probs = zip(*action_probs.items())
        return random.choices(actions, weights=probs, k=1)[0]


class RandomPlayer(Player):
    def action(self, state):
        return random.choice(state.legal_actions(self.player_id))


class GreedyPlayer(Player):
    """
    Offer: Lowest two cards
    Bid: Highest two cards if public offer in hand, lowest two otherwise
    """

    def action(self, state):
        legal_actions = state.legal_actions(self.player_id)
        action_decoder = {
            Phase.OFFER: decode_offer,
            Phase.BID: decode_bid,
        }[state._phase]
        legal_action_cards = [
            action_decoder(action, state.max_card) for action in legal_actions
        ]
        sorted_card_pairs = sorted(legal_action_cards, key=sum)

        match (state._phase):
            case Phase.OFFER:
                cards_to_be_played = sorted_card_pairs[0]
            case Phase.BID:
                hand = state._hands[self.player_id]
                public_offer = state._auctions[state._round].offer_public
                cards_to_be_played = (
                    sorted_card_pairs[-1]
                    if public_offer in hand
                    else sorted_card_pairs[0]
                )
            case _:
                raise Exception()

        action_encoder = {
            Phase.OFFER: encode_offer,
            Phase.BID: encode_bid,
        }[state._phase]

        return action_encoder(*cards_to_be_played, state.max_card)


class HumanInTheLoopPlayer(Player):
    def action(self, state):
        print(f"You are player: {self.player_id}")
        print(
            state._string_representation(
                player=self.player_id, hide_private_information=True
            )
        )
        prompt = {
            Phase.OFFER: "Place Offer: ",
            Phase.BID: "Place Bid: ",
        }[state._phase]
        print(prompt, end="", flush=True)

        action_encoder = {
            Phase.OFFER: encode_offer,
            Phase.BID: encode_bid,
        }[state._phase]

        cards = input(prompt)
        c1, c2 = [int(c) for c in cards.split(",")]
        return action_encoder(c1, c2, state.max_card)
