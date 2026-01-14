import os
import random
from abc import abstractmethod
from pathlib import Path

import pyspiel
import torch

from cleptoninja import Phase, decode_bid, decode_offer, encode_bid, encode_offer
from model import ActorCriticModel, encode_observation, legal_action_mask, select_action


class Player:
    @property
    def name(self):
        return type(self).__name__

    @abstractmethod
    def action(self, state: pyspiel.State, player_id: int) -> int: ...

    def __str__(self):
        return self.name

    def __repr__(self) -> str:
        return self.__str__()


class PolicyPlayer(Player):
    def __init__(self, policy):
        self.policy = policy

    def action(self, state, player_id):
        action_probs = self.policy.action_probabilities(state, player_id)
        actions, probs = zip(*action_probs.items())
        return random.choices(actions, weights=probs, k=1)[0]


class RandomPlayer(Player):
    def action(self, state, player_id):
        return random.choice(state.legal_actions(player_id))


class GreedyPlayer(Player):
    """
    Offer: Lowest two cards
    Bid: Highest two cards if public offer in hand, lowest two otherwise
    """

    def action(self, state, player_id):
        legal_actions = state.legal_actions(player_id)
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
                hand = state._hands[player_id]
                public_offer = state._auctions[state._round].offer_public

                if public_offer in hand:
                    # Bid low
                    cards_to_be_played = sorted_card_pairs[-1]
                else:
                    # Bid high, keep card matching offer in hand
                    sorted_card_pairs = [
                        pair for pair in sorted_card_pairs if public_offer not in pair
                    ]
                    cards_to_be_played = sorted_card_pairs[0]
            case _:
                raise Exception()

        action_encoder = {
            Phase.OFFER: encode_offer,
            Phase.BID: encode_bid,
        }[state._phase]

        return action_encoder(*cards_to_be_played, state.max_card)


class HumanInTheLoopPlayer(Player):
    def action(self, state, player_id):
        print(
            state._string_representation(
                player=player_id, hide_private_information=True
            )
        )
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
        action = None
        while action not in state._legal_actions():
            try:
                cards = input(prompt)
                first_card, second_card = [int(c) for c in cards]
                action = action_encoder(first_card, second_card, state.max_card)
            except ValueError as e:
                print("Invalid action!!!", e)

        os.system("cls" if os.name == "nt" else "clear")

        return action


class ActorCriticPlayer(Player):
    def __init__(self, model: ActorCriticModel, name_suffix=""):
        self.model = model
        self.name_suffix = name_suffix

    @property
    def name(self):
        return f"{super().name}{self.name_suffix}"

    def action(self, state, player_id):
        observation = torch.tensor(
            encode_observation(state, player_id), dtype=torch.float32, device="cpu"
        )
        mask = torch.tensor(
            legal_action_mask(state, player_id), dtype=torch.bool, device="cpu"
        )
        action, *_ = select_action(self.model, observation, mask)
        return action

    def export(self, path: Path):
        checkpoint = self.model.checkpoint
        checkpoint["name_suffix"] = self.name_suffix

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path):
        checkpoint = torch.load(path)

        model = ActorCriticModel(
            obs_dim=checkpoint["obs_dim"],
            num_actions=checkpoint["num_actions"],
            hidden=checkpoint["hidden"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return cls(model, name_suffix=checkpoint["name_suffix"])
