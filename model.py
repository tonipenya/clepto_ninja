from pathlib import Path
from typing import Counter

import torch
import torch.nn as nn
from torch.distributions import Categorical

from cleptoninja import Auction, CleptoNinjaState, Phase


class ActorCriticModel(nn.Module):
    """
    Shared actor–critic network.

    Given an observation tensor, outputs:
    - action logits for a discrete policy
    - a scalar state-value estimate V(s)

    Used by PPO as a policy–value (actor–critic) model with a shared backbone.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden = hidden

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, num_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.trunk(obs)
        return self.actor(x), self.critic(x).squeeze(-1)

    @property
    def checkpoint(self):
        return {
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "hidden": self.hidden,
            "state_dict": self.state_dict(),
        }


def encode_observation(state: CleptoNinjaState, player_index: int) -> list[int | float]:
    player_count = len(state._hands)
    max_card_value = state.max_card

    phase_onehot = _onehot(0 if state._phase == Phase.OFFER else 1, size=2)
    round_one_hot = _onehot(state._round, player_count)
    player_onehot = _onehot(player_index, player_count)
    hand_multihot = _multihot(state._hands[player_index], max_card_value, offset=-1)

    auctions = [
        _encode_auction(auction, player_index, player_count, max_card_value)
        for auction in state._auctions
    ]
    missing_auctions_count = player_count - len(auctions)
    auctions += [
        _empty_encoded_auction(player_count, max_card_value)
        for _ in range(missing_auctions_count)
    ]

    encoded = (
        phase_onehot
        + round_one_hot
        + player_onehot
        + hand_multihot
        + _flatten(auctions)
        + [missing_auctions_count / player_count]
    )

    return encoded


def legal_action_mask(state: CleptoNinjaState, player_index: int):
    size = 2 * state.max_card**2
    mask = [0] * size
    for a in state._legal_actions(player_index):
        mask[a] = 1
    return mask


@torch.no_grad()
def select_action(model, observations, mask):
    """
    Sample a legal action from the policy and return PPO-relevant quantities.

    Returns:
    - action (int): sampled action index
    - logprob (float): log π(a|s) of the sampled action
    - entropy (float): policy entropy at the state
    - value (float): critic value estimate V(s)
    """
    logits, value = model(observations)
    dist = _masked_categorical(logits, mask)
    action = dist.sample()
    return (
        int(action.item()),
        float(dist.log_prob(action).item()),
        float(dist.entropy().item()),
        float(value.item()),
    )


def _masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """
    Create a categorical distribution with invalid actions masked out.

    Logits corresponding to mask == False are assigned a large negative value
    so they are never sampled and have zero probability.
    """
    masked_logits = logits.clone()
    masked_logits[~mask] = -1e9
    return Categorical(logits=masked_logits)


def _empty_encoded_auction(player_count, max_card_value) -> list[int | float]:
    expected_bid_count = player_count - 1
    bids_multihot = [
        _multihot([], max_card_value, offset=-1) for _ in range(expected_bid_count)
    ]

    encoded = _flatten(
        [
            [0] * player_count,  # auctioneer_onehot
            0,  # show_offer_private
            [[0] * max_card_value, [0] * max_card_value],  # offer
            expected_bid_count,
            1,  # missing_bids_ration
            bids_multihot,
            [0] * expected_bid_count,  # bid_values
            [0] * expected_bid_count,  # is_bid_present
            [0] * expected_bid_count,  # is_bid_visible
        ]
    )
    return encoded


def _encode_auction(
    auction: Auction, player_index: int, player_count: int, max_card_value: int
) -> list[int | float]:
    auctioneer_onehot = _onehot(auction.auctioneer, player_count)
    show_offer_private = int(
        auction.auctioneer == player_index or auction.all_bids_received
    )
    offer = [
        _onehot(auction.offer_public, max_card_value, offset=-1),
        _onehot(
            auction.offer_private if show_offer_private else None,
            max_card_value,
            offset=-1,
        ),
    ]

    missing_bids_count = auction.expected_bid_count - len(auction.bids)
    is_bid_pressent = [1] * len(auction.bids) + [0] * missing_bids_count
    is_bid_visible = [
        int(bid.bidder == player_index or auction.all_bids_received)
        for bid in auction.bids
    ] + [0] * missing_bids_count
    bids_cards = [
        bid.cards if show_bid else []
        for show_bid, bid in zip(is_bid_visible, auction.bids)
    ] + [[]] * missing_bids_count
    bids_multihot = [
        _multihot(cards, max_card_value, offset=-1) for cards in bids_cards
    ]
    bid_values = [sum(cards) / (2 * max_card_value) for cards in bids_cards]

    encoded = _flatten(
        [
            auctioneer_onehot,
            show_offer_private,
            offer,
            auction.expected_bid_count,
            missing_bids_count / (player_count - 1),
            bids_multihot,
            bid_values,
            is_bid_pressent,
            is_bid_visible,
        ]
    )
    return encoded


def _flatten(xs):
    out = []
    stack = [xs]
    while stack:
        cur = stack.pop()
        if isinstance(cur, list):
            stack.extend(reversed(cur))
        else:
            out.append(cur)
    return out


def _onehot(index, size, offset=0):
    res = [0] * size
    if index is not None:
        res[index + offset] = 1

    return res


def _multihot(cards, size, offset=0):
    res = [0] * size
    for index, count in Counter(cards).items():
        res[index + offset] = count

    return res
