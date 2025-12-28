from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple

import pyspiel

SET_PAYOFFS = {
    2: 1,
    3: 7,
    "full": 11,
    # 4: 161,
    4: 25,
}


# Encode-Decode caveats
# Openspiel action ids must be integers and unique
#
# For card pairs (offers and bids) this can be achieved by adding each cards
# with an offset (the multiplication by max_card_value bit)
#
# To differenciate between bids and offers, bids are further padded by max_card_value **2
def encode_offer(public_card: int, private_card: int, max_card_value: int) -> int:
    return (public_card - 1) * max_card_value + (private_card - 1)


def decode_offer(action: int, max_card_value: int) -> tuple[int, int]:
    public_card = action // max_card_value
    private_card = action % max_card_value
    return public_card + 1, private_card + 1


def encode_bid(first_card: int, second_card: int, max_card_value: int) -> int:
    return max_card_value**2 + (first_card - 1) * max_card_value + (second_card - 1)


def decode_bid(action: int, max_card_value: int) -> tuple[int, int]:
    x = action - max_card_value**2
    first_card = x // max_card_value + 1
    second_card = x % max_card_value + 1

    return first_card, second_card


def legal_offers(hand: list[int]) -> list[tuple[int, int]]:
    offers = []
    for public_card in hand:
        for private_card in hand:
            if public_card == private_card:
                continue
            offers.append((public_card, private_card))

    return offers


def legal_bids(hand: list[int]) -> list[tuple[int, int]]:
    # Second loop avoids repetitions to reduce search space. E.g. (c1,c2) == (c2,c1)
    bids = []
    for i in range(len(hand)):
        for j in range(i + 1, len(hand)):
            first_card, second_card = hand[i], hand[j]
            bids.append((first_card, second_card))

    return bids


def hand_payoff(hand: list[int]) -> float:
    # sets_counter is a dictionary-like where (k,v): (set_size, number of sets of that size)
    # e.g. {2:3, 3:1} -> the loot contains two pairs and a three-of-a-kind
    sets_counter = Counter(Counter(hand).values())

    # extract full-houses (a three-of-a-kind and a pair)
    while sets_counter[3] > 0 and sets_counter[2] > 0:
        sets_counter[3] -= 1
        sets_counter[2] -= 1
        sets_counter["full"] += 1

    payoff = sum((sets_counter[set] * value for set, value in SET_PAYOFFS.items()))

    return payoff


class Phase(Enum):
    OFFER = auto()
    BID = auto()


class Bid(NamedTuple):
    bidder: int
    cards: tuple[int, int]

    @property
    def value(self):
        return sum(self.cards)


@dataclass
class Auction:
    auctioneer: int
    expected_bid_count: int
    offer_public: int
    offer_private: int
    bids: list[Bid] = field(default_factory=list)
    bidding_order: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.bidding_order = [
            p for p in range(self.expected_bid_count + 1) if p != self.auctioneer
        ]

    @property
    def all_bids_received(self):
        return len(self.bids) == self.expected_bid_count

    @property
    def best_bid(self):
        best_bid = self.bids[0]
        for bid in self.bids:
            if best_bid is None or bid.value > best_bid.value:
                best_bid = bid

        return best_bid

    @property
    def next_bidder(self):
        return self.bidding_order[len(self.bids)]


@dataclass
class ResolvedAuction:  # TODO: Merge with Auction?
    auctioneer: int
    offer_public: int
    offer_private: int
    bids: list[Bid]
    winner: int
    winning_bid_value: int

    @staticmethod
    def from_auction(auction: Auction) -> "ResolvedAuction":
        return ResolvedAuction(
            auctioneer=auction.auctioneer,
            offer_public=auction.offer_public,
            offer_private=auction.offer_private,
            bids=list(auction.bids),
            winner=auction.best_bid.bidder,
            winning_bid_value=auction.best_bid.value,
        )


GAME_TYPE = pyspiel.GameType(
    short_name="clepto_ninja",
    long_name="Clepto Ninja!",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=4,  # TODO: Make this dependant on the params again
    min_num_players=4,
    # TODO: Research each of these parameters
    provides_information_state_string=True,
    provides_observation_string=True,
    provides_information_state_tensor=False,
    provides_observation_tensor=False,
    parameter_specification={
        "players": 4,
    },
)


class CleptoNinja(pyspiel.Game):
    def __init__(self, params=None):
        params = params or {}
        self._num_players = int(params.get("players", 4))
        self.max_card = 2 * self._num_players  # card ids are 1..2N
        self._game_length = self._num_players  # N auctions / rounds

        max_poker_count = self.max_card // self._num_players
        filling_pairs_count = self.max_card % self._num_players
        max_hand_payoff = (
            max_poker_count * SET_PAYOFFS[4] + filling_pairs_count * SET_PAYOFFS[2]
        )
        game_info = pyspiel.GameInfo(
            num_distinct_actions=self._num_distinct_actions(),
            max_chance_outcomes=0,
            num_players=self._num_players,
            min_utility=0,
            max_utility=max_hand_payoff,
            utility_sum=0.0,  # unknown; general-sum
            max_game_length=self._game_length * self._num_players,
        )
        super().__init__(GAME_TYPE, game_info, params)

    def new_initial_state(self):
        return CleptoNinjaState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        # Using observation_string() / information_state_string() only.
        return None

    def _num_distinct_actions(self) -> int:
        # Two action families:
        # OFFER: ordered (public, private) with cards in 1..M (M=2N), excluding equal.
        # BID:   unordered (c1,c2) represented as ordered ids with c1 < c2.
        M = self.max_card
        offer_space = M * M
        bid_space = M * M
        return offer_space + bid_space


class CleptoNinjaState(pyspiel.State):
    def __init__(self, game: CleptoNinja):
        super().__init__(game)
        self._game = game
        self.max_card = game.max_card
        self.game_length = game._game_length
        self._phase = Phase.OFFER
        self._hands: list[list[int]] = [
            list(range(1, self.max_card + 1)) for _ in range(self._game.num_players())
        ]
        self._auctions: list[Auction] = []
        self._history: list[ResolvedAuction] = []

    @property
    def _round(self):
        return min(
            len([auction for auction in self._auctions if auction.all_bids_received]),
            self._game.num_players() - 1,  # When all auctions have been resolved
        )

    # ---------- Core OpenSpiel API ----------
    def current_player(self) -> int:
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if self._phase == Phase.OFFER:
            return len(self._auctions)
        # Phase.BID
        return self._auctions[self._round].next_bidder

    def _legal_actions(self, player=None) -> list[int]:
        player = player if player is not None else self.current_player()

        if self.is_terminal() or player != self.current_player():
            return []

        match self._phase:
            case Phase.OFFER:
                return [
                    encode_offer(o, a, self.max_card)
                    for o, a in legal_offers(self._hands[player])
                ]
            case Phase.BID:
                return [
                    encode_bid(b, a, self.max_card)
                    for b, a in legal_bids(self._hands[player])
                ]
            case _:
                return []

    def _apply_action(self, action: int) -> None:
        if self.is_terminal():
            return

        player = self.current_player()
        match self._phase:
            case Phase.OFFER:
                public_card, private_card = decode_offer(action, self.max_card)
                self._remove_from_hand(player, [public_card, private_card])
                self._auctions.append(
                    Auction(
                        auctioneer=player,
                        offer_public=public_card,
                        offer_private=private_card,
                        expected_bid_count=self._game.num_players() - 1,
                    )
                )
                self._phase = (
                    Phase.OFFER
                    if len(self._auctions) < self._game.num_players()
                    else Phase.BID
                )
            case Phase.BID:
                first_card, second_card = decode_bid(action, self.max_card)
                self._remove_from_hand(player, [first_card, second_card])

                current_auction = self._auctions[self._round]
                current_auction.bids.append(
                    Bid(bidder=player, cards=(first_card, second_card))
                )

                if current_auction.all_bids_received:
                    self._exchange_cards(current_auction)
                    self._history.append(ResolvedAuction.from_auction(current_auction))

    def _action_to_string(self, player: int, action: int) -> str:
        M = self.max_card
        if action < M * M:
            pub, priv = decode_offer(action, M)
            return f"{player}: OFFER(pub={pub}, priv={priv})"
        c1, c2 = decode_bid(action, M)
        return f"{player}: BID({c1},{c2})"

    def is_terminal(self) -> bool:
        return len(self._auctions) == self._game.num_players() and all(
            auction.all_bids_received for auction in self._auctions
        )

    def returns(self) -> list[float]:
        if not self.is_terminal():
            return [0.0] * self._game.num_players()

        return [
            float(hand_payoff(self._hands[p])) for p in range(self._game.num_players())
        ]

    # ---------- Information / observation ----------

    def information_state_string(self, player: int) -> str:
        parts = []
        parts.append(f"phase={self._phase.name}")
        parts.append(f"player={player}")
        if self._phase == Phase.BID:
            parts.append(f"round={self._round}/{self.game_length}")

        parts.append(f"Hand={self._hands[player]}")

        auctions = []
        for auction in self._auctions:
            auction_parts = []
            offer_private = (
                auction.offer_private
                if (player == auction.auctioneer or auction.all_bids_received)
                else "?"
            )
            auction_parts.append(
                f"{auction.auctioneer}: Offer({auction.offer_public}, {offer_private})"
            )
            bids = [
                (
                    f"{bid.bidder}:{bid.cards}"
                    if auction.all_bids_received or player == bid.bidder
                    else "?"
                )
                for bid in auction.bids
            ]

            auction_parts.append(f", bids={bids}")
            best_bid = auction.best_bid if auction.all_bids_received else None
            auction_parts.append(f", winner={best_bid}")
            auctions.append(", ".join(auction_parts))
        parts.append(" | ".join(auctions))

        return "\n".join(parts)

    def observation_string(self, player: int) -> str:
        # For simplicity, use the same as information state string.
        return self.information_state_string(player)

    # ---------- Helpers ----------
    def _exchange_cards(self, auction) -> None:
        winner = auction.best_bid.bidder
        self._hands[winner] += [auction.offer_public, auction.offer_private]
        self._hands[auction.auctioneer] += auction.best_bid.cards

        # other bidders recover their bids
        for bidder, cards in auction.bids:
            if bidder == winner:
                continue
            self._hands[bidder] += cards

    def _remove_from_hand(self, player: int, cards: list[int]) -> None:
        for card in cards:
            self._hands[player].remove(card)

    def _string_representation(
        self, player=None, hide_private_information=False
    ) -> str:
        parts = []
        header = []
        header.append(f"phase={self._phase.name}")
        header.append(f"player={player}")
        if self._phase == Phase.BID:
            header.append(f"round={self._round}/{self.game_length}")
        parts.append(" | ".join(header))

        parts.append("Hands ---------------------")
        for hand_idx, hand in enumerate(self._hands):
            if hand_idx != player and hide_private_information:
                hand = ["*" for _ in hand]
            parts.append(f"{hand_idx}: {sorted(hand)}")

        parts.append("")
        parts.append("Auctions ------------------")
        for auction in self._auctions:
            offer_private = (
                auction.offer_private
                if (
                    player == auction.auctioneer
                    or auction.all_bids_received
                    or not hide_private_information
                )
                else "??"
            )
            parts.append(
                f"{auction.auctioneer}: Offer({auction.offer_public}, {offer_private})"
            )
            bids = [
                (
                    bid
                    if auction.all_bids_received
                    or player == bid.bidder
                    or not hide_private_information
                    else "??"
                )
                for bid in auction.bids
            ]

            parts.append(f"   bids: {bids}")
            best_bid = auction.best_bid if auction.all_bids_received else None
            parts.append(f"   winner bid: {best_bid}")

        return "\n".join(parts)

    def __str__(self) -> str:
        return self._string_representation()


def register_game():
    pyspiel.register_game(GAME_TYPE, CleptoNinja)
