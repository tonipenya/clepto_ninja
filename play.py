from enum import Enum, auto

import pyspiel
from textual import on
from textual.app import App, ComposeResult
from textual.containers import (
    CenterMiddle,
    Container,
    Horizontal,
    HorizontalGroup,
    Vertical,
)
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Label

from cleptoninja import Auction as GameAuction
from cleptoninja import CleptoNinjaState, Phase, encode_bid, encode_offer
from cleptoninja import register_game as register_cleptoninja_game
from player import ActorCriticPlayer, GreedyPlayer, Player, RandomPlayer

END_MATCH_SCORE = 40


class CardRole(Enum):
    PUBLIC_OFFER = auto()
    PRIVATE_OFFER = auto()
    BID = auto()


class Card(Container):
    show_card_value = reactive(True, recompose=True)
    is_selected = reactive(True, recompose=True)
    role: CardRole | None = None

    class Clicked(Message):
        def __init__(self, *args, card: "Card", **kwargs):
            self.card = card
            super().__init__(*args, **kwargs)

    def __init__(
        self,
        *args,
        card_value: int,
        show_card_value: bool,
        enabled: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.value = card_value
        self.is_selected = False
        self.show_card_value = show_card_value
        self.enabled = enabled

    def on_click(self):
        if not self.enabled:
            return
        self.post_message(self.Clicked(card=self))

    def compose(self):
        self.set_class(not self.show_card_value, "card-back")
        self.set_class(self.is_selected, "selected")
        yield Label(str(self.value) if self.show_card_value else "")


class Auction(Horizontal):
    def __init__(self, *args, game_auction: GameAuction, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_auction = game_auction
        self.border_title = " Auction "

    def compose(self):
        yield Card(card_value=self.game_auction.offer_public, show_card_value=True)
        yield Card(
            card_value=self.game_auction.offer_private,
            show_card_value=self.game_auction.all_bids_received,
        )

        show_bid_values = self.game_auction.all_bids_received
        for bid in self.game_auction.bids:
            for card_value in bid.cards:
                yield Card(card_value=card_value, show_card_value=show_bid_values)


class PlayerView(Vertical):
    state: CleptoNinjaState = reactive(None, recompose=True)

    class CardsPlayed(Message):
        pass

    def __init__(
        self,
        *args,
        player: Player,
        player_index: int,
        state: CleptoNinjaState,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.player = player
        self.state: CleptoNinjaState = state
        self.player_index = player_index
        self.border_title = player.name

    @property
    def is_active_auction(self):
        return (
            self.state._phase == Phase.BID
            and self.player_index == self.state._round
            and not self.state.is_terminal()
        )

    @property
    def is_player_turn(self):
        return (
            self.player_index == self.state.current_player()
            and not self.state.is_terminal()
        )

    @on(Card.Clicked)
    def on_card_clicked(self, event: Card.Clicked):
        cards = self.query(Card)
        selected_cards = [card for card in cards if card.is_selected]

        if not event.card.is_selected and len(selected_cards) >= 2:
            # Attempting to select a third card
            return

        match self.state._phase, selected_cards, event.card:
            # Deselect card
            case _, _, Card(is_selected=True):
                event.card.show_card_value = True
                event.card.role = None
                event.card.is_selected = False

            case Phase.OFFER, [], Card(is_selected=False):
                event.card.show_card_value = True
                event.card.role = CardRole.PUBLIC_OFFER
                event.card.is_selected = True

            case (
                Phase.OFFER,
                [Card(role=CardRole.PRIVATE_OFFER)],
                Card(is_selected=False),
            ):
                event.card.show_card_value = True
                event.card.role = CardRole.PUBLIC_OFFER
                event.card.is_selected = True

            case (
                Phase.OFFER,
                [Card(role=CardRole.PUBLIC_OFFER)],
                Card(is_selected=False),
            ):
                event.card.show_card_value = False
                event.card.role = CardRole.PRIVATE_OFFER
                event.card.is_selected = True

            case Phase.BID, _, Card(is_selected=False):
                event.card.show_card_value = True
                event.card.role = CardRole.BID
                event.card.is_selected = True

    def on_button_pressed(self):
        cards = self.query(Card)
        selected_cards = [card for card in cards if card.is_selected]

        if len(selected_cards) != 2:
            return

        first_card, second_card = selected_cards[0:2]

        match self.state._phase:
            case Phase.OFFER:
                public_card = (
                    first_card
                    if first_card.role == CardRole.PUBLIC_OFFER
                    else second_card
                )
                private_card = (
                    first_card
                    if first_card.role == CardRole.PRIVATE_OFFER
                    else second_card
                )

                action = encode_offer(
                    public_card.value, private_card.value, self.state.max_card
                )
            case Phase.BID:
                action = encode_bid(
                    first_card.value, second_card.value, self.state.max_card
                )
            case _:
                raise ValueError

        self.state.apply_action(action)
        self.post_message(self.CardsPlayed())

    def next_action(self):
        if isinstance(self.player, HumanPlayer):
            pass  # Wait for human to press buttons
        else:
            action = self.player.action(self.state, self.state.current_player())
            self.state.apply_action(action)
            self.post_message(self.CardsPlayed())

    def compose(self):
        self.set_class(self.is_active_auction, "active-auction")
        self.set_class(self.is_player_turn, "current-player")

        # Hand
        cards = [
            Card(
                card_value=card_value,
                enabled=self.is_player_turn,
                show_card_value=self.is_player_turn or self.state.is_terminal(),
            )
            for card_value in sorted(self.state._hands[self.player_index])
        ]
        yield HorizontalGroup(*cards)

        if self.state.is_terminal():
            return

        # Auction
        auction = (
            self.state._auctions[self.player_index]
            if len(self.state._auctions) > self.player_index
            else None
        )
        if auction:
            yield Auction(game_auction=auction)

        # Actions
        if self.is_player_turn and isinstance(self.player, HumanPlayer):
            action = self.state._phase
            yield CenterMiddle(
                Button(action.name.capitalize()), classes="player-actions"
            )


class Scoreboard(CenterMiddle):
    scores: reactive[list[tuple[str, int]]] = reactive([], recompose=True)

    def __init__(self, *args, scores: list[tuple[str, int]], **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = scores

    def compose(self):
        table = DataTable()
        table.add_columns("player", "scores")
        table.add_rows(self.scores)
        table.show_cursor = False

        yield table


class CleptoNinjaApp(App):
    CSS_PATH = "style.tcss"

    def __init__(self, *args, players: list[Player], **kwargs):
        super().__init__(*args, **kwargs)
        self.players = players

        register_cleptoninja_game()
        self.game = pyspiel.load_game("clepto_ninja")
        self.state: CleptoNinjaState = self.game.new_initial_state()
        self.player_views = [
            PlayerView(id=f"player{i}", player=player, player_index=i, state=self.state)
            for i, player in enumerate(self.players)
        ]
        self.current_player_view.next_action()

    @property
    def current_player_view(self) -> PlayerView:
        return [
            view
            for view in self.player_views
            if view.player_index == self.state.current_player()
        ][0]

    def compose(self) -> ComposeResult:
        yield Label()
        yield self.player_views[2]
        yield Label()
        yield self.player_views[1]
        yield Scoreboard(scores=[(player.name, 0) for player in self.players])
        yield self.player_views[3]
        yield Label()
        yield self.player_views[0]
        yield Label()

    @on(PlayerView.CardsPlayed)
    def on_hand_played(self, _: PlayerView.CardsPlayed):
        if self.state.is_terminal():
            self.update_score_board()
            self.next_game()

        if isinstance(self.current_player_view.player, HumanPlayer):
            self.refresh_player_views()

        scoreboard = self.query_exactly_one(Scoreboard)
        max_score = max(score for _, score in scoreboard.scores)
        if max_score >= END_MATCH_SCORE:
            return

        self.current_player_view.next_action()

    def refresh_player_views(self):
        for player_view in self.player_views:
            player_view.state = self.state
            player_view.mutate_reactive(PlayerView.state)

    def update_score_board(self):
        scoreboard = self.query_exactly_one(Scoreboard)
        for i, (player_name, score) in enumerate(scoreboard.scores):
            payoff = int(self.state.returns()[i])
            scoreboard.scores[i] = (player_name, score + payoff)

        scoreboard.mutate_reactive(Scoreboard.scores)

    def next_game(self):
        self.state = self.game.new_initial_state()

        for player_view in self.player_views:
            player_view.state = self.state
            # Rotate offer order w.r.t. game.state
            player_view.player_index = (player_view.player_index + 1) % len(
                self.players
            )


class HumanPlayer(Player):
    def action(self, state, player_id):
        raise Exception


if __name__ == "__main__":
    app = CleptoNinjaApp(
        players=[
            HumanPlayer(),
            RandomPlayer(),
            GreedyPlayer(),
            ActorCriticPlayer.load("checkpoints/best_actor_critic_player.pt"),
        ]
    )
    app.run()
