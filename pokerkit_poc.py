from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pokerkit import Automation
from pokerkit.games import NoLimitShortDeckHoldem


@dataclass(frozen=True)
class ShortDeckNoLimitConfig:
    num_players: int = 2
    starting_stacks: Tuple[int, int] = (60, 60)
    ante: int = 0
    small_blind: int = 1
    big_blind: int = 2
    min_bet: int = 1
    streets: Tuple[str, ...] = ("preflop", "flop", "turn", "river")

    @property
    def runtime_values(self) -> Dict[str, Any]:
        return {
            "num_players": self.num_players,
            "raw_starting_stacks": list(self.starting_stacks),
            "raw_antes": [self.ante, self.ante],
            "raw_blinds_or_straddles": [self.small_blind, self.big_blind],
            "min_bet": self.min_bet,
            "streets": list(self.streets),
        }


@dataclass
class StrategyTracker:
    action_counts: Counter = field(default_factory=Counter)
    street_counts: Counter = field(default_factory=Counter)
    payoff_samples: List[float] = field(default_factory=list)
    strategy_history: List[Dict[str, float]] = field(default_factory=list)
    last_policy: Dict[str, float] = field(default_factory=dict)

    def record_action(self, action: str, street: Any) -> None:
        self.action_counts[action] += 1
        self.street_counts[str(street)] += 1

    def record_payoff(self, payoff: float) -> None:
        self.payoff_samples.append(float(payoff))

    def record_policy_snapshot(self, policy: Dict[str, float]) -> None:
        self.last_policy = dict(policy)
        self.strategy_history.append(dict(policy))

    def summary(self) -> Dict[str, Any]:
        total = sum(self.action_counts.values())
        action_freq = {k: float(v) / total if total else 0.0 for k, v in self.action_counts.items()}
        if self.payoff_samples:
            avg_payoff = sum(self.payoff_samples) / len(self.payoff_samples)
            min_payoff = min(self.payoff_samples)
            max_payoff = max(self.payoff_samples)
        else:
            avg_payoff = 0.0
            min_payoff = 0.0
            max_payoff = 0.0

        return {
            "total_observed_actions": total,
            "action_frequency": dict(sorted(action_freq.items())),
            "street_frequency": dict(sorted(self.street_counts.items())),
            "avg_payoff": avg_payoff,
            "min_payoff": min_payoff,
            "max_payoff": max_payoff,
            "last_policy": self.last_policy,
        }

    def strategy_impact_summary(self) -> Dict[str, Any]:
        if len(self.strategy_history) < 2:
            return {
                "observed_policy_snapshots": len(self.strategy_history),
                "policy_shift": self.last_policy,
                "avg_payoff": self.summary()["avg_payoff"],
            }

        first = self.strategy_history[0]
        last = self.strategy_history[-1]
        delta = {k: float(last.get(k, 0.0) - first.get(k, 0.0)) for k in set(first) | set(last)}
        return {
            "observed_policy_snapshots": len(self.strategy_history),
            "first_policy": first,
            "last_policy": last,
            "policy_shift": dict(sorted(delta.items())),
            "avg_payoff": self.summary()["avg_payoff"],
        }


@dataclass(frozen=True)
class NodeObservation:
    street: str
    actor_index: int
    public_board_key: str
    private_key: str
    full_info_key: str

    @staticmethod
    def _rank_value(card: Any) -> int:
        text = str(card)
        if len(text) < 2:
            return 0
        rank = text[:-1].upper()
        rank_map = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
        return rank_map.get(rank, 0)

    @staticmethod
    def _rank_label(value: int) -> str:
        labels = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T", 9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "3", 2: "2"}
        return labels.get(value, str(value))

    @staticmethod
    def _canonical_exact_cards(cards: List[Any]) -> str:
        ordered = sorted(cards, key=lambda card: (NodeObservation._rank_value(card), str(card)))
        return "".join(str(card) for card in ordered)

    @staticmethod
    def _canonical_preflop_hole(cards: List[Any]) -> str:
        if len(cards) != 2:
            return "unknown"
        ranks = sorted((NodeObservation._rank_value(card) for card in cards))
        rank_text = "".join(NodeObservation._rank_label(value) for value in ranks)
        suited = str(cards[0])[-1].lower() == str(cards[1])[-1].lower()
        if ranks[0] == ranks[1]:
            return rank_text
        return f"{rank_text}{'s' if suited else 'o'}"

    @classmethod
    def from_state(cls, state, actor_index: Optional[int] = None) -> "NodeObservation":
        street = ActionSpaceReducer._current_street_name(state)
        actor_index = actor_index if actor_index is not None else int(getattr(state, "actor_index", 0) or 0)
        board_cards = list(getattr(state, "board_cards", []) or [])
        hole_cards = list(getattr(state, "hole_cards", []) or [])
        public_board_key = cls._canonical_exact_cards(board_cards) if board_cards else "preflop"
        private_key = cls._canonical_preflop_hole(hole_cards) if street == "preflop" else cls._canonical_exact_cards(hole_cards)
        full_info_key = f"{street}:{actor_index}:{public_board_key}:{private_key}"
        return cls(
            street=street,
            actor_index=actor_index,
            public_board_key=public_board_key,
            private_key=private_key,
            full_info_key=full_info_key,
        )


@dataclass
class StreetActionRule:
    allow_limp: bool = False
    opening_raise_amounts: Tuple[int, ...] = (4,)
    bet_amounts: Tuple[int, ...] = ()
    raise_amounts: Tuple[int, ...] = ()
    allowed_bet_pcts: Tuple[float, ...] = ()
    allow_all_in: bool = True
    raise_multiplier: Optional[float] = 4.0
    raise_only_all_in: bool = False
    first_to_act_allowed: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "StreetActionRule":
        if not data:
            return cls()
        bet_amounts = tuple(int(v) for v in data.get("bet_amounts", ()))
        raise_amounts = tuple(int(v) for v in data.get("raise_amounts", ()))
        opening_raise_amounts = tuple(int(v) for v in data.get("opening_raise_amounts", bet_amounts or (4,)))
        first_to_act_allowed = tuple(str(v) for v in data.get("first_to_act_allowed", ()))
        return cls(
            allow_limp=bool(data.get("allow_limp", False)),
            opening_raise_amounts=opening_raise_amounts,
            bet_amounts=bet_amounts,
            raise_amounts=raise_amounts,
            allowed_bet_pcts=tuple(float(v) for v in data.get("allowed_bet_pcts", ())),
            allow_all_in=bool(data.get("allow_all_in", True)),
            raise_multiplier=data.get("raise_multiplier"),
            raise_only_all_in=bool(data.get("raise_only_all_in", False)),
            first_to_act_allowed=first_to_act_allowed,
        )


@dataclass
class StructuredActionPolicy:
    streets: Dict[str, StreetActionRule] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "StructuredActionPolicy":
        if not data:
            return cls()
        streets = data.get("streets", {})
        return cls(
            streets={
                street_name: StreetActionRule.from_dict(raw_rule)
                for street_name, raw_rule in streets.items()
            }
        )

    @classmethod
    def from_json_path(cls, path: str) -> "StructuredActionPolicy":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class ActionSpaceReducer:
    max_legal_actions: Optional[int] = None
    allowed_bet_amounts: Optional[Tuple[int, ...]] = None
    allow_check_or_call: bool = True
    allow_fold: bool = True
    policy: Optional[StructuredActionPolicy] = None
    warn_on_empty: bool = True
    _warned_empty_actions: bool = field(default=False, init=False, repr=False)

    @staticmethod
    def _current_street_name(state) -> str:
        street = getattr(state, "street", None)

        if street is not None:
            board_count = getattr(street, "board_dealing_count", None)
            if board_count == 0:
                return "preflop"
            if board_count == 3:
                return "flop"
            if board_count == 4:
                return "turn"
            if board_count == 5:
                return "river"

        board_cards = list(getattr(state, "board_cards", []) or [])
        if len(board_cards) == 0:
            return "preflop"
        if len(board_cards) == 3:
            return "flop"
        if len(board_cards) == 4:
            return "turn"
        if len(board_cards) == 5:
            return "river"

        street_index = getattr(state, "street_index", None)
        if street_index is not None:
            if street_index == 0:
                return "preflop"
            if street_index == 1:
                return "flop"
            if street_index == 2:
                return "turn"
            if street_index == 3:
                return "river"

        return "unknown"

    @staticmethod
    def _nearest_pot_targets(state, percentages: Tuple[float, ...]) -> Tuple[int, ...]:
        pot = int(getattr(state, "total_pot_amount", 0) or 0)
        if pot <= 0 or not percentages:
            return ()
        targets = []
        for pct in percentages:
            target = max(1, round(pot * float(pct)))
            targets.append(target)
        return tuple(sorted(set(targets)))

    @staticmethod
    def _allowed_amounts_from_pot(state, percentages: Tuple[float, ...], legal_actions: List[Tuple[str, Optional[int]]]) -> Tuple[int, ...]:
        targets = ActionSpaceReducer._nearest_pot_targets(state, percentages)
        if not targets:
            return ()
        if not legal_actions:
            return ()
        legal_amounts = [int(amount) for _, amount in legal_actions if amount is not None]
        if not legal_amounts:
            return ()
        matched = []
        for target in targets:
            if not legal_amounts:
                continue
            closest = min(legal_amounts, key=lambda value: abs(value - target))
            matched.append(closest)
        return tuple(sorted(set(matched)))

    @staticmethod
    def _legal_bet_amounts_for_street(state, street_rule: StreetActionRule, legal_actions: List[Tuple[str, Optional[int]]]) -> Tuple[int, ...]:
        legal_bets = []
        for _, amount in legal_actions:
            if amount is None:
                continue
            legal_bets.append(int(amount))
        if street_rule.opening_raise_amounts:
            legal_bets.extend(int(x) for x in street_rule.opening_raise_amounts)
        if street_rule.allowed_bet_pcts:
            legal_bets.extend(ActionSpaceReducer._allowed_amounts_from_pot(state, street_rule.allowed_bet_pcts, legal_actions))
        return tuple(sorted(set(legal_bets)))

    @staticmethod
    @staticmethod
    def _is_raise_state(state) -> bool:
        current_bet = int(getattr(state, "current_bet", 0) or 0)
        last_bet = int(getattr(state, "last_bet", 0) or 0)
        previous_bet = int(getattr(state, "previous_bet", 0) or 0)
        return current_bet > 0 or last_bet > 0 or previous_bet > 0

    @staticmethod
    def _is_first_to_act_preflop_limp_candidate(state) -> bool:
        if ActionSpaceReducer._current_street_name(state) != "preflop":
            return False

        if ActionSpaceReducer._is_raise_state(state):
            return False

        actor_index = getattr(state, "actor_index", None)
        opener_index = getattr(state, "opener_index", None)
        if actor_index is None or opener_index is None:
            return False

        if actor_index != opener_index:
            return False

        completion_count = int(getattr(state, "completion_betting_or_raising_count", 0) or 0)
        completion_amount = int(getattr(state, "completion_betting_or_raising_amount", 0) or 0)
        if completion_count > 0 or completion_amount > 0:
            return False

        acted_players = getattr(state, "acted_player_indices", set())
        if acted_players:
            return False

        return True

    @staticmethod
    def _is_first_to_act_for_street(state) -> bool:
        actor_index = getattr(state, "actor_index", None)
        opener_index = getattr(state, "opener_index", None)
        if actor_index is None or opener_index is None:
            return False
        if actor_index != opener_index:
            return False

        completion_count = int(getattr(state, "completion_betting_or_raising_count", 0) or 0)
        completion_amount = int(getattr(state, "completion_betting_or_raising_amount", 0) or 0)
        if completion_count > 0 or completion_amount > 0:
            return False

        acted_players = getattr(state, "acted_player_indices", set())
        if acted_players:
            return False

        return True

    @staticmethod
    def _chosen_amount_for_policy(state, action, street_rule: StreetActionRule) -> Optional[int]:
        if not isinstance(action, tuple) or len(action) != 2:
            return None
        _, amount = action
        if amount is None:
            return None
        return int(amount)

    def _street_rule_for_state(self, state) -> StreetActionRule:
        if self.policy is None:
            return StreetActionRule()
        return self.policy.streets.get(self._current_street_name(state), StreetActionRule())

    def reduce(
        self,
        state,
        legal_actions: List[Tuple[str, Optional[int]]],
    ) -> List[Tuple[str, Optional[int]]]:
        mandatory: List[Tuple[str, Optional[int]]] = []
        bet_like: List[Tuple[str, Optional[int]]] = []
        street_rule = self._street_rule_for_state(state)
        current_street = self._current_street_name(state)
        is_raise_state = self._is_raise_state(state)

        first_to_act_allowed = set(str(v).lower() for v in street_rule.first_to_act_allowed)
        is_first_to_act = self._is_first_to_act_for_street(state)

        for action in legal_actions:
            name, amount = action
            name = name.lower()

            if is_first_to_act and first_to_act_allowed:
                if name not in first_to_act_allowed:
                    continue

            if name in {"check_or_call", "check", "call"}:
                if not self.allow_check_or_call:
                    continue

                is_limp_preflop = (
                    current_street == "preflop"
                    and not street_rule.allow_limp
                    and ActionSpaceReducer._is_first_to_act_preflop_limp_candidate(state)
                )
                if is_limp_preflop:
                    continue

                mandatory.append(action)
                continue
            if name in {"fold", "fold_action"} and self.allow_fold:
                mandatory.append(action)
                continue
            if name in {"bet_or_raise", "bet", "raise", "complete_bet_or_raise_to"}:
                if amount is None:
                    continue

                amount_int = int(amount)
                active_bet = int(getattr(state, "current_bet", 0) or getattr(state, "last_bet", 0) or 0)
                base_bet = int((street_rule.bet_amounts or street_rule.opening_raise_amounts or (1,))[0])
                min_raise_target = None
                if street_rule.raise_multiplier is not None:
                    if active_bet > 0:
                        min_raise_target = max(base_bet, int(round(active_bet * street_rule.raise_multiplier)))
                    else:
                        min_raise_target = base_bet

                if is_raise_state and active_bet > 0:
                    if min_raise_target is not None and amount_int < min_raise_target:
                        continue
                    if street_rule.raise_amounts and amount_int < min(street_rule.raise_amounts):
                        continue
                else:
                    if amount_int < base_bet:
                        continue

                # Preserve the canonical PokerKit legal family and apply the policy as a
                # state-aware filter rather than replacing the legal family with a single
                # hardcoded amount set.
                if self.allowed_bet_amounts is not None:
                    if amount_int not in self.allowed_bet_amounts:
                        continue

                if not state.can_complete_bet_or_raise_to(amount_int):
                    continue

                if street_rule.allowed_bet_pcts:
                    pot_targets = self._nearest_pot_targets(state, street_rule.allowed_bet_pcts)
                    if street_rule.raise_only_all_in:
                        all_in = int(getattr(state, "stacks", [0, 0])[getattr(state, "actor_index", 0)] if hasattr(state, "stacks") and hasattr(state, "actor_index") else 0)
                        if amount_int < all_in and pot_targets and amount_int not in pot_targets:
                            continue

                chosen_amount = self._chosen_amount_for_policy(state, action, street_rule)
                if chosen_amount is not None and not state.can_complete_bet_or_raise_to(int(chosen_amount)):
                    continue
                bet_like.append((name, chosen_amount))

        filtered = mandatory + bet_like

        if self.max_legal_actions is not None and self.max_legal_actions > 0:
            if len(filtered) > self.max_legal_actions:
                allowed_bet_slots = max(self.max_legal_actions - len(mandatory), 0)
                filtered = mandatory + bet_like[:allowed_bet_slots]

        raw_bet_actions = [action for action in legal_actions if action[0].lower() in {"bet_or_raise", "bet", "raise", "complete_bet_or_raise_to"}]
        if raw_bet_actions and not any(name.lower() in {"bet_or_raise", "bet", "raise", "complete_bet_or_raise_to"} for name, _ in filtered):
            filtered = [action for action in filtered if action[0].lower() in {"check_or_call", "call", "check", "fold", "fold_action"}]

        if not legal_actions:
            return []

        if not filtered:
            if self.warn_on_empty and not self._warned_empty_actions:
                print(
                    "WARNING: policy filter removed all legal actions; "
                    "returning empty legal actions by policy. "
                    f"street={current_street}, actor={getattr(state, 'actor_index', None)}, "
                    f"legal={legal_actions}"
                )
                self._warned_empty_actions = True
            return []

        return filtered


def build_config() -> ShortDeckNoLimitConfig:
    return ShortDeckNoLimitConfig(
        num_players=2,
        starting_stacks=(60, 60),
        ante=0,
        small_blind=1,
        big_blind=2,
        min_bet=1,
        streets=("preflop", "flop", "turn", "river"),
    )


def build_state(spec: ShortDeckNoLimitConfig):
    return NoLimitShortDeckHoldem.create_state(
        automations=(
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        ante_trimming_status=False,
        raw_antes=(spec.ante, spec.ante),
        raw_blinds_or_straddles=(spec.small_blind, spec.big_blind),
        min_bet=spec.min_bet,
        raw_starting_stacks=spec.starting_stacks,
        player_count=spec.num_players,
    )


def legal_actions_for_state(
    state,
    reducer: Optional[ActionSpaceReducer] = None,
) -> List[Tuple[str, Optional[int]]]:
    actions: List[Tuple[str, Optional[int]]] = []

    if callable(getattr(state, "can_check_or_call", None)) and state.can_check_or_call():
        actions.append(("check_or_call", 0))

    if callable(getattr(state, "can_fold", None)) and state.can_fold():
        actions.append(("fold", 0))

    can_complete = getattr(state, "can_complete_bet_or_raise_to", None)
    if callable(can_complete):
        for amount in (1, 2, 4, 8, 16, 32, 60):
            if can_complete(amount):
                actions.append(("bet_or_raise", int(amount)))

    if reducer is not None:
        filtered = reducer.reduce(state, actions)
        return filtered
    return actions


def choose_uniform_action(
    state,
    reducer: Optional[ActionSpaceReducer] = None,
) -> Optional[Tuple[str, Optional[int]]]:
    legal = legal_actions_for_state(state, reducer=reducer)
    if not legal:
        return None
    return random.choice(legal)


def apply_action(state, action: Tuple[str, Optional[int]]) -> None:
    name, amount = action
    name = name.lower()

    if name in {"check_or_call", "check", "call"}:
        state.check_or_call()
        return
    if name in {"fold", "fold_action"}:
        state.fold()
        return
    if name in {"bet_or_raise", "bet", "raise", "complete_bet_or_raise_to"}:
        target = int(amount) if amount is not None else 1
        state.complete_bet_or_raise_to(target)
        return
    raise ValueError(f"Unsupported action for this state: {action}")


def _advance_dealers(state) -> None:
    if callable(getattr(state, "can_deal_hole", None)) and state.can_deal_hole():
        state.deal_hole()
        return
    if callable(getattr(state, "can_deal_board", None)) and state.can_deal_board():
        state.deal_board()


def _policy_from_action_counts(tracker: StrategyTracker) -> Dict[str, float]:
    total = sum(tracker.action_counts.values())
    if total == 0:
        return {"check_or_call": 0.5, "bet_or_raise": 0.5, "fold": 0.0}
    return {action: count / total for action, count in tracker.action_counts.items()}


def river_terminal_resolution(state) -> str:
    """Describe the terminal resolution mode for the current state.

    We intentionally leave the actual winner determination to PokerKit. For a
    river showdown in which the last bet/raise was called, the state should be
    resolved through the standard showdown path. If the hand has reached an all-in
    situation, PokerKit's runout-count/showdown machinery is used to sample the
    appropriate runouts rather than using a synthetic rule. In both cases the
    terminal path is considered a PokerKit-resolved state, not a custom rule.
    """
    if getattr(state, 'status', None) is False:
        return 'terminal'
    if getattr(state, 'all_in_status', False):
        return 'all_in_runout'
    if ActionSpaceReducer._current_street_name(state) == 'river':
        return 'river_showdown'
    return 'nonterminal'


def advance_state_to_terminal(
    state,
    *,
    reducer: Optional[ActionSpaceReducer] = None,
    max_steps: int = 200,
) -> tuple[int, Any]:
    """Advance a PokerKit state through the full hand until it is terminal.

    River resolution follows the real PokerKit path: a last bet/raise being called
    should continue through the native showdown flow and not be treated as a
    synthetic custom terminal state. All-in situations remain under PokerKit's
    runout-selection/showdown machinery so repeated runouts are sampled by the
    engine itself.
    """
    for step in range(max_steps):
        if getattr(state, 'status', None) is False:
            return step, state

        resolution = river_terminal_resolution(state)
        if resolution == 'river_showdown':
            if callable(getattr(state, 'can_select_runout_count', None)) and state.can_select_runout_count():
                state.select_runout_count(None)
                continue
            if callable(getattr(state, 'can_show_or_muck_hole_cards', None)) and state.can_show_or_muck_hole_cards():
                state.show_or_muck_hole_cards()
                continue

        legal = legal_actions_for_state(state, reducer=reducer)
        if not legal:
            if callable(getattr(state, 'can_select_runout_count', None)) and state.can_select_runout_count():
                state.select_runout_count(None)
                continue
            if callable(getattr(state, 'can_show_or_muck_hole_cards', None)) and state.can_show_or_muck_hole_cards():
                state.show_or_muck_hole_cards()
                continue
            if callable(getattr(state, 'can_deal_board', None)) and state.can_deal_board():
                state.deal_board()
                continue
            break

        action = choose_uniform_action(state, reducer=reducer)
        if action is None:
            break

        apply_action(state, action)

        if getattr(state, 'status', None) is False:
            return step + 1, state

    return max_steps, state


def simulate_uniform_hand_to_showdown(
    spec: ShortDeckNoLimitConfig,
    *,
    verbose: bool = False,
    max_steps: int = 200,
    reducer: Optional[ActionSpaceReducer] = None,
) -> Dict[str, Any]:
    state = build_state(spec)

    if hasattr(state, "can_collect_bets") and state.can_collect_bets():
        state.collect_bets()
    if hasattr(state, "can_post_blind_or_straddle"):
        for _ in range(2):
            if state.can_post_blind_or_straddle():
                state.post_blind_or_straddle()
    if hasattr(state, "can_deal_hole"):
        for _ in range(getattr(state, "player_count", spec.num_players) * 2):
            if state.can_deal_hole():
                state.deal_hole()

    tracker = StrategyTracker()
    trace: List[Dict[str, Any]] = []
    step = 0

    while step < max_steps:
        if getattr(state, 'status', None) is False:
            break

        legal = legal_actions_for_state(state, reducer=reducer)
        if not legal:
            if callable(getattr(state, 'can_select_runout_count', None)) and state.can_select_runout_count():
                state.select_runout_count(None)
                if verbose:
                    trace.append({
                        "step": step,
                        "event": "runout_count_selection",
                        "street": str(getattr(state, "street", "unknown")),
                        "runout_count": getattr(state, "runout_count", None),
                    })
                step += 1
                continue
            if callable(getattr(state, "can_show_or_muck_hole_cards", None)) and state.can_show_or_muck_hole_cards():
                state.show_or_muck_hole_cards()
                if verbose:
                    trace.append({
                        "step": step,
                        "event": "show_or_muck",
                        "street": str(getattr(state, "street", "unknown")),
                        "showdown_index": getattr(state, "showdown_index", None),
                    })
                step += 1
                continue
            if callable(getattr(state, "can_deal_board", None)) and state.can_deal_board():
                _advance_dealers(state)
                if verbose:
                    trace.append({
                        "step": step,
                        "event": "deal_or_advance",
                        "street": str(getattr(state, "street", "unknown")),
                        "board": list(getattr(state, "board_cards", []) or []),
                        "hole": list(getattr(state, "hole_cards", []) or []),
                    })
                step += 1
                continue
            break

        action = choose_uniform_action(state, reducer=reducer)
        if action is None:
            break

        action_name, amount = action
        tracker.record_action(action_name, getattr(state, "street", "unknown"))
        tracker.record_payoff(float(getattr(state, "total_pot_amount", 0) or 0))

        if verbose:
            raw_actions = []
            if callable(getattr(state, "can_check_or_call", None)) and state.can_check_or_call():
                raw_actions.append(["check_or_call", 0])
            if callable(getattr(state, "can_fold", None)) and state.can_fold():
                raw_actions.append(["fold", 0])
            can_complete = getattr(state, "can_complete_bet_or_raise_to", None)
            if callable(can_complete):
                for amount in (1, 2, 4, 8, 16, 32, 60):
                    if can_complete(amount):
                        raw_actions.append(["bet_or_raise", int(amount)])
            trace.append({
                "step": step,
                "actor": getattr(state, "actor_index", None),
                "street": str(getattr(state, "street", "unknown")),
                "raw_legal_actions": raw_actions,
                "filtered_legal_actions": [list(a) for a in legal],
                "chosen_action": [action_name, amount],
                "stacks_before": list(getattr(state, "stacks", []) or []),
                "board_before": list(getattr(state, "board_cards", []) or []),
            })

        apply_action(state, action)

        if getattr(state, "status", None) is False:
            break

        if callable(getattr(state, "can_deal_board", None)) and state.can_deal_board() and getattr(state, "street", None) is not None:
            prev_board = list(getattr(state, "board_cards", []) or [])
            state.deal_board()
            if verbose:
                trace.append({
                    "step": step,
                    "event": "board_deal",
                    "street_after": str(getattr(state, "street", "unknown")),
                    "board_before": prev_board,
                    "board_after": list(getattr(state, "board_cards", []) or []),
                })

        tracker.record_policy_snapshot(_policy_from_action_counts(tracker))
        step += 1

        if getattr(state, "status", None) is not None and str(getattr(state, "status")).lower().endswith("terminal"):
            break

    result = {
        "final_stacks": list(getattr(state, "stacks", []) or []),
        "final_board": list(getattr(state, "board_cards", []) or []),
        "final_hole": list(getattr(state, "hole_cards", []) or []),
        "final_status": str(getattr(state, "status", "unknown")),
        "stats": tracker.summary(),
        "strategy_impact": tracker.strategy_impact_summary(),
    }
    if verbose:
        result["trace"] = trace
    return result


def example_regret_update() -> Dict[str, Any]:
    actions = ["check_or_call", "bet_or_raise", "fold"]
    policy = {action: 1.0 / len(actions) for action in actions}
    counterfactual_values = {
        "check_or_call": 0.10,
        "bet_or_raise": -0.05,
        "fold": -0.12,
    }
    average_value = sum(policy[action] * counterfactual_values[action] for action in actions)
    regrets = {
        action: counterfactual_values[action] - average_value for action in actions
    }
    return {
        "info_set": "preflop:player_0:[]",
        "uniform_policy": policy,
        "counterfactual_values": counterfactual_values,
        "average_value": average_value,
        "regret_update": regrets,
    }


def _pp(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def compact_action_repr(action: Tuple[str, Optional[int]]) -> str:
    name, amount = action
    if amount is None:
        return str(name)
    return f"{name}:{amount}"


def parse_action_repr(text: str) -> Tuple[str, Optional[int]]:
    if text is None:
        return ('', None)
    if ':' not in text:
        return (text, 0)
    name, amount = text.split(':', 1)
    return (name, int(amount))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short-deck HUNL smoke test")
    parser.add_argument("--verbose", action="store_true", help="Print the full action trace and board progression.")
    parser.add_argument("--policy-json", type=str, default=None, help="Optional JSON file describing street-aware action space rules.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of uniform smoke-test hands to run.")
    return parser.parse_args()


def build_default_policy() -> StructuredActionPolicy:
    return StructuredActionPolicy(
        streets={
            "preflop": StreetActionRule(
                allow_limp=False,
                bet_amounts=(4,),
                raise_amounts=(8, 16),
                opening_raise_amounts=(4,),
            ),
            "flop": StreetActionRule(
                bet_amounts=(4,),
                raise_amounts=(8, 16, 32),
                allowed_bet_pcts=(0.5, 1.0),
                allow_all_in=True,
            ),
            "turn": StreetActionRule(
                bet_amounts=(4,),
                raise_amounts=(8, 16, 32),
                allowed_bet_pcts=(0.5, 1.0),
                allow_all_in=True,
            ),
            "river": StreetActionRule(
                bet_amounts=(4,),
                raise_amounts=(8, 16, 32),
                allowed_bet_pcts=(0.3, 1.0),
                allow_all_in=True,
                raise_only_all_in=True,
            ),
        }
    )


def main() -> None:
    args = parse_args()
    spec = build_config()
    policy = build_default_policy()
    if args.policy_json:
        policy = StructuredActionPolicy.from_json_path(args.policy_json)

    reducer = ActionSpaceReducer(
        max_legal_actions=6,
        allowed_bet_amounts=(1, 2, 4, 8, 16, 32, 60),
        policy=policy,
    )

    print("Short-deck HUNL config:")
    print(_pp(spec.runtime_values))

    state = build_state(spec)
    print("PokerKit runtime state:")
    print(type(state).__name__)
    print("Initial stacks:")
    print(_pp(list(state.stacks)))

    print("\nAction-space reducer:")
    print(_pp({
        "max_legal_actions": reducer.max_legal_actions,
        "allowed_bet_amounts": list(reducer.allowed_bet_amounts or []),
        "policy_streets": {k: {
            "allow_limp": v.allow_limp,
            "bet_amounts": list(v.bet_amounts),
            "raise_amounts": list(v.raise_amounts),
            "opening_raise_amounts": list(v.opening_raise_amounts),
            "allowed_bet_pcts": list(v.allowed_bet_pcts),
            "allow_all_in": v.allow_all_in,
            "raise_only_all_in": v.raise_only_all_in,
        } for k, v in policy.streets.items()},
    }))

    print(f"\nUniform random smoke test: {args.iterations} iterations")
    summaries = []
    for idx in range(args.iterations):
        smoke = simulate_uniform_hand_to_showdown(spec, verbose=args.verbose, reducer=reducer)
        summaries.append({
            "iteration": idx,
            "final_stacks": smoke["final_stacks"],
            "final_status": smoke["final_status"],
            "stats": smoke["stats"],
            "strategy_impact": smoke["strategy_impact"],
        })

    print(_pp(summaries))

    if args.verbose:
        print("\nVerbose traces for all iterations:")
        for idx in range(args.iterations):
            smoke = simulate_uniform_hand_to_showdown(spec, verbose=True, reducer=reducer)
            print(f"\n--- iteration {idx} ---")
            print(_pp(smoke.get("trace", [])))

    print("\nExample regret update under a uniform strategy:")
    print(_pp(example_regret_update()))


if __name__ == "__main__":
    main()
