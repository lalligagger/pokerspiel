import random

import pyspiel
import pytest

# Importing this module registers the PokerKit-backed OpenSpiel game.
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401

if "python_pokerkit_wrapper" not in {game.short_name for game in pyspiel.registered_games()}:
    pytest.skip("PokerKit OpenSpiel wrapper not registered in this environment", allow_module_level=True)


SHORTDECK_PARAMS = {
    "variant": "NoLimitShortDeckHoldem",
    "num_players": 2,
    "blinds": "1 2",
    "stack_sizes": "200 200",
    "antes": "0 0",
    "num_streets": 4,
}

LIMIT_HOLD_EM_PARAMS = {
    "variant": "FixedLimitTexasHoldem",
    "num_players": 2,
    "blinds": "1 2",
    "stack_sizes": "200 200",
    "antes": "0 0",
    "num_streets": 4,
    "small_bet": 2,
    "big_bet": 4,
}


def run_random_rollout(label: str, params: dict) -> None:
    game = pyspiel.load_game("python_pokerkit_wrapper", params)
    state = game.new_initial_state()

    legal = state.legal_actions()
    print(f"[{label}] initial legal_actions={legal[:10]} ... total={len(legal)}")

    steps = 0
    while not state.is_terminal() and steps < 100:
        legal = state.legal_actions()
        action = random.choice(legal)
        state.apply_action(action)
        steps += 1

    print(f"[{label}] terminal={state.is_terminal()} steps={steps} returns={state.returns()}")


if __name__ == "__main__":
    print("OpenSpiel Python PokerKit wrapper smoke test")
    print("registered games:", [g.short_name for g in pyspiel.registered_games() if 'pokerkit' in g.short_name.lower()])
    run_random_rollout("shortdeck", SHORTDECK_PARAMS)
    run_random_rollout("limit_holdem", LIMIT_HOLD_EM_PARAMS)
