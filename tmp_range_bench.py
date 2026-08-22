import time
import pyspiel
from app_solver import GAME_CONFIGS, resolve_node_specs, prepare_selected_node_probes, snapshot_probe_states, aggregate_selected_node_ranges

iterations = [100, 200, 500, 1000, 2000]
for samples in [1, 10, 100, 1000]:
    print(f"=== samples_per_node={samples} ===")
    game = pyspiel.load_game('python_pokerkit_wrapper', GAME_CONFIGS['hulh'])
    solver = pyspiel.ExternalSamplingMCCFRSolver(game)
    for i in range(1, 2001):
        solver.run_iteration()
        if i in iterations:
            t0 = time.perf_counter()
            probes = prepare_selected_node_probes(
                game,
                resolve_node_specs('hulh-preflop', ()),
                samples_per_node=samples,
                max_attempts=max(5000, samples * 50),
            )
            records = snapshot_probe_states(solver.average_policy(), probes)
            agg = aggregate_selected_node_ranges(records)
            dt = time.perf_counter() - t0
            first = agg['nodes'][0]['action_frequencies'] if agg.get('nodes') else {}
            print(f"iter={i:4d} elapsed_ms={dt * 1000:8.2f} first_to_act={first}")
