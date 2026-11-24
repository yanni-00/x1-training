"""Offline checks for the post-clip potential-based timing term (no simulator required).

Run: python humanoid/tests/check_lateral_landing_timing_reward.py

P5: the discounted sum of the timing terms is zero over a completed swing, a
cancelled swing, and a terminal step that ends a swing in the air; a truncation
without a terminal leaves the expected unfinished potential.
P6: the term is returned as-is (it is added after the r7 positive-only clip), so
a negative claw-back is never zeroed.
"""

import importlib.util
import math
from pathlib import Path

import torch

MODULE = Path(__file__).resolve().parents[1] / 'envs' / 'f1' / 'f1_lateral_landing_timing_metric.py'
_spec = importlib.util.spec_from_file_location('f1_lateral_landing_timing_metric', MODULE)
metric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metric)

CYCLE = 0.7
GAMMA = 0.994 ** 0.5
WEIGHT = 50.0
DT = 0.01
PARAMS = dict(touchdown_n=40.0, liftoff_n=5.0, min_air_steps=3, max_gap_steps=1,
              stand_com_threshold=0.05, cycle_time=CYCLE, command_scale=0.4)
BASE = (0.156, -0.156)


def new_state(num_envs=1):
    return metric.new_timing_state(num_envs, torch.device('cpu'))


def step(state, fz, xy, terminal=False, yaw=0.0, cmd_x=0.0, cmd_y=0.4):
    def vec(value):
        tensor = torch.tensor(value, dtype=torch.float32)
        return tensor.repeat(len(fz)) if tensor.dim() == 0 else tensor
    return float(metric.timing_step(
        state,
        torch.tensor(xy, dtype=torch.float32),
        torch.tensor(fz, dtype=torch.float32),
        vec(yaw), vec(cmd_x), vec(cmd_y), vec(terminal),
        GAMMA, WEIGHT, DT, **PARAMS))


def pair(left_y=BASE[0], right_y=BASE[1], left_x=0.0):
    return [[[left_x, left_y], [0.0, right_y]]]


def discounted(terms):
    return sum(GAMMA ** k * t for k, t in enumerate(terms))


def swing_terms(state, start_y, end_y, airborne=35, mid_gap=0, air_after_gap=0,
                drift_y=0.0, terminal_at_end=False, yaw=0.0):
    step(state, [[300.0, 300.0]], pair(start_y), yaw=yaw)
    terms = []
    y = start_y
    for k in range(airborne):
        if k:
            y += drift_y
        terms.append(step(state, [[0.0, 300.0]], pair(y), yaw=yaw))
    for _ in range(mid_gap):
        terms.append(step(state, [[20.0, 300.0]], pair(y), yaw=yaw))
    for _ in range(air_after_gap):
        terms.append(step(state, [[0.0, 300.0]], pair(y), yaw=yaw))
    terms.append(step(state, [[400.0, 300.0]], pair(end_y), terminal=terminal_at_end, yaw=yaw))
    return terms


def main():
    full = swing_terms(new_state(), BASE[0], BASE[0] + 0.28, drift_y=0.28 / 35)
    print(f'P5 full swing: discounted {discounted(full):+.6f}, sum {sum(full):+.5f}, '
          f'nonzero steps {sum(1 for t in full if t != 0.0)}')
    assert abs(discounted(full)) < 1e-4

    cancel = swing_terms(new_state(), BASE[0], BASE[0] + 0.28, airborne=2, drift_y=0.10,
                         mid_gap=1, air_after_gap=1)
    print(f'P5 cancelled swing: discounted {discounted(cancel):+.6f}')
    assert abs(discounted(cancel)) < 1e-4

    # A swing still in the air at the terminal step must claw back its potential.
    state = new_state()
    step(state, [[300.0, 300.0]], pair(BASE[0]))
    terms = []
    y = BASE[0]
    for k in range(20):
        y += 0.01
        terms.append(step(state, [[0.0, 300.0]], pair(y)))
    terms.append(step(state, [[0.0, 300.0]], pair(y), terminal=True))
    print(f'P5 terminal mid-swing: discounted {discounted(terms):+.6f}, '
          f'last term {terms[-1]:+.4f}')
    assert abs(discounted(terms)) < 1e-4

    # Truncation without a terminal keeps the unfinished potential (documented).
    state = new_state()
    step(state, [[300.0, 300.0]], pair(BASE[0]))
    trunc = []
    y = BASE[0]
    for k in range(20):
        y += 0.01
        trunc.append(step(state, [[0.0, 300.0]], pair(y)))
    residual = discounted(trunc)
    print(f'P5 truncated swing (no terminal): residual {residual:+.4f} (expected nonzero)')
    assert residual > 0.0

    # P6: the claw-back is returned negative and is not clipped inside the metric.
    state = new_state()
    step(state, [[300.0, 300.0]], pair(BASE[0]))
    y = BASE[0]
    for k in range(10):
        y += 0.02
        step(state, [[0.0, 300.0]], pair(y))
    compensation = step(state, [[0.0, 300.0]], pair(y), terminal=True)
    print(f'P6 claw-back term: {compensation:+.4f} (negative by construction)')
    assert compensation < 0.0

    batch = new_state(4096)
    xy = torch.tensor([[[0.0, BASE[0]], [0.0, BASE[1]]]], dtype=torch.float32).repeat(4096, 1, 1)
    fz = torch.full((4096, 2), 300.0)
    term = metric.timing_step(batch, xy, fz, torch.zeros(4096), torch.zeros(4096),
                              torch.full((4096,), 0.4), torch.zeros(4096, dtype=torch.bool),
                              GAMMA, WEIGHT, DT, **PARAMS)
    print(f'batch 4096 init term max {float(term.max()):.4f}')
    assert float(term.max()) == 0.0

    print('all timing metric checks passed')


if __name__ == '__main__':
    main()