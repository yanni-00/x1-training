"""Offline checks for the touchdown-settled lateral landing metric (no simulator required).

Run: python humanoid/tests/check_lateral_landing_reward.py
"""

import importlib.util
import math
from pathlib import Path

import torch

MODULE = Path(__file__).resolve().parents[1] / 'envs' / 'f1' / 'f1_lateral_landing_metric.py'
_spec = importlib.util.spec_from_file_location('f1_lateral_landing_metric', MODULE)
metric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metric)

CYCLE = 0.7
PARAMS = dict(touchdown_n=40.0, liftoff_n=5.0, min_air_steps=3, max_gap_steps=1,
              stand_com_threshold=0.05, cycle_time=CYCLE, command_scale=0.4)
BASE = (0.156, -0.156)


def new_state(num_envs=1):
    return metric.new_landing_state(num_envs, torch.device('cpu'))


def step(state, fz, xy, yaw=0.0, cmd_x=0.0, cmd_y=0.4):
    """One metric step; fz is (N,2), xy is (N,2,2), yaw is (N,)."""
    def vec(value):
        tensor = torch.tensor(value, dtype=torch.float32)
        return tensor.repeat(len(fz)) if tensor.dim() == 0 else tensor
    return metric.landing_step(
        state,
        torch.tensor(xy, dtype=torch.float32),
        torch.tensor(fz, dtype=torch.float32),
        vec(yaw),
        vec(cmd_x),
        vec(cmd_y),
        **PARAMS)


def pair(left_y=BASE[0], right_y=BASE[1], left_x=0.0):
    return [[[left_x, left_y], [0.0, right_y]]]


def fresh():
    state = new_state()
    step(state, [[300.0, 300.0]], pair())
    return state


def swing_to(state, start_y, end_y, cmd_y=0.4, airborne=5, mid_gap=0, air_after_gap=0,
             yaw=0.0, start_x=0.0, end_x=0.0):
    """Lift the left foot from start_y and land it at end_y."""
    step(state, [[300.0, 300.0]], pair(start_y, left_x=start_x), yaw=yaw, cmd_y=cmd_y)
    for _ in range(airborne):
        step(state, [[0.0, 300.0]], pair(start_y, left_x=start_x), yaw=yaw, cmd_y=cmd_y)
    for _ in range(mid_gap):
        step(state, [[20.0, 300.0]], pair(start_y, left_x=start_x), yaw=yaw, cmd_y=cmd_y)
    for _ in range(air_after_gap):
        step(state, [[0.0, 300.0]], pair(start_y, left_x=start_x), yaw=yaw, cmd_y=cmd_y)
    return float(step(state, [[400.0, 300.0]], pair(end_y, left_x=end_x), yaw=yaw, cmd_y=cmd_y))


def main():
    ideal = swing_to(fresh(), BASE[0], BASE[0] + 0.28)
    print(f'ideal landing (+0.28 m): {ideal:.4f}')
    assert math.isclose(ideal, 1.0, abs_tol=1e-5)

    zero = swing_to(fresh(), BASE[0], BASE[0])
    backward = swing_to(fresh(), BASE[0], BASE[0] - 0.28)
    print(f'no advance: {zero:.4f} | backward landing: {backward:.4f}')
    assert zero == 0.0 and math.isclose(backward, -1.0, abs_tol=1e-5)

    overshoot = swing_to(fresh(), BASE[0], BASE[0] + 0.56)
    small_ok = swing_to(fresh(), BASE[0], BASE[0] + 0.07, cmd_y=0.1)
    small_over = swing_to(fresh(), BASE[0], BASE[0] + 0.28, cmd_y=0.1)
    print(f'overshoot +0.56: {overshoot:.4f} | small cmd exact +0.07: {small_ok:.4f} | '
          f'small cmd over +0.28: {small_over:.4f}')
    assert math.isclose(overshoot, 0.0, abs_tol=1e-5)
    assert math.isclose(small_ok, 0.25, abs_tol=1e-5)
    assert math.isclose(small_over, -0.5, abs_tol=1e-5)

    right = swing_to(fresh(), BASE[0], BASE[0] - 0.28, cmd_y=-0.4)
    print(f'right command, -0.28 m landing: {right:.4f}')
    assert math.isclose(right, 1.0, abs_tol=1e-5)

    state = new_state()
    step(state, [[300.0, 300.0]], pair(), cmd_x=0.4, cmd_y=0.4)
    for _ in range(5):
        step(state, [[0.0, 300.0]], pair(), cmd_x=0.4, cmd_y=0.4)
    gated = float(step(state, [[400.0, 300.0]], pair(BASE[0] + 0.28), cmd_x=0.4, cmd_y=0.4))
    print(f'omnidirectional command gate: {gated:.4f}')
    assert gated == 0.0

    short = swing_to(fresh(), BASE[0], BASE[0] + 0.28, airborne=2)
    ramp = swing_to(fresh(), BASE[0], BASE[0] + 0.28, airborne=5, mid_gap=1)
    interrupted = swing_to(fresh(), BASE[0], BASE[0] + 0.28, airborne=2, mid_gap=1, air_after_gap=3)
    print(f'short airborne: {short:.4f} | landing ramp (gap 1): {ramp:.4f} | '
          f'interrupted airborne: {interrupted:.4f}')
    assert short == 0.0 and math.isclose(ramp, 1.0, abs_tol=1e-5) and interrupted == 0.0

    no_landing = float(step(fresh(), [[400.0, 300.0]], pair(BASE[0] + 0.28)))
    print(f'touchdown without liftoff: {no_landing:.4f}')
    assert no_landing == 0.0

    yawed = swing_to(fresh(), BASE[0], BASE[0], yaw=math.pi / 2, start_x=0.0, end_x=-0.28)
    print(f'frozen 90 deg heading, world -x swing: {yawed:.4f}')
    assert math.isclose(yawed, 1.0, abs_tol=1e-5)

    # Reciprocation: +0.28 then -0.28 nets zero.
    state = fresh()
    first = swing_to(state, BASE[0], BASE[0] + 0.28)
    second = swing_to(state, BASE[0] + 0.28, BASE[0])
    print(f'reciprocating landing: {first:.4f} + {second:.4f} = {first + second:.4f}')
    assert math.isclose(first, 1.0, abs_tol=1e-5) and math.isclose(second, -1.0, abs_tol=1e-5)

    # Two forward steps then a full return still nets zero (no per-event clamp).
    state = fresh()
    a = swing_to(state, BASE[0], BASE[0] + 0.28)
    b = swing_to(state, BASE[0] + 0.28, BASE[0] + 0.56)
    c = swing_to(state, BASE[0] + 0.56, BASE[0])
    print(f'advance/advance/return: {a:.4f} + {b:.4f} + {c:.4f} = {a + b + c:.4f}')
    assert math.isclose(a + b + c, 0.0, abs_tol=1e-5)

    # Stance slip before liftoff is not credited.
    state = fresh()
    for _ in range(5):
        step(state, [[300.0, 300.0]], pair(BASE[0] + 0.20))
    slipped = swing_to(state, BASE[0] + 0.20, BASE[0] + 0.20)
    print(f'stance slip only: {slipped:.4f}')
    assert slipped == 0.0

    # Command change in the air: the swing is cancelled, the foot must land, and
    # only the following swing can score.
    state = new_state()
    step(state, [[300.0, 300.0]], pair(), cmd_x=0.4, cmd_y=0.0)
    for _ in range(5):
        step(state, [[0.0, 300.0]], pair(), cmd_x=0.4, cmd_y=0.0)
    crossed = float(step(state, [[0.0, 300.0]], pair(), cmd_x=0.0, cmd_y=0.4))
    landed = float(step(state, [[400.0, 300.0]], pair(BASE[0] + 0.28), cmd_x=0.0, cmd_y=0.4))
    retry = swing_to(state, BASE[0] + 0.28, BASE[0] + 0.56, cmd_y=0.4)
    print(f'air command switch: change {crossed:.4f}, landing {landed:.4f}, next swing {retry:.4f}')
    assert crossed == 0.0 and landed == 0.0 and math.isclose(retry, 1.0, abs_tol=1e-5)

    # Direction flip in the air behaves the same.
    state = new_state()
    step(state, [[300.0, 300.0]], pair())
    for _ in range(5):
        step(state, [[0.0, 300.0]], pair())
    flipped = float(step(state, [[0.0, 300.0]], pair(), cmd_y=-0.4))
    flip_land = float(step(state, [[400.0, 300.0]], pair(BASE[0] - 0.28), cmd_y=-0.4))
    print(f'air direction flip: change {flipped:.4f}, landing {flip_land:.4f}')
    assert flipped == 0.0 and flip_land == 0.0

    # The speed target is frozen at liftoff: a command drop mid-swing does not
    # shrink the target.
    state = fresh()
    for _ in range(5):
        step(state, [[0.0, 300.0]], pair(), cmd_y=0.4)
    step(state, [[0.0, 300.0]], pair(), cmd_y=0.1)
    frozen = float(step(state, [[400.0, 300.0]], pair(BASE[0] + 0.07), cmd_y=0.1))
    print(f'target frozen at liftoff, +0.07 m under a dropped command: {frozen:.4f}')
    assert math.isclose(frozen, 0.07 / 0.28, abs_tol=1e-5)

    # Reset re-arms and never scores on the first step.
    state = fresh()
    state['armed'][0] = False
    rearm = float(step(state, [[0.0, 300.0]], pair()))
    print(f'first step after reset: {rearm:.4f}')
    assert rearm == 0.0

    batch = new_state(4096)
    reward = step(batch, [[300.0, 300.0]] * 4096, pair() * 4096)
    print(f'4096-env initialisation: ok, reward max {float(reward.max()):.4f}')
    assert float(reward.max()) == 0.0

    partial = new_state(2)
    xy = [[[0.0, BASE[0]], [0.0, BASE[1]]], [[1.0, 1.0], [1.0, -1.0]]]
    step(partial, [[300.0, 300.0], [300.0, 300.0]], xy)
    partial['armed'][1] = False
    moved = [[[0.0, BASE[0]], [0.0, BASE[1]]], [[5.0, 2.0], [5.0, 1.0]]]
    step(partial, [[300.0, 300.0], [0.0, 300.0]], moved)
    print(f'partial reset baselines: env0 {partial["baseline_xy"][0].tolist()} '
          f'env1 {partial["baseline_xy"][1].tolist()}')
    assert torch.allclose(partial['baseline_xy'][0], torch.tensor([[0.0, BASE[0]], [0.0, BASE[1]]]))
    assert torch.allclose(partial['baseline_xy'][1], torch.tensor([[5.0, 2.0], [5.0, 1.0]]))

    print('all lateral landing metric checks passed')


if __name__ == '__main__':
    main()