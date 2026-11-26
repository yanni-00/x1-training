"""Unit checks for the late-swing lateral direction reward terms."""

import importlib.util
from pathlib import Path

import torch

try:  # Full package import works inside the training image.
    from humanoid.envs.f1.f1_lateral_swing_direction_metric import (
        swing_direction_step,
        update_air_time,
    )
except ModuleNotFoundError:  # CPU-only check without isaacgym: load the pure module.
    metric_path = (Path(__file__).resolve().parents[2]
                   / 'humanoid/envs/f1/f1_lateral_swing_direction_metric.py')
    spec = importlib.util.spec_from_file_location('f1_swing_metric', metric_path)
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    swing_direction_step = metric.swing_direction_step
    update_air_time = metric.update_air_time

DT = 0.01
V_REF = 0.4
RAMP_S = 0.25


def run():
    # P1: wrong-direction swing velocity earns nothing inside the term clip.
    swing = torch.ones(1, 2)
    air = torch.full((1, 2), 0.1)
    gate = torch.tensor([True])
    r = swing_direction_step(swing, torch.full((1, 2), -0.3), air, gate, V_REF, RAMP_S)
    assert torch.allclose(r, torch.zeros(1)), 'wrong-direction velocity must be clipped to zero'

    # P2: correct-direction velocity earns a positive, ramp-scaled reward.
    r = swing_direction_step(swing, torch.full((1, 2), 0.4), air, gate, V_REF, RAMP_S)
    expected = 2 * (0.1 / RAMP_S) ** 2 * 1.0
    assert abs(float(r) - expected) < 1e-6, 'correct-direction reward mismatch'

    # P3: twice the swing time pays four times the ramp term.
    air2 = torch.full((1, 2), 0.2)
    r2 = swing_direction_step(swing, torch.full((1, 2), 0.4), air2, gate, V_REF, RAMP_S)
    assert abs(float(r2) - 4 * expected) < 1e-5, 'ramp must be quadratic in air time'

    # P4: non-lateral and standing commands both gate the term to zero.
    r = swing_direction_step(swing, torch.full((1, 2), 0.4), air, torch.tensor([False]), V_REF, RAMP_S)
    assert torch.allclose(r, torch.zeros(1)), 'gate must zero the reward'

    # P5: velocity is clamped at v_ref and cannot exceed the per-step budget.
    r_fast = swing_direction_step(swing, torch.full((1, 2), 0.8), torch.full((1, 2), 1.0),
                                  gate, V_REF, RAMP_S)
    r_cap = 2 * 1.0
    assert abs(float(r_fast) - r_cap) < 1e-6, 'progress must clamp at one'

    # P6: air time accrues while airborne and resets on contact.
    a1 = update_air_time(torch.zeros(1, 2), torch.ones(1, 2), DT)
    assert torch.allclose(a1, torch.full((1, 2), DT)), 'air time must accrue by dt'
    a2 = update_air_time(a1, torch.tensor([[1.0, 0.0]]), DT)
    assert torch.allclose(a2, torch.tensor([[2 * DT, 0.0]])), 'contact must reset air time'

    print('P1-P6 pass: late-swing lateral direction reward checks are green')


if __name__ == '__main__':
    run()