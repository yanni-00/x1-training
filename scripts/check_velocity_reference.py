"""Analytic command-contract checks; fixtures are not robot performance claims."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT/'source/whole_body_tracking/whole_body_tracking/utils/velocity_reference.py'
spec = importlib.util.spec_from_file_location('velocity_reference', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
checks = {}


def planner(yaw=0.0, limits=((-2, 2), (-2, 2), (-2, 2))):
    result = module.VelocityReferenceRoot(1, limits)
    result.reset([0], [[0.0, 0.0]], [yaw])
    return result


root = planner(0.7)
root.request([[0.6, 0, 0]], [True])
for _ in range(500):
    root.step(0.02)
expected = torch.tensor([[6*math.cos(0.7), 6*math.sin(0.7)]])
torch.testing.assert_close(root.position, expected, rtol=1e-4, atol=3e-5)
assert float(root.yaw[0]) == float(torch.tensor(0.7))
checks['fixed_heading_straight_command'] = True
for sign in [1, -1]:
    root = planner()
    root.request([[1, 0, sign*math.pi/2]], [True])
    for _ in range(50):
        root.step(0.02)
    torch.testing.assert_close(root.position, torch.tensor([[2/math.pi, sign*2/math.pi]]),
                               rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(root.yaw, torch.tensor([sign*math.pi/2]), rtol=1e-4, atol=3e-5)
checks['positive_and_negative_commanded_quarter_turns'] = True
root = planner()
root.request([[0, 0.5, 0]], [True])
root.step(1.0)
torch.testing.assert_close(root.position, torch.tensor([[0.0, 0.5]]))
checks['lateral_reference_axis'] = True
position, yaw = root.position.clone(), root.yaw.clone()
root.request([[0, 0, 0]], [True])
for _ in range(100):
    root.step(0.02)
assert torch.equal(position, root.position) and torch.equal(yaw, root.yaw)
root.request([[1, 1, 1]], [False])
root.step(0.02)
assert torch.equal(position, root.position) and torch.equal(yaw, root.yaw)
checks['centered_stick_and_disabled_mode_hold_reference'] = True
root.request([[0.5, 0, 0]], [True])
assert torch.equal(position, root.position)
root.step(0.02)
torch.testing.assert_close(root.position-position, torch.tensor([[0.01, 0.0]]))
checks['restart_without_reference_reset'] = True
axes = module.joystick_axes(torch.tensor([0.4]), torch.tensor([0.2]), torch.tensor([0.3]))
torch.testing.assert_close(axes, torch.tensor([[0.4, -0.2, -0.3]]))
checks['existing_joystick_axis_signs'] = True
restricted = planner(limits=((0, 1), (0, 0), (-1, 1)))
try:
    restricted.request([[0.5, 0.1, 0]], [True])
    raise AssertionError('Unsupported lateral command was silently accepted')
except ValueError:
    pass
checks['unsupported_axis_rejected'] = True
for dt in [0, float('nan')]:
    try:
        root.step(dt)
        raise AssertionError('Invalid dt accepted')
    except ValueError:
        pass
try:
    root.request([[float('nan'), 0, 0]], [True])
    raise AssertionError('Non-finite command accepted')
except ValueError:
    pass
checks['invalid_command_and_dt_rejected'] = True
report = {'scope': 'Reference-coordinate command contract only, no actual robot or gait claims',
          'fixture_kind': 'Analytic test choices; not measured hardware range or performance',
          'module_sha256': hashlib.sha256(MODULE.read_bytes()).hexdigest(), 'checks': checks,
          'all_passed': all(checks.values()), 'behavior_acceptance': False,
          'remaining': ['Continuous human gait asset and support-aware command transitions',
                        'Native command-conditioned PPO and actual command-tracking rollout',
                        'Joystick deployment/estimator/limiter offline integration']}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report))
