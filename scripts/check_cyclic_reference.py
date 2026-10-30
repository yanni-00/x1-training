"""Analytic phase/stop checks using actual audited candidate support phases."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT/'source/whole_body_tracking/whole_body_tracking/utils/cyclic_reference_clock.py'
spec = importlib.util.spec_from_file_location('cyclic_clock', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--asset', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
asset = json.loads(a.asset.read_text())
assert hashlib.sha256((a.asset.parent/asset['csv']).read_bytes()).hexdigest() == asset['csv_sha256']
fps, period, supports = asset['input_fps'], asset['cycle_intervals'], asset['mujoco_support_frames']
dt = 1/fps


def clock():
    result = module.CyclicSupportPhaseClock(1, fps, period, supports, acceleration=2.0)
    result.reset([0], [0])
    return result


checks = {}
c = clock()
initial = c.unwrapped_phase.clone()
previous = initial.clone()
for _ in range(10000):  # analytic coverage fixture, not measured physical walking
    c.step(torch.tensor([1.0]), dt)
    assert (c.unwrapped_phase >= previous).all()
    assert 0 <= float(c.phase[0]) < period
    previous = c.unwrapped_phase.clone()
expected = initial+fps*(10000*dt-1/(2*c.acceleration))
torch.testing.assert_close(c.unwrapped_phase, expected, rtol=1e-6, atol=1e-4)
assert int(c.cycles[0]) > 1 and float(c.rate[0]) == 1.0
checks['long_go_repeats_reference_without_exhaustion_or_phase_reset'] = True
for _ in range(50):
    c.step(torch.tensor([0.5]), dt)
assert float(c.rate[0]) == 0.5
checks['nonzero_speed_change_preserves_advancing_phase'] = True
for _ in range(200):
    c.step(torch.tensor([0.0]), dt)
assert float(c.rate[0]) == 0 and float(c.phase[0]) in supports
held = c.unwrapped_phase.clone()
for _ in range(100):
    c.step(torch.tensor([0.0]), dt)
assert torch.equal(c.unwrapped_phase, held)
c.step(torch.tensor([1.0]), dt)
assert (c.unwrapped_phase > held).all()
checks['stop_on_support_hold_and_restart_without_reset'] = True
c = clock()
c.reset([0], [len(supports)-1])
c.rate[:] = 1
c.was_go[:] = True
before = c.unwrapped_phase.clone()
for _ in range(200):
    c.step(torch.tensor([0.0]), dt)
assert int(c.cycles[0]) >= 1 and float(c.phase[0]) in supports
assert (c.unwrapped_phase > before).all() and float(c.rate[0]) == 0
checks['braking_can_cross_reference_seam_without_teleport'] = True
try:
    module.CyclicSupportPhaseClock(1, fps, period, supports[:1])
    raise AssertionError('Insufficient support candidates accepted')
except ValueError:
    pass
checks['insufficient_support_rejected'] = True
report = {'scope': 'Reference clock contract only, not gait dynamics or policy acceptance',
          'asset_sha256': asset['csv_sha256'], 'module_sha256': hashlib.sha256(MODULE.read_bytes()).hexdigest(),
          'checks': checks, 'all_passed': all(checks.values()), 'behavior_acceptance': False,
          'native_fk_recomputed': False}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
