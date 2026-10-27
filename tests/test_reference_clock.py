"""Behavioral contracts for reference time, independent of Isaac Lab."""
import importlib.util,json
from pathlib import Path
import torch
module=Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/reference_clock.py'
spec=importlib.util.spec_from_file_location('clock',module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.SupportPhaseClock(2,50,[0,40,80,120,160,200]);c.reset(torch.tensor([0,1]),torch.tensor([0.,0.]));dt=.02
for _ in range(50):c.step(torch.tensor([False,False]),dt)
assert torch.equal(c.phase,torch.zeros(2)) and torch.equal(c.rate,torch.zeros(2)), 'STOP from standing advanced reference'
for _ in range(45):c.step(torch.tensor([True,True]),dt)
assert torch.all(c.phase>0) and torch.all(c.rate>0)
previous=c.phase.clone()
for _ in range(300):
 c.step(torch.tensor([False,True]),dt)
 assert torch.all(c.phase>=previous), 'Reference moved backwards'
 assert torch.all(c.phase-previous<=50*dt+1e-4), 'Reference teleported'
 previous=c.phase.clone()
assert c.rate[0]==0 and torch.any(c.support==c.phase[0]), 'STOP failed to hold a support candidate'
held=c.phase[0].clone()
for _ in range(50):c.step(torch.tensor([False,True]),dt)
assert c.phase[0]==held, 'Stopped reference drifted'
assert c.exhausted[1] and c.phase[1]==200 and c.rate[1]==0, 'Reference exhaustion wrapped instead of holding'
for _ in range(10):c.step(torch.tensor([True,False]),dt)
assert c.phase[0]>held and c.rate[0]>0, 'GO after STOP did not resume'
print(json.dumps({'stop_from_stand':True,'stop_at_support':True,'hold_no_drift':True,'restart_monotonic':True,'per_step_advance_bounded':True,'end_holds_without_wrap':True}))
