"""[OMA] Analytic shared-engine contracts; fixtures are not physical walking."""
import argparse,hashlib,importlib,json,sys,types
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
namespace=types.ModuleType('oma_reference');namespace.__path__=[str(ROOT/'source/whole_body_tracking/whole_body_tracking/utils')]
sys.modules['oma_reference']=namespace
Engine=importlib.import_module('oma_reference.velocity_gait_reference').VelocityGaitReference
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
asset=json.loads((ROOT/'data/command_gaits/selected_gait.json').read_text())
fps,n,nominal=asset['input_fps'],asset['cycle_intervals'],asset['nominal_reference_speed_mps']
# Exact analytic fixture: a sinusoidal joint and body offset, known derivatives.
phase=np.arange(n)*2*np.pi/n;omega=2*np.pi*fps/n
q=np.tile(np.sin(phase)[:,None],(1,29));dq=np.tile((omega*np.cos(phase))[:,None],(1,29))
pos=np.zeros((n,2,3));pos[:,:,2]=1;pos[:,1,0]=np.sin(phase)
v=np.zeros_like(pos);v[:,1,0]=omega*np.cos(phase)
quat=np.zeros((n,2,4));quat[:,:,0]=1
arrays={'fps':[fps],'cycle_frames':[n],'nominal_reference_speed_mps':[nominal],
        'support_frames':asset['mujoco_support_frames'],'joint_pos':q,'joint_vel':dq,
        'body_pos_w':pos,'body_quat_w':quat,'body_lin_vel_w':v,'body_ang_vel_w':np.zeros_like(pos)}
e=Engine(arrays,1,((0,nominal),(0,0),(0,0)))
e.reset([0],[len(asset['mujoco_support_frames'])-1],[[0,0]],[0])
checks={};start=e.clock.unwrapped_phase.clone();last=start.clone();travel=0.0
for command,steps in [(nominal,600),(nominal*.5,80),(0,200)]:
    e.request([[command,0,0]],[True])
    for _ in range(steps):
        previous=e.clock.unwrapped_phase.clone();e.step(1/fps)
        advance=float(e.clock.unwrapped_phase[0]-previous[0])
        travel+=nominal*advance/fps
        assert (e.clock.unwrapped_phase>=last).all();last=e.clock.unwrapped_phase.clone()
        assert abs(float(e.root.position[0,0])-travel)<2e-4
        assert float(e.root.position[0,1])==0 and float(e.root.yaw[0])==0
        assert torch.isfinite(e.body_pos_w).all() and torch.isfinite(e.body_lin_vel_w).all()
assert int(e.clock.cycles[0])>1 and float(e.clock.rate[0])==0
assert float(e.clock.phase[0]) in asset['mujoco_support_frames']
checks['cyclic_phase_and_global_root_advance_agree_through_speed_change_and_support_stop']=True
assert torch.count_nonzero(e.body_lin_vel_w)==0 and torch.count_nonzero(e.body_ang_vel_w)==0
assert torch.count_nonzero(e.joint_vel)==0
held=e.body_pos_w.clone();held_phase=e.clock.unwrapped_phase.clone()
for _ in range(100):e.step(1/fps)
assert torch.equal(e.body_pos_w,held) and torch.equal(e.clock.unwrapped_phase,held_phase)
checks['centered_command_holds_body_root_and_joint_targets']=True
e.request([[nominal,0,0]],[True]);e.step(1/fps)
assert (e.clock.unwrapped_phase>held_phase).all()
checks['restart_preserves_phase_and_global_root']=True
try:e.request([[nominal,.1,0]],[True]);raise AssertionError('Unsupported axis accepted')
except ValueError:pass
checks['forward_only_scope_rejects_unqualified_lateral_axis']=True
# Rotate the full template/reference derivative analytically by pi/2.
e.reset([0],[0],[[2,3]],[np.pi/2]);e.request([[nominal,0,0]],[True]);e.initialize_rate([0])
local=e.sample('body_pos_w')[0].numpy();world=e.body_pos_w[0].numpy()
np.testing.assert_allclose(world[:,0],2-local[:,1],atol=1e-6)
np.testing.assert_allclose(world[:,1],3+local[:,0],atol=1e-6)
np.testing.assert_allclose(e.body_lin_vel_w[0,:,0].numpy(),0,atol=1e-6)
np.testing.assert_allclose(e.body_lin_vel_w[0,:,1].numpy(),e.sample('body_lin_vel_w')[0,:,0].numpy()+nominal,atol=1e-6)
checks['template_position_and_velocity_use_same_world_heading_transform']=True
# Nonzero commanded yaw is exercised only as an analytic transform fixture.
turn=Engine(arrays,1,((0,nominal),(0,0),(-1,1)))
turn.reset([0],[0],[[0,0]],[0]);turn.request([[nominal,0,.5]],[True]);turn.initialize_rate([0])
local=turn.sample('body_pos_w')[0].numpy();expected=turn.sample('body_lin_vel_w')[0].numpy()
expected[:,0]+=nominal;expected[:,0]-=.5*local[:,1];expected[:,1]+=.5*local[:,0]
np.testing.assert_allclose(turn.body_lin_vel_w[0].numpy(),expected,atol=1e-6)
np.testing.assert_allclose(turn.body_ang_vel_w[0,:,2].numpy(),.5,atol=1e-6)
checks['commanded_yaw_derivative_includes_cross_product_and_angular_rate']=True
report={'scope':'Analytic reference-engine fixtures only; no native FK/dynamics/behavior acceptance',
        'checks':checks,'all_passed':all(checks.values()),'behavior_acceptance':False,
        'module_sha256':hashlib.sha256((ROOT/'source/whole_body_tracking/whole_body_tracking/utils/velocity_gait_reference.py').read_bytes()).hexdigest()}
Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
