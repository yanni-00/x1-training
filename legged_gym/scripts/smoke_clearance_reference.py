"""Check G1 geometry, H1 fallback semantics and real PPO continuation."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import isaacgym  # noqa: F401
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.h1.h1 import H1Robot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

args=get_args()
cfg, train_cfg=task_registry.get_cfgs(args.task)
base, base_train=task_registry.get_cfgs('g1r24')
before,after=class_to_dict(base),class_to_dict(cfg)
assert set(after['rewards'])-set(before['rewards'])=={'feet_clearance_reference_height'}
del after['rewards']['feet_clearance_reference_height']
assert before==after
for key in ('algorithm','policy'):
    assert class_to_dict(getattr(train_cfg,key))==class_to_dict(getattr(base_train,key))
checkpoint=Path(train_cfg.runner.resume_path)
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='58b8a0d0b96342388321291052b23efb89b226f22387bc47e1fad1215f56a381'
robot=ET.parse(Path(__file__).resolve().parents[2]/'resources/robots/g1/g1_29dof_anneal_23dof.urdf').getroot()
offsets=[]
for side in ('left','right'):
    for c in robot.find("link[@name='%s_ankle_roll_link']"%side).findall('collision'):
        offsets.append(-float(c.find('origin').get('xyz').split()[2])+float(c.find('geometry/sphere').get('radius')))
target=cfg.rewards.feet_clearance_reference_height
assert abs(max(offsets)-target)<1e-12
# Use actual production reward functions on controlled tensor states. Test the
# missing-field fallback against the explicit H1 reference, and G1 zero error.
dummy=SimpleNamespace(num_envs=2,commands=torch.zeros(2,11),foot_indices=torch.tensor([[.25,.25],[.75,.75]]),
    foot_pos_world=torch.zeros(2,2,3),measured_foot_scan=torch.zeros(2,2,4),desired_contact_states=torch.zeros(2,2),
    cfg=SimpleNamespace(rewards=SimpleNamespace()))
dummy.commands[:,6]=.1
dummy.foot_pos_world[0,:,2]=target
dummy.foot_pos_world[1,:,2]=target+.1
legacy=H1Robot._reward_feet_clearance_cmd_linear(dummy)
dummy.cfg.rewards.feet_clearance_reference_height=.07
assert torch.equal(legacy,H1Robot._reward_feet_clearance_cmd_linear(dummy))
dummy.cfg.rewards.feet_clearance_reference_height=target
assert torch.allclose(H1Robot._reward_feet_clearance_cmd_linear(dummy),torch.zeros(2),atol=1e-10)
dummy.desired_contact_states[:]=1
assert not H1Robot._reward_feet_clearance_cmd_linear(dummy).any()
assert not H1Robot._reward_feet_clearance_cmd_polynomial(dummy).any()
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=train_cfg,
    log_root=str(Path('resources/smoke_clearance_reference').resolve()))
initial=runner.current_learning_iteration
assert initial==10800
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==10810
assert all(torch.isfinite(value).all() for value in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_CLEARANCE_REFERENCE_OK '+json.dumps({'task':args.task,'reference_height':target,
    'geometry_verified':True,'legacy_fallback_equal':True,'g1_reference_and_contact_mask_checks':True,
    'reward_names_and_scales_equal_r24':True,'other_environment_config_equal_r24':True,
    'PPO_and_policy_equal_r24':True,'resume_iter':initial,'final_iter':runner.current_learning_iteration,'finite':True}))
