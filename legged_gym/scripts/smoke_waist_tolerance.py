"""Check one tolerance leaf and ten real PPO updates; no altered actions."""
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import isaacgym
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

args=get_args()
cfg,tc=task_registry.get_cfgs(args.task)
base,bt=task_registry.get_cfgs('g1r39')
before,after=class_to_dict(base),class_to_dict(cfg)
old=before['rewards']['waist_posture_tolerance_rad'];new=after['rewards']['waist_posture_tolerance_rad']
assert new==old/2
after['rewards']['waist_posture_tolerance_rad']=old
assert before==after,'Unexpected field outside waist tolerance'
ta,tb=class_to_dict(tc),class_to_dict(bt)
ta['runner']['experiment_name']=tb['runner']['experiment_name']
assert ta==tb,'Unexpected input/optimizer/PPO/seed/network change'
assert task_registry.get_task_class(args.task) is task_registry.get_task_class('g1r39')
path=Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='5b753bde8247ce500fb848c0562975a5698ff6d0f4605b78aa8b4cd5de0dd730'
saved=torch.load(str(path),map_location='cpu')
assert saved['iter']==12000 and 'optimizer_state_dict' in saved
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
assert not env.use_disturb and env.num_actions==23
assert env.curriculum_scale==base.rewards.curriculum_init
assert set(env.reward_scales)=={k for k,v in class_to_dict(base.rewards.scales).items() if v!=0}
angles=env.dof_pos[:,env.waist_posture_inds]-env.default_dof_pos[:,env.waist_posture_inds]
expected=(angles.abs()-new).clamp(min=0).square().sum(dim=1)
assert torch.allclose(env._reward_waist_posture_excess(),expected)
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_waist_tolerance').resolve()))
assert runner.current_learning_iteration==12000
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==12010
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_WAIST_TOLERANCE_OK '+json.dumps({'task':args.task,'before_tolerance_rad':old,
    'after_tolerance_rad':new,'resume_iter':12000,'final_iter':12010,'finite':True,
    'learned_all23_joint_actions':True,'no_action_override':True,
    'same_robot_class_PD_reward_formula_scales_commands_initialization_collision_termination_PPO':True}))
