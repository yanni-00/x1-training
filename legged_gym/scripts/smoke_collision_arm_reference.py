"""Check reference-only mapping, unchanged safety/offsets, and actual PPO."""
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import isaacgym  # noqa: F401
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.g1.g1r32 import G1R32Robot
from legged_gym.envs.g1.g1r36 import G1R36Robot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

args=get_args();cfg,tc=task_registry.get_cfgs(args.task)
base,bt=task_registry.get_cfgs('g1r35')
before,after=class_to_dict(base),class_to_dict(cfg)
assert cfg.rewards.arm_rest_shoulder_roll_rad==.26
del after['rewards']['arm_rest_shoulder_roll_rad']
assert before==after,'Unexpected weights/commands/PD/initialization/collision/termination/observation change'
tb,ta=class_to_dict(bt),class_to_dict(tc)
ta['runner']['experiment_name']=tb['runner']['experiment_name']
assert tb==ta,'Unexpected input/seed/PPO/network/update change'
for name in ('step','_compute_torques','_reset_dofs','_resample_commands',
             '_reward_shoulder_deviation','_reward_standing_joint_deviation','_reward_orientation_control'):
    assert getattr(G1R36Robot,name) is getattr(G1R32Robot,name),name
path=Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='5b753bde8247ce500fb848c0562975a5698ff6d0f4605b78aa8b4cd5de0dd730'
saved=torch.load(str(path),map_location='cpu')
assert saved['iter']==12000 and 'optimizer_state_dict' in saved
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
assert env.curriculum_scale==base.rewards.curriculum_init
# Reset builds the foot-phase state before directly inspecting pose errors.
env.reset()
original=G1R32Robot._arm_rest_error(env)
expected=original.clone()
expected[:,env.arm_rest_shoulder_roll_inds]-=env.arm_rest_shoulder_roll_reference
assert torch.allclose(env._arm_rest_error(),expected)
assert torch.allclose(env.default_dof_pos,
    torch.tensor([[base.init_state.default_joint_angles[n] for n in env.dof_names]],device=env.device))
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_collision_arm_reference').resolve()))
assert runner.current_learning_iteration==12000
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==12010
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_COLLISION_ARM_REFERENCE_OK '+json.dumps({'task':args.task,
    'changed_environment_fields':['rewards.arm_rest_shoulder_roll_rad'],
    'shoulder_roll_reference_rad':env.arm_rest_shoulder_roll_reference.cpu().tolist(),
    'same_R35_input_optimizer_weights_commands_PPO_PD_observations_offsets_initialization_collision_termination':True,
    'reference_error_mapping_verified':True,'no_direct_action_or_reset_override':True,
    'resume_iter':12000,'final_iter':12010,'finite':True,'new_reward_terms':False}))
