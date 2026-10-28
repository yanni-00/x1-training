"""Check the frame semantics, unchanged configs and finite matched-input PPO."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import isaacgym  # noqa: F401
from isaacgym.torch_utils import quat_from_angle_axis,quat_rotate_inverse
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.g1.g1r29 import G1R29Robot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

args=get_args();cfg,tc=task_registry.get_cfgs(args.task)
base,bt=task_registry.get_cfgs('g1r30')
before,after=class_to_dict(base),class_to_dict(cfg)
assert after['rewards'].pop('orientation_control_body')=='torso_link'
assert before==after, 'Unexpected environment/scales/reference/mask change'
t_before,t_after=class_to_dict(bt),class_to_dict(tc)
t_after['runner']['experiment_name']=t_before['runner']['experiment_name']
assert t_before==t_after,'Unexpected seed/PPO/network/input/update-count change'
assert getattr(cfg.rewards,'arm_rest_elbow_weight',.25)==.25
assert hashlib.sha256(Path(tc.runner.resume_path).read_bytes()).hexdigest()=='17b4a0a3dbbf9844841f00a7e8d4c8030911c753d68043b8aa8ea89a2cd75985'
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
env.reset()
torso=env.rigid_body_states[:,env.orientation_body_index,3:7]
save_torso=torso.clone();save_gravity=env.projected_gravity.clone()
save_command=env.commands[:,env.command_body_pitch_dim].clone()
axis=torch.tensor([0,1,0],device=env.device,dtype=torch.float)
angles=torch.linspace(-.2,.2,env.num_envs,device=env.device)  # Synthetic fixture design.
q=quat_from_angle_axis(angles,axis)
identity=torch.zeros_like(q);identity[:,3]=1
tolerance=8*torch.finfo(q.dtype).eps  # Explicit float32 verification tolerance.
torso[:]=q;env.projected_gravity[:]=env.gravity_vec;env.commands[:,env.command_body_pitch_dim]=0
assert torch.allclose(env._reward_orientation_control(),torch.sin(angles)**2,atol=tolerance,rtol=tolerance)
assert torch.all(G1R29Robot._reward_orientation_control(env)==0)
env.commands[:,env.command_body_pitch_dim]=angles
assert torch.all(env._reward_orientation_control()<=tolerance)
torso[:]=identity;env.commands[:,env.command_body_pitch_dim]=0
env.projected_gravity[:]=quat_rotate_inverse(q,env.gravity_vec)
assert torch.all(env._reward_orientation_control()<=tolerance)
assert torch.all(G1R29Robot._reward_orientation_control(env)>0)
torso[:]=q
assert torch.allclose(env._reward_orientation_control(),G1R29Robot._reward_orientation_control(env),atol=tolerance,rtol=tolerance)
torso[:]=save_torso;env.projected_gravity[:]=save_gravity
env.commands[:,env.command_body_pitch_dim]=save_command
env.reset()
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_torso_orientation_frame').resolve()))
assert runner.current_learning_iteration==11400
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==11410
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_TORSO_ORIENTATION_FRAME_OK '+json.dumps({'task':args.task,'body':'torso_link',
    'body_index':env.orientation_body_index,'changed_environment_fields':['rewards.orientation_control_body'],
    'synthetic_frame_and_command_cases_passed':True,'same_frame_matches_original_component':True,
    'elbow_weight':.25,'matched_R30_input_updates_seed_PPO_curriculum':True,
    'resume_iter':11400,'final_iter':11410,'finite':True,'new_reward_terms':False}))
