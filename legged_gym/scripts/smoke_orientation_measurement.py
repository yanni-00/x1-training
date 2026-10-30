"""Verify world-body measurement, unchanged policy/control, and real PPO updates."""
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import isaacgym
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.isaacgym_utils import get_euler_xyz

args=get_args()
cfg,tc=task_registry.get_cfgs(args.task)
base,bt=task_registry.get_cfgs('g1r39')
before,after=class_to_dict(base),class_to_dict(cfg)
assert after['rewards'].pop('orientation_control_include_pelvis') is True
assert before==after,'Unexpected configuration outside orientation measurement'
ta,tb=class_to_dict(tc),class_to_dict(bt)
ta['runner']['experiment_name']=tb['runner']['experiment_name']
assert ta==tb,'Unexpected input/optimizer/PPO/seed/network change'
cls=task_registry.get_task_class(args.task);parent=task_registry.get_task_class('g1r39')
for name in ('step','calculate_action','reset_idx','check_termination','_compute_torques',
             'compute_observations','_arm_rest_error','_reward_waist_posture_excess'):
    assert getattr(cls,name) is getattr(parent,name),name+' changed'
path=Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='5b753bde8247ce500fb848c0562975a5698ff6d0f4605b78aa8b4cd5de0dd730'
saved=torch.load(str(path),map_location='cpu')
assert saved['iter']==12000 and 'optimizer_state_dict' in saved
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
assert not env.use_disturb and env.num_actions==23
assert env.body_names[0]=='pelvis'
assert env.body_names[env.orientation_body_index]=='torso_link'
assert env.curriculum_scale==base.rewards.curriculum_init
assert set(env.reward_scales)=={k for k,v in class_to_dict(base.rewards.scales).items() if v!=0}
env.step(torch.zeros(env.num_envs,env.num_actions,device=env.device))
assert torch.allclose((env.root_states[:,3:7]*env.rigid_body_states[:,0,3:7]).sum(dim=1).abs(),
    torch.ones(env.num_envs,device=env.device),atol=1e-5),'Root is not the measured pelvis frame'
pitch_command=env.commands[:,env.command_body_pitch_dim]
errors=[]
for quat in (env.root_states[:,3:7],env.rigid_body_states[:,env.orientation_body_index,3:7]):
    rpy=get_euler_xyz(quat)
    errors.append((torch.sin(rpy[:,1])-torch.sin(pitch_command)).square()
        +(torch.sin(rpy[:,0])*torch.cos(rpy[:,1])).square())
expected=torch.maximum(errors[0],errors[1])
actual=env._reward_orientation_control()
assert torch.allclose(actual,expected,atol=1e-5),'Quaternion measurement disagrees with independent RPY projection'
assert torch.all(actual>=parent._reward_orientation_control(env)-1e-6)
env.cfg.rewards.orientation_control_include_pelvis=False
assert torch.equal(env._reward_orientation_control(),parent._reward_orientation_control(env))
env.cfg.rewards.orientation_control_include_pelvis=True
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_orientation_measurement').resolve()))
assert runner.current_learning_iteration==12000
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==12010
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_ORIENTATION_MEASUREMENT_OK '+json.dumps({'task':args.task,'resume_iter':12000,'final_iter':12010,
    'root_body':'pelvis','orientation_body':'torso_link','independent_RPY_projection_verified':True,
    'measurement_disabled_reduces_exactly_to_R39':True,'finite':True,'all23_actions_learned':True,
    'same_PD_arm_reference_waist_band_reward_components_scales_commands_initialization_collision_termination_PPO':True}))
