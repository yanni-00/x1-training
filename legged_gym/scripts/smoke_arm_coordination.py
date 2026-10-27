"""Exercise reference direction, neutral/masked cases and real PPO loading."""
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym  # noqa: F401
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

args = get_args()
cfg, tc = task_registry.get_cfgs(args.task)
base, bt = task_registry.get_cfgs('g1r28')
before, after = class_to_dict(base), class_to_dict(cfg)
added = {'arm_swing_pitch_amplitude','arm_swing_phase_offset','arm_swing_speed_reference','arm_swing_pitch_weight'}
assert set(after['rewards']) - set(before['rewards']) == added
for key in added:
    after['rewards'].pop(key)
assert before == after, 'Unexpected environment/config change'
t_before, t_after = class_to_dict(bt), class_to_dict(tc)
t_after['runner']['experiment_name'] = t_before['runner']['experiment_name']
assert t_before == t_after, 'Unmatched input/seed/PPO/network/length'
path = Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest() == '17b4a0a3dbbf9844841f00a7e8d4c8030911c753d68043b8aa8ea89a2cd75985'
cfg.env.num_envs = 16
env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
env.reset()  # Initialize the existing foot clocks before the reference checks.
actual_feet = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], side+'_ankle_roll_link')
               for side in ('left','right')]
assert env.feet_indices.tolist() == actual_feet, 'Foot phase side mapping changed'
q, commands = env.dof_pos.clone(), env.commands.clone()
standing, interrupted, feet = env.standing_envs_mask.clone(), env.interrupt_mask.clone(), env.foot_indices.clone()
try:
    env.dof_pos[:] = env.default_dof_pos
    env.dof_pos[:,env.arm_rest_elbow_inds] += cfg.rewards.arm_rest_elbow_rad
    env.commands[:,0] = cfg.rewards.arm_swing_speed_reference
    env.standing_envs_mask[:] = False
    env.interrupt_mask[:] = False
    left_phase = cfg.rewards.arm_swing_phase_offset/(2*math.pi)
    env.foot_indices[:,0] = left_phase
    env.foot_indices[:,1] = (left_phase+.5)%1
    error = env._arm_rest_error()
    left, right = env.swing_shoulder_pitch_inds
    amplitude = cfg.rewards.arm_swing_pitch_amplitude
    assert torch.allclose(error[:,left], torch.full_like(error[:,left],-amplitude))
    assert torch.allclose(error[:,right], torch.full_like(error[:,right],amplitude))
    # At left-foot-forward, the right arm has negative pitch: URDF FK proves
    # that this moves the hanging hand forward. Matching the reference costs0.
    env.dof_pos[:,left] = amplitude
    env.dof_pos[:,right] = -amplitude
    assert torch.allclose(env._reward_shoulder_deviation(),torch.zeros(env.num_envs,device=env.device),atol=1e-6)
    env.standing_envs_mask[:] = True
    env.dof_pos[:,env.swing_shoulder_pitch_inds] = 0
    assert torch.allclose(env._arm_rest_error()[:,env.swing_shoulder_pitch_inds],torch.zeros((env.num_envs,2),device=env.device))
    env.standing_envs_mask[:] = False
    env.commands[:,0] = 0
    assert torch.allclose(env._arm_rest_error()[:,env.swing_shoulder_pitch_inds],torch.zeros((env.num_envs,2),device=env.device))
    env.commands[:,0] = -cfg.rewards.arm_swing_speed_reference
    assert torch.all(env._arm_rest_error()[:,left]>0) and torch.all(env._arm_rest_error()[:,right]<0)
    env.interrupt_mask[:] = True
    assert torch.equal(env._reward_shoulder_deviation(),torch.zeros(env.num_envs,device=env.device))
finally:
    env.dof_pos.copy_(q); env.commands.copy_(commands)
    env.standing_envs_mask.copy_(standing); env.interrupt_mask.copy_(interrupted); env.foot_indices.copy_(feet)
runner, _ = task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_arm_coordination').resolve()))
assert runner.current_learning_iteration == 11400
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration == 11410
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_ARM_COORDINATION_OK '+json.dumps({'task':args.task,'resume_iter':11400,'final_iter':11410,
    'reference_direction_neutral_reverse_interrupt_checks':True,'feet_left_right_mapping_verified':True,
    'matched_R28_input_duration_seed_PPO_curriculum':True,'new_reward_terms':False,'finite':True}))
