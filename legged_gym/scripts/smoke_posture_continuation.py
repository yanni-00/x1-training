"""Check actual R28 state restoration and ten PPO updates from accepted R27."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym  # noqa: F401, before torch
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

args = get_args()
cfg, tc = task_registry.get_cfgs(args.task)
base, bt = task_registry.get_cfgs('g1r27')
before, after = class_to_dict(base), class_to_dict(cfg)
assert after['rewards']['curriculum_init'] == 0.9714232100731219
after['rewards']['curriculum_init'] = before['rewards']['curriculum_init']
assert before == after, 'Environment changed beyond reconstructed curriculum state'
t_before, t_after = class_to_dict(bt), class_to_dict(tc)
for key in ('experiment_name', 'resume_path'):
    t_after['runner'][key] = t_before['runner'][key]
assert t_before == t_after, 'Unexpected seed/PPO/network/duration change'
path = Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest() == '17b4a0a3dbbf9844841f00a7e8d4c8030911c753d68043b8aa8ea89a2cd75985'
saved = torch.load(str(path), map_location='cpu')
assert saved['iter'] == 11400 and 'optimizer_state_dict' in saved
cfg.env.num_envs = 16
env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
initial_scale = env.curriculum_scale
assert initial_scale == cfg.rewards.curriculum_init and env.learning_iter == 0
for _ in range(100):
    env.training_curriculum()
assert env.curriculum_scale == pow(initial_scale, cfg.rewards.penalize_curriculum_sigma)
env.learning_iter = 0
env.curriculum_scale = initial_scale
runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=tc,
    log_root=str(Path('resources/smoke_posture_continuation').resolve()))
assert runner.current_learning_iteration == 11400
assert env.curriculum_scale == initial_scale
runner.learn(num_learning_iterations=10, init_at_random_ep_len=True)
assert runner.current_learning_iteration == 11410 and env.curriculum_scale == initial_scale
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_POSTURE_CONTINUATION_OK ' + json.dumps({
    'task':args.task, 'input_iter':11400, 'final_iter':11410,
    'curriculum_init':initial_scale, 'finite':True,
    'unchanged_objective_PD_commands_PPO_network_seed_duration':True,
    'only_restored_environment_field':'rewards.curriculum_init',
    'full_environment_state_restored':False}))
