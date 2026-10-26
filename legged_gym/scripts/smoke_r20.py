import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import isaacgym  # noqa: F401
import torch
from legged_gym.envs import task_registry

# r20 冒烟（GM 开跑前，r19a 启动门教训: 启动错误 5min 内拦, 但冒烟 0 元）
# 验证: g1r20 任务注册可解析 / env 可建 / waist_posture_excess 可计算非 NaN /
#       resume model_4800 可载入 / 关键 scales 生效
env_cfg, train_cfg = task_registry.get_cfgs(name='g1r20')
env_cfg.env.num_envs = 16
env, _ = task_registry.make_env(name='g1r20', env_cfg=env_cfg)
print('env created, num_envs =', env.num_envs)
print('[g1r20] waist posture inds =', getattr(env, 'waist_posture_inds', 'MISSING'))

obs, critic_obs = env.reset()
actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
obs, critic_obs, _, dones, infos = env.step(actions)
r = env._reward_waist_posture_excess()
assert torch.isfinite(r).all(), 'NaN in waist_posture_excess'
print('waist_posture_excess raw mean = %.4f (tolerance %.3f rad)' % (
    r.mean().item(), env.cfg.rewards.waist_posture_tolerance_rad))
print('scales: tracking_lin_vel=%s orientation=%s shoulder=%s torso_dev=%s waist_excess=%s termination=%s' % (
    env.cfg.rewards.scales.tracking_lin_vel, env.cfg.rewards.scales.orientation_control,
    env.cfg.rewards.scales.shoulder_deviation, env.cfg.rewards.scales.torso_deviation,
    env.cfg.rewards.scales.waist_posture_excess, env.cfg.rewards.scales.termination))
print('resume_path =', train_cfg.runner.resume_path)
assert os.path.isfile(train_cfg.runner.resume_path), 'resume checkpoint missing'
print('base_height_target =', env.cfg.rewards.base_height_target)
print('SMOKE OK')
