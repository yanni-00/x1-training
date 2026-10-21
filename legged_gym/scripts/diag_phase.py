import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymutil  # noqa: F401
from legged_gym.envs import *  # noqa: F401
from legged_gym.utils import get_args, task_registry  # noqa: F401
import torch

# lap10 崩溃诊断: play_diag 8步即终止 vs eval_cmd 89.5% 存活 —— 分相测试
# 阶段1 0-2s: cmd=0 纯站立(姿态检查); 阶段2 2s+: cmd_vx=0.5(行走)
# 输出每帧记录, 终止时打印末6帧定位死因(高度/俯仰/速度轨迹)
import argparse
parser = argparse.ArgumentParser(add_help=False)
args, _ = parser.parse_known_args()
args = get_args()

env_cfg, _ = task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs = 1
env_cfg.env.episode_length_s = 100000
env_cfg.terrain.curriculum = False
env_cfg.noise.add_noise = False
env_cfg.domain_rand.randomize_friction = False
env_cfg.domain_rand.randomize_load = False
env_cfg.domain_rand.randomize_gains = False
env_cfg.domain_rand.randomize_link_props = False
env_cfg.domain_rand.randomize_base_mass = False
env_cfg.commands.resampling_time = 100000
env_cfg.rewards.penalize_curriculum = False
env_cfg.terrain.mesh_type = 'trimesh'
env_cfg.terrain.num_rows = 1
env_cfg.terrain.num_cols = 1
env_cfg.terrain.max_init_terrain_level = 1
env_cfg.terrain.selected = True
env_cfg.terrain.selected_terrain_type = "random_uniform"
env_cfg.terrain.terrain_kwargs = {
    "random_uniform": {"min_height": -0.00, "max_height": 0.00,
                       "step": 0.005, "downsampled_scale": 0.2},
}

env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
_, train_cfg = task_registry.get_cfgs(name=args.task)
train_cfg.runner.resume = True
train_cfg.runner.resume_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                              '..', '..', 'resources', 'checkpoints', 'model_11550_g1int.pt'))
ppo_runner, train_cfg = task_registry.make_alg_runner(
    env=env, name=args.task, args=args, train_cfg=train_cfg)
ac = ppo_runner.get_inference_policy(device=env.device)
print("load_path:", train_cfg.runner.resume_path)

obs, critic_obs = env.reset()
log = []
import sys
WARMUP = int(os.environ.get('DIAG_WARMUP', '100'))  # 0=纯 play_diag 行为(无站立相)
for t in range(1000):
    with torch.inference_mode():
        actions, _ = ac.act_inference(obs, privileged_obs=critic_obs)
    obs, critic_obs, _, dones, infos = env.step(actions)
    if t < WARMUP:
        env.commands[:, 0] = 0.0
    else:
        env.commands[:, 0] = 0.5
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    log.append((t, env.base_lin_vel[0, 0].item(), env.root_states[0, 2].item(),
                env.projected_gravity[0, 0].item(), env.projected_gravity[0, 1].item(),
                dones[0].item()))
    if dones[0] and not env.time_out_buf[0]:
        print(f"TERMINATED at t={t}; last 6 frames:")
        for r in log[-6:]:
            print(f"  t={r[0]} vx={r[1]:.2f} h={r[2]:.3f} gx={r[3]:.3f} gy={r[4]:.3f} done={r[5]}")
        break
else:
    st = log[200:]
    print("survived 1000 steps")
    print(f"stand(0-2s): gx={sum(r[3] for r in log[:100])/100:.3f} h={sum(r[2] for r in log[:100])/100:.3f}")
    print(f"steady(>4s): vx={sum(r[1] for r in st)/len(st):.3f} h={sum(r[2] for r in st)/len(st):.3f} "
          f"gx={sum(r[3] for r in st)/len(st):.3f}")
