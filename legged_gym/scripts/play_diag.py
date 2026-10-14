import numpy as np
import os
import sys
# [OMA g1] 按脚本位置回溯两级到 x1-training/（与 train.py 同一教训）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401
from legged_gym.utils import get_args, task_registry  # noqa: F401
import torch

# play 诊断轮（卡§1 行为证据补充）：单 env + 固定 vx=0.5 命令，打时间序列 CSV
# ——E_cmd 数字之外，看"实际速度曲线 vs 命令"的动态（跟踪滞后/幅度衰减/方向丢失）
CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                              '..', '..', 'resources', 'checkpoints', 'model_4800_g1int.pt'))
OUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__),
                              '..', '..', 'resources', 'play_diag_4800.csv'))
CMD_VX = 0.5
EPISODE_STEPS = 1000


def play_diag(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    # play.py 先例的评估条件
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 100000
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False            # 诊断段关噪声：看能力上限
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_load = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_link_props = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.commands.resampling_time = 100000  # 命令全程冻结
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
    train_cfg.runner.resume_path = CHECKPOINT
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    actor_critic = ppo_runner.get_inference_policy(device=env.device)
    print("load_path:", CHECKPOINT)

    obs, critic_obs = env.reset()
    rows = []
    for t in range(EPISODE_STEPS):
        with torch.inference_mode():
            actions, _ = actor_critic.act_inference(obs, privileged_obs=critic_obs)
        obs, critic_obs, _, dones, infos = env.step(actions)
        # 固定前进命令（play.py 先例：直接写 commands）
        env.commands[:, 0] = CMD_VX
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        rows.append([
            t,
            env.commands[0, 0].item(), env.commands[0, 1].item(), env.commands[0, 2].item(),
            env.base_lin_vel[0, 0].item(), env.base_lin_vel[0, 1].item(),
            env.base_ang_vel[0, 2].item(),
            env.root_states[0, 2].item(),          # base height
            (env.projected_gravity[0, 0].item(), env.projected_gravity[0, 1].item())[0],
        ])
        if dones[0] and not env.time_out_buf[0]:
            print(f"TERMINATED at step {t} (non-timeout)")
            break

    arr = np.array(rows)
    np.savetxt(OUT_CSV, arr, delimiter=',',
               header='step,cmd_vx,cmd_vy,cmd_wz,vel_x,vel_y,vel_wz,base_h,grav_x', comments='')
    # 控制台摘要
    vx = arr[:, 4]
    steady = arr[200:][:, 4]  # 4s 后视为稳态段
    print("=" * 60)
    print(f"PLAY DIAG  cmd_vx={CMD_VX}  steps={len(arr)}  terminated={bool(dones[0] and not env.time_out_buf[0])}")
    print(f"vel_x: mean(all)={vx.mean():.3f}  mean(steady>4s)={steady.mean():.3f}  std={steady.std():.3f}")
    print(f"vel_y mean(steady)={arr[200:][:, 5].mean():.3f}  vel_wz mean(steady)={arr[200:][:, 6].mean():.3f}")
    print(f"base_h mean(steady)={arr[200:][:, 7].mean():.3f} m")
    print(f"steady-state cmd tracking error (vx): {abs(steady.mean() - CMD_VX):.3f}")
    print(f"CSV -> {OUT_CSV}")
    print("=" * 60)


if __name__ == '__main__':
    args = get_args()
    play_diag(args)
