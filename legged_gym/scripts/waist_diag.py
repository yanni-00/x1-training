import numpy as np
import os
import sys
# [OMA g1] 同 train.py：按脚本位置回溯两级到 x1-training/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401
from legged_gym.utils import get_args, task_registry  # noqa: F401
import torch

# [lap11b exp-20261021-001] 腰关节直测：直腰门 |waist_pitch|≤8° 需要腰三关节实际角度，
# play_diag CSV(9列) 只有 base_h/grav_x——骨盆垂直≠腰直(r18 实测: 骨盆 0.036 但
# waist_pitch+29.8°/yaw+45.8°)。本脚本 = play_diag 条件 + 腰三关节逐帧记录。
# 阶段: 0-2s cmd=0 纯站立(直腰能力) → 2s+ cmd_vx=0.5(行走中腰姿态)。
CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                              '..', '..', 'resources', 'checkpoints', 'model_7800_g1int.pt'))
OUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__),
                            '..', '..', 'resources', 'waist_diag_7800.csv'))
CMD_VX = 0.5
STAND_STEPS = 100  # 0-2s
EPISODE_STEPS = 1000


def waist_diag(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    # play_diag.py 完全一致的评估条件
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 100000
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
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
    env_cfg.terrain.selected_terrain_type = 'random_uniform'
    env_cfg.terrain.terrain_kwargs = {
        'random_uniform': {'min_height': -0.00, 'max_height': 0.00,
                           'step': 0.005, 'downsampled_scale': 0.2},
    }

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.resume = True
    train_cfg.runner.resume_path = CHECKPOINT
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    ac = ppo_runner.get_inference_policy(device=env.device)
    print('load_path:', CHECKPOINT)

    wi = env.torso_inds  # 腰三关节索引（g1.py 启动打印 waist_inds）
    dof_names = [env.dof_names[i] for i in wi] if hasattr(env, 'dof_names') else wi
    print('waist joints:', list(zip(wi, dof_names)))
    wi = torch.tensor(wi, device=env.device, dtype=torch.long)

    obs, critic_obs = env.reset()
    rows = []
    for t in range(EPISODE_STEPS):
        with torch.inference_mode():
            actions, _ = ac.act_inference(obs, privileged_obs=critic_obs)
        obs, critic_obs, _, dones, infos = env.step(actions)
        if t < STAND_STEPS:
            env.commands[:, 0] = 0.0
        else:
            env.commands[:, 0] = CMD_VX
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        w = env.dof_pos[0, wi].cpu().numpy()  # 腰三关节 (rad)
        rows.append([
            t, env.commands[0, 0].item(),
            env.base_lin_vel[0, 0].item(), env.root_states[0, 2].item(),
            env.projected_gravity[0, 0].item(),
            float(w[0]), float(w[1]), float(w[2]),
        ])
        if dones[0] and not env.time_out_buf[0]:
            print(f'TERMINATED at step {t} (non-timeout)')
            break

    arr = np.array(rows)
    np.savetxt(OUT_CSV, arr, delimiter=',',
               header='step,cmd_vx,vel_x,base_h,grav_x,waist0,waist1,waist2',
               comments='')

    def seg(a, lo, hi):
        return a[a[:, 0] >= lo][: a[:, 0] < hi] if len(a) else a

    stand = arr[arr[:, 0] < STAND_STEPS]
    steady = arr[arr[:, 0] >= 200]
    print('=' * 60)
    print(f'WAIST DIAG  cmd_vx={CMD_VX}  steps={len(arr)}')
    for name, s in [('stand(0-2s)', stand), ('steady(>4s)', steady)]:
        if len(s) == 0:
            print(f'{name}: no data')
            continue
        w_deg = s[:, 5:8] * 180.0 / np.pi
        print(f'{name}: vx={s[:, 2].mean():+.3f} base_h={s[:, 3].mean():.3f} grav_x={s[:, 4].mean():+.3f}')
        print(f'          waist0(yaw?)={w_deg[:, 0].mean():+.1f}°  waist1(roll?)={w_deg[:, 1].mean():+.1f}°  '
              f'waist2(pitch?)={w_deg[:, 2].mean():+.1f}°  (deg, mean)')
    print(f'CSV -> {OUT_CSV}')
    print('=' * 60)


if __name__ == '__main__':
    args = get_args()
    waist_diag(args)
