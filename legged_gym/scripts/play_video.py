import numpy as np
import os
import shutil
import subprocess
import sys
# [OMA g1] 同 train.py：按脚本位置回溯两级到 x1-training/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import isaacgym  # noqa: F401
from isaacgym import gymapi  # noqa: F401
from legged_gym.envs import *  # noqa: F401
from legged_gym.utils import get_args, task_registry  # noqa: F401
import torch

# play 视频轮（play 三纪律：必须同步产 mp4，下判定前先看行为形态）
# 评估条件与 play_diag.py 完全一致（单 env + cmd_vx=0.5 冻结 + 关噪声/DR），
# 视频即 play_diag CSV 的可视化佐证——CSV 数字与视频行为必须对得上。
# 录制方式：viewer 帧抓取（须 xvfb-run -a 运行、不加 --headless；
# camera sensor 离屏渲染在本 headless 服务器实测返回空帧 0x0，viewer 路径已验证可用）。
# 用法：xvfb-run -a python play_video.py --task=g1int
CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                              '..', '..', 'resources', 'checkpoints', 'model_14550_g1int.pt'))
OUT_MP4 = os.path.abspath(os.path.join(os.path.dirname(__file__),
                            '..', '..', 'resources', 'play_video_14550.mp4'))
FRAME_DIR = '/tmp/play_video_frames_14550'
CMD_VX = 0.5
EPISODE_STEPS = 1000
FRAME_EVERY = 2          # 50Hz sim → 25fps 视频
FFMPEG = 'ffmpeg'


def play_video(args):
    if args.headless:
        print("REFUSING: --headless gives empty camera frames on this server; "
              "run under xvfb-run without --headless")
        return
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    # 与 play_diag 完全一致的评估条件
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
    train_cfg.runner.resume_path = CHECKPOINT
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    actor_critic = ppo_runner.get_inference_policy(device=env.device)
    print("load_path:", CHECKPOINT)
    assert env.viewer is not None, "viewer 未创建（需要 xvfb-run 且不加 --headless）"

    if os.path.isdir(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR)

    vx_log = []
    n_frames = 0
    obs, critic_obs = env.reset()
    for t in range(EPISODE_STEPS):
        with torch.inference_mode():
            actions, _ = actor_critic.act_inference(obs, privileged_obs=critic_obs)
        obs, critic_obs, _, dones, infos = env.step(actions)
        env.commands[:, 0] = CMD_VX
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        vx_log.append(env.base_lin_vel[0, 0].item())

        # 环绕相机（与 play.py 同款：跟随机器人旋转一整圈）
        theta = 2 * np.pi * t / EPISODE_STEPS
        look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
        cam_pos = look_at + 2.0 * np.array([np.cos(theta), np.sin(theta), 0.3])
        env.gym.viewer_camera_look_at(
            env.viewer, env.envs[0],
            gymapi.Vec3(*cam_pos), gymapi.Vec3(*look_at))
        if t % FRAME_EVERY == 0:
            env.gym.write_viewer_image_to_file(
                env.viewer, os.path.join(FRAME_DIR, f'f_{n_frames:04d}.png'))
            n_frames += 1

        if dones[0] and not env.time_out_buf[0]:
            print(f"TERMINATED at step {t} (non-timeout)")
            break

    # ffmpeg: PNG 序列 → h264 mp4
    rc = subprocess.call([
        FFMPEG, '-y', '-loglevel', 'error',
        '-framerate', str(int(50 / FRAME_EVERY)),
        '-i', os.path.join(FRAME_DIR, 'f_%04d.png'),
        '-pix_fmt', 'yuv420p', '-crf', '23', OUT_MP4])
    if rc == 0:
        shutil.rmtree(FRAME_DIR)
    vx = np.array(vx_log)
    steady = vx[200:]
    print("=" * 60)
    print(f"PLAY VIDEO  cmd_vx={CMD_VX}  frames={n_frames}  ffmpeg_rc={rc}")
    print(f"vel_x: mean(steady>4s)={steady.mean():.3f}  std={steady.std():.3f} "
          f"(对齐 play_diag 口径)")
    print(f"MP4 -> {OUT_MP4}")
    print("=" * 60)


if __name__ == '__main__':
    args = get_args()
    play_video(args)
