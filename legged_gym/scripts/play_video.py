import numpy as np
import os
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
# 录制方式：camera sensor 离屏渲染（无需 viewer/Xvfb），原始 RGBA 帧落盘，ffmpeg 合成 mp4。
CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                              '..', '..', 'resources', 'checkpoints', 'model_4800_g1int.pt'))
OUT_MP4 = os.path.abspath(os.path.join(os.path.dirname(__file__),
                            '..', '..', 'resources', 'play_video_4800.mp4'))
FRAMES_RAW = os.path.abspath(os.path.join(os.path.dirname(__file__),
                             '..', '..', 'resources', 'play_video_4800.raw'))
CMD_VX = 0.5
EPISODE_STEPS = 1000
FRAME_EVERY = 2          # 50Hz sim → 25fps 视频
CAM_W, CAM_H = 640, 480
FFMPEG = 'ffmpeg'


def play_video(args):
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

    # 离屏相机：环绕跟随机器人（20s 仿真一整圈）
    cam_props = gymapi.CameraProperties()
    cam_props.width = CAM_W
    cam_props.height = CAM_H
    camera = env.gym.create_camera_sensor(env.envs[0], cam_props)

    obs, critic_obs = env.reset()
    if os.path.exists(FRAMES_RAW):
        os.remove(FRAMES_RAW)
    n_frames = 0
    black_frames = 0
    with open(FRAMES_RAW, 'wb') as fraw:
        for t in range(EPISODE_STEPS):
            with torch.inference_mode():
                actions, _ = actor_critic.act_inference(obs, privileged_obs=critic_obs)
            obs, critic_obs, _, dones, infos = env.step(actions)
            env.commands[:, 0] = CMD_VX
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0

            if t % FRAME_EVERY == 0:
                theta = 2 * np.pi * t / EPISODE_STEPS
                look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
                cam_pos = look_at + 2.0 * np.array(
                    [np.cos(theta), np.sin(theta), 0.3])
                env.gym.set_camera_location(
                    camera, env.envs[0],
                    gymapi.Vec3(*cam_pos), gymapi.Vec3(*look_at))
                env.gym.render_all_camera_sensors(env.envs[0])
                img = env.gym.get_camera_image(
                    env.envs[0], camera, gymapi.IMAGE_COLOR)
                img = np.asarray(img)
                if img.ndim == 1:
                    img = img.reshape(CAM_H, CAM_W, -1)
                img = np.ascontiguousarray(img[..., :4])
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                if img.max() == 0:
                    black_frames += 1
                fraw.write(img.tobytes())
                n_frames += 1

            if dones[0] and not env.time_out_buf[0]:
                print(f"TERMINATED at step {t} (non-timeout)")
                break

    # ffmpeg: RGBA raw → h264 mp4
    rc = subprocess.call([
        FFMPEG, '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgba',
        '-s', f'{CAM_W}x{CAM_H}', '-r', str(int(50 / FRAME_EVERY)),
        '-i', FRAMES_RAW, '-pix_fmt', 'yuv420p', OUT_MP4])
    if rc == 0:
        os.remove(FRAMES_RAW)
    print("=" * 60)
    print(f"PLAY VIDEO  cmd_vx={CMD_VX}  frames={n_frames}  "
          f"black_frames={black_frames}  ffmpeg_rc={rc}")
    print(f"MP4 -> {OUT_MP4}")
    if black_frames > n_frames * 0.2:
        print("WARN: >20% frames black — offscreen render issue, "
              "check EGL/driver before judging from this video")
    print("=" * 60)


if __name__ == '__main__':
    args = get_args()
    play_video(args)
