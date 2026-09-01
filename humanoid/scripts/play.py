# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.
#
# Adapted from F1 exp_A09 play.py for X1-12DOF (exp_010_1):
# same video/CSV sync protocol; 12 DOF, no motion-ref / lumbar-torso metrics.


import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

from humanoid.envs import *
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from datetime import datetime

import csv


def _gait_state_label(left_on, right_on):
    if left_on and right_on:
        return "double"
    if not left_on and not right_on:
        return "flight"
    return "single"


def _draw_play_hud(img, env, robot_index, play_step, target_vel, current_vel_x, avg_vel,
                   left_force, right_force, l_on, r_on):
    img_h, img_w = img.shape[:2]
    base_x = img_w - 1200
    base_y = 55
    line_height = 48

    def draw_outlined_text(image, text, pos, color, scale=0.9):
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    draw_outlined_text(
        img, f"step={play_step:4d} | CMD:{target_vel:.2f} REAL:{current_vel_x:.2f} AVG:{avg_vel:.2f}",
        (base_x, base_y), (255, 255, 0), 1.0)
    l_color = (0, 255, 0) if l_on else (0, 0, 255)
    r_color = (0, 255, 0) if r_on else (0, 0, 255)
    draw_outlined_text(
        img, f"L-FOOT: {'ON ' if l_on else 'OFF'} ({left_force:.1f} N)",
        (base_x, base_y + line_height), l_color)
    draw_outlined_text(
        img, f"R-FOOT: {'ON ' if r_on else 'OFF'} ({right_force:.1f} N)",
        (base_x, base_y + line_height * 2), r_color)
    if l_on and r_on:
        state_text, state_color = "STATE: *** DOUBLE SUPPORT ***", (0, 255, 255)
    elif not l_on and not r_on:
        state_text, state_color = "STATE: >>> FLIGHT PHASE <<<", (255, 0, 255)
    else:
        state_text, state_color = "STATE: SINGLE SUPPORT", (200, 200, 200)
    draw_outlined_text(img, state_text, (base_x, base_y + line_height * 3), state_color, 1.0)
    phase = env._get_phase()[robot_index].item()
    # X1-12DOF: L/R hip_pitch, knee (indices 0,3 / 6,9)
    lp = env.dof_pos[robot_index, 0].item() * 57.3
    lk = env.dof_pos[robot_index, 3].item() * 57.3
    rp = env.dof_pos[robot_index, 6].item() * 57.3
    rk = env.dof_pos[robot_index, 9].item() * 57.3
    draw_outlined_text(
        img,
        f"ph={phase:.3f} | L hp/kn={lp:+.1f}/{lk:+.1f}  R hp/kn={rp:+.1f}/{rk:+.1f}",
        (base_x, base_y + line_height * 4), (180, 255, 180), 0.95)


# control_dt = sim.dt * decimation = 0.001 * 10
PLAY_DT = 0.01
VIDEO_RECORD_EVERY = 2

# v5 八段时序验收：前进→停→慢前→停→横移→停→转向→停（32s）
# 指令在 v5 训练域内（lin_vel_x [0.1,0.4] / lin_vel_y [-0.4,0.4] / ang_vel_yaw [-0.6,0.6]）：
#   forward      (0.25,0,0)  主验收点（cycle0.7×0.25→步长≈17.5cm）
#   slow_forward (0.15,0,0)  低速前向（替代 v6 倒退：v5 无负 vx 训练域）
#   lateral      (0,0.15,0)  横移
#   rotate       (0,0,0.2)   原地转（|raw_wz|>0.15 → yaw_hold 透传，不注入）
#   stop         (0,0,0)     raw 零速；obs 的 cmd_wz 仍可由 yaw_hold 注入 hold_wz（±0.25）
CMD_SCHEDULE = [
    ("forward",      8.0, ( 0.25, 0.0,  0.0)),
    ("stop",         2.0, ( 0.0,  0.0,  0.0)),
    ("slow_forward", 8.0, ( 0.15, 0.0,  0.0)),
    ("stop",         2.0, ( 0.0,  0.0,  0.0)),
    ("lateral",      4.0, ( 0.0,  0.15, 0.0)),
    ("stop",         2.0, ( 0.0,  0.0,  0.0)),
    ("rotate",       4.0, ( 0.0,  0.0,  0.2)),
    ("stop",         2.0, ( 0.0,  0.0,  0.0)),
]
SEG_TOTAL_S = sum(_d for _, _d, _ in CMD_SCHEDULE)      # 32s（8+2+8+2+4+2+4+2）
SEG_TOTAL_STEPS = int(SEG_TOTAL_S / PLAY_DT)            # 3200 步

def sched_at(t_s):
    """返回 t_s 时刻的调度段 (label, (vx,vy,wz))；超出总时长返回末段。"""
    _acc = 0.0
    for _lbl, _dur, _cmd in CMD_SCHEDULE:
        if t_s < _acc + _dur:
            return _lbl, _cmd
        _acc += _dur
    return CMD_SCHEDULE[-1][0], CMD_SCHEDULE[-1][2]

# X1-12DOF short names (dof_names order)
JOINT_SHORT_NAMES = [
    'Ll_hp', 'Ll_hr', 'Ll_hy', 'Ll_kn', 'Ll_ap', 'Ll_ar',
    'Rl_hp', 'Rl_hr', 'Rl_hy', 'Rl_kn', 'Rl_ap', 'Rl_ar',
]

DOF_SUMMARY_GROUPS = [
    ('L_leg', [0, 1, 2, 3, 4, 5], ['hip_p', 'hip_r', 'hip_y', 'knee', 'ank_p', 'ank_r']),
    ('R_leg', [6, 7, 8, 9, 10, 11], ['hip_p', 'hip_r', 'hip_y', 'knee', 'ank_p', 'ank_r']),
]


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.env.episode_length_s = 1000
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.continuous_push = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_torque = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    # play: 关闭整机 lag，避免与踝辨识 tauLPF 叠加重延迟
    env_cfg.domain_rand.add_lag = False
    env_cfg.domain_rand.add_dof_lag = False
    env_cfg.domain_rand.add_imu_lag = False
    env_cfg.noise.curriculum = False
    env_cfg.commands.heading_command = False  # 同步训练配置：关闭 heading 跟踪
    # play 手写八段调度：禁止训练 gait 边界 resample 覆盖 commands（x1 env 每步 _resample_commands）
    env_cfg.commands.gait = ["walk_omnidirectional"]
    env_cfg.commands.gait_time_range = {"walk_omnidirectional": [1.0e6, 1.0e6]}

    # --- 踝关节阶跃辨识名义值（固定点，非随机区间）---
    # pitch: coulomb0.5 viscous0.225 arm0.15; roll: coulomb0.5 viscous0 arm0.035; tauLPF=8ms
    env_cfg.domain_rand.randomize_coulomb_friction = True
    env_cfg.domain_rand.joint_coulomb_range = [0.0, 0.0]
    env_cfg.domain_rand.joint_viscous_range = [0.0, 0.0]
    env_cfg.domain_rand.ankle_pitch_joint_coulomb_range = [0.5, 0.5]
    env_cfg.domain_rand.ankle_pitch_joint_viscous_range = [0.225, 0.225]
    env_cfg.domain_rand.ankle_roll_joint_coulomb_range = [0.5, 0.5]
    env_cfg.domain_rand.ankle_roll_joint_viscous_range = [0.0, 0.0]

    # play armature 固定标称值（训练 DR 中心值），与 x1_dh_stand_config.py 一致
    env_cfg.domain_rand.randomize_joint_armature = True
    env_cfg.domain_rand.randomize_joint_armature_each_joint = True
    env_cfg.domain_rand.joint_1_armature_range = [0.208, 0.208]    # left_hip_pitch (L/R 平均)
    env_cfg.domain_rand.joint_2_armature_range = [0.025, 0.025]    # left_hip_roll (DR 中心)
    env_cfg.domain_rand.joint_3_armature_range = [0.0148, 0.0148]  # left_hip_yaw (L/R 平均)
    env_cfg.domain_rand.joint_4_armature_range = [0.2728, 0.2728]  # left_knee_pitch (L/R 平均)
    env_cfg.domain_rand.joint_5_armature_range = [0.15, 0.15]      # left_ankle_pitch
    env_cfg.domain_rand.joint_6_armature_range = [0.035, 0.035]    # left_ankle_roll
    env_cfg.domain_rand.joint_7_armature_range = [0.208, 0.208]    # right_hip_pitch (同 L)
    env_cfg.domain_rand.joint_8_armature_range = [0.025, 0.025]    # right_hip_roll (DR 中心)
    env_cfg.domain_rand.joint_9_armature_range = [0.0148, 0.0148]  # right_hip_yaw (同 L)
    env_cfg.domain_rand.joint_10_armature_range = [0.2728, 0.2728] # right_knee_pitch (同 L)
    env_cfg.domain_rand.joint_11_armature_range = [0.15, 0.15]     # right_ankle_pitch
    env_cfg.domain_rand.joint_12_armature_range = [0.035, 0.035]   # right_ankle_roll

    env_cfg.domain_rand.enable_delivery = True
    env_cfg.domain_rand.delivery_tau_d = 0.008
    env_cfg.domain_rand.delivery_joint_ids = [4, 5, 10, 11]

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # reset 后 gait_time 已生成；play 窗口内不再触发 resample
    env.gait_time[:] = 10**9
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = -1
    train_cfg.runner.checkpoint = -1
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name,
            '0_exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env_cfg.sim.dt * env_cfg.control.decimation)
    robot_index = 0
    csv_log_start = 0
    csv_log_end = SEG_TOTAL_STEPS - 1  # CSV 覆盖全部八段（gate 分析需各段完整数据）
    num_dof = env_cfg.env.num_actions  # 12

    assert num_dof == 12, f"exp_010_1 expects 12 DOF, got {num_dof}"
    assert len(env.dof_names) == 12, f"dof_names len={len(env.dof_names)}"

    print(f"[play] little_step_v5 gate (域 x[0.1,0.4], cycle 0.7, 低抬 0.02, yaw_hold)  X1-12DOF")
    print(f"[play] 八段时序 总长 {SEG_TOTAL_S}s / {SEG_TOTAL_STEPS} 步: " +
          " → ".join(f"{_l}({_d:.0f}s)" for _l, _d, _ in CMD_SCHEDULE))
    print(f"[play] heading_command={env_cfg.commands.heading_command}  "
          f"yaw_hold={env_cfg.commands.yaw_hold}  "
          f"target_feet_height={env_cfg.rewards.target_feet_height} "
          f"max={env_cfg.rewards.target_feet_height_max}")
    print("[play] yaw_hold: raw_wz=0 段 → obs cmd_wz=hold_wz(±clip); |raw_wz|>0.15 → 透传 raw（仅 rotate 段）")
    print(f"[play] csv_log steps {csv_log_start}–{csv_log_end}")
    print("[play] ankle ID plant (nominal):")
    print(f"  pitch: Fc=0.5 B=0.225 arm=0.15 | roll: Fc=0.5 B=0 arm=0.035")
    print(f"  delivery LPF: enable={env.cfg.domain_rand.enable_delivery} "
          f"tau_d={env.cfg.domain_rand.delivery_tau_d}s ids={list(env.cfg.domain_rand.delivery_joint_ids)}")
    print(f"  coulomb_on={env.cfg.domain_rand.randomize_coulomb_friction} "
          f"armature_on={env.cfg.domain_rand.randomize_joint_armature} "
          f"add_lag={env.cfg.domain_rand.add_lag}")
    print("[dof] Isaac order:")
    for i, name in enumerate(env.dof_names):
        print(f"  [{i:2d}] {name:32s} → {JOINT_SHORT_NAMES[i]}")

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        custom_save_path = "/personal/train-more"
        run_name_str = args.run_name if args.run_name is not None else "test"
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name_str}.mp4"
        video_filepath = os.path.join(custom_save_path, file_name)
        os.makedirs(custom_save_path, exist_ok=True)
        print(f"[VIDEO] Recording to: {video_filepath}")
        video = cv2.VideoWriter(video_filepath, fourcc, 50.0, (1920, 1080))

    obs = env.get_observations()
    _lbl0, _cmd0 = sched_at(0.0)
    env.commands[:, 0] = _cmd0[0]
    env.commands[:, 1] = _cmd0[1]
    env.commands[:, 2] = _cmd0[2]
    env.commands[:, 3] = 0.0
    # v3 yaw_hold：play 手写指令绕过 resample，raw 必须同步，否则注入会覆盖掉调度指令
    env.raw_yaw_cmd[:] = env.commands[:, 2]
    _prev_seg_label = None
    print(f"[play] cold start seg={_lbl0} cmd={_cmd0}")

    frame_count = 0
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    vel_sum = 0.0
    step_accum = 0

    all_dof_log = [[] for _ in range(num_dof)]  # (act_rad, des_rad, err_rad)
    body_yaw_log = []
    swing_feet_z_log = []  # (side, feet_z_m) 非接触时刻脚底离地高度，拖地风险预检

    _csv_dir = "/personal/train-more"
    os.makedirs(_csv_dir, exist_ok=True)
    _csv_path = os.path.join(_csv_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_joints.csv")
    _csv_headers = [
        'step', 'video_frame', 'time_s', 'phase',
        'base_roll', 'base_pitch', 'base_yaw',
        'cmd_x', 'cmd_y', 'cmd_wz', 'raw_wz', 'base_vel_x', 'base_vel_y', 'base_vel_yaw', 'base_pos_x', 'base_pos_y',
        'foot_fz_l', 'foot_fz_r', 'gait_state', 'cmd_scene',
    ]
    for _sn in JOINT_SHORT_NAMES:
        # X1 无 motion ref：des = PD 目标 (default + action*scale)
        _csv_headers += [f'{_sn}_des', f'{_sn}_act', f'{_sn}_err']
    _csv_file = open(_csv_path, 'w', newline='')
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow(_csv_headers)
    print(f"[CSV] 开始记录到: {_csv_path}")
    print(f"[sync] play_dt={PLAY_DT}s  video_fps=50  record_every={VIDEO_RECORD_EVERY} render ticks")
    print(f"[sync] CSV step N 时间 ≈ N×{PLAY_DT}s；step=-1 为 reset 标定")

    _video_frame_idx = 0
    _last_actions = torch.zeros(env.num_envs, num_dof, device=env.device)

    def _foot_contacts():
        _fz_l = env.contact_forces[robot_index, env.feet_indices[0].item(), 2].item()
        _fz_r = env.contact_forces[robot_index, env.feet_indices[1].item(), 2].item()
        return _fz_l, _fz_r, _fz_l > 1.0, _fz_r > 1.0

    def _joint_des_act_err(actions_t):
        """des/act/err in rad for 12 DOF."""
        out = []
        for ji in range(num_dof):
            act = env.dof_pos[robot_index, ji].item()
            des = (env.default_dof_pos[robot_index, ji]
                   + actions_t[robot_index, ji] * env.cfg.control.action_scale).item()
            out.append((des, act, act - des))
        return out

    def _build_csv_row(play_step, vf_idx, actions_t):
        _phase = env._get_phase()[robot_index].item()
        _body_roll_deg = env.base_euler_xyz[robot_index, 0].item() * 57.3
        _body_pitch_deg = env.base_euler_xyz[robot_index, 1].item() * 57.3
        _body_yaw_deg = env.base_euler_xyz[robot_index, 2].item() * 57.3
        _fz_l, _fz_r, _l_on, _r_on = _foot_contacts()
        _gait = _gait_state_label(_l_on, _r_on)
        _base_vx = env.root_states[robot_index, 7].item()
        _base_vy = env.root_states[robot_index, 8].item()
        _base_px = env.root_states[robot_index, 0].item()
        _base_py = env.root_states[robot_index, 1].item()
        _cmd_x = env.commands[robot_index, 0].item()
        _cmd_y = env.commands[robot_index, 1].item()
        _cmd_wz = env.commands[robot_index, 2].item()
        _raw_wz = env.raw_yaw_cmd[robot_index].item()
        _base_wz = env.root_states[robot_index, 11].item()  # root_states[10:13]=ang vel, z 分量
        _time_s = max(play_step, 0) * PLAY_DT
        _row = [
            play_step,
            vf_idx if vf_idx > 0 else '',
            f"{_time_s:.4f}",
            f"{_phase:.4f}",
            f"{_body_roll_deg:.3f}", f"{_body_pitch_deg:.3f}", f"{_body_yaw_deg:.3f}",
            f"{_cmd_x:.4f}", f"{_cmd_y:.4f}", f"{_cmd_wz:.4f}", f"{_raw_wz:.4f}",
            f"{_base_vx:.4f}", f"{_base_vy:.4f}", f"{_base_wz:.4f}",
            f"{_base_px:.4f}", f"{_base_py:.4f}",
            f"{_fz_l:.2f}", f"{_fz_r:.2f}", _gait,
            sched_at(max(play_step, 0) * PLAY_DT)[0],
        ]
        for des, act, err in _joint_des_act_err(actions_t):
            _row += [f"{des * 57.3:.3f}", f"{act * 57.3:.3f}", f"{err * 57.3:.3f}"]
        return _row, _phase, _body_yaw_deg

    def _log_csv_step(play_step, vf_idx, actions_t):
        _row, _phase, _byaw = _build_csv_row(play_step, vf_idx, actions_t)
        _csv_writer.writerow(_row)
        _lp = env.dof_pos[robot_index, 0].item() * 57.3
        _lk = env.dof_pos[robot_index, 3].item() * 57.3
        print(f"[S{play_step:4d}|ph={_phase:.3f}|vf={vf_idx if vf_idx > 0 else '-'}] "
              f"L_hp/kn={_lp:+.1f}/{_lk:+.1f}°  base_yaw={_byaw:+.1f}°")
        if play_step < 0:
            return _row
        for ji, (des, act, err) in enumerate(_joint_des_act_err(actions_t)):
            all_dof_log[ji].append((act, des, err))
        body_yaw_log.append((play_step, _byaw))
        return _row

    def _capture_video_frame(play_step, avg_vel):
        nonlocal _video_frame_idx, frame_count
        if not RENDER:
            return -1
        frame_count += 1
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        if frame_count % VIDEO_RECORD_EVERY != 0:
            return -1
        img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
        img = np.reshape(img, (1080, 1920, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _fz_l, _fz_r, _l_on, _r_on = _foot_contacts()
        _draw_play_hud(
            img, env, robot_index, play_step,
            env.commands[0, 0].item(), env.base_lin_vel[0, 0].item(), avg_vel,
            _fz_l, _fz_r, _l_on, _r_on)
        video.write(img[..., :3])
        _video_frame_idx += 1
        return _video_frame_idx

    # reset 标定帧（A09 同协议）
    _vf_reset = _capture_video_frame(-1, 0.0)
    _log_csv_step(-1, vf_idx=_vf_reset, actions_t=_last_actions)
    if _vf_reset > 0:
        print(f"[sync] reset 标定帧 → video_frame={_vf_reset}  step=-1")

    for i in range(SEG_TOTAL_STEPS):  # 八段时序全程（0.01s/步），CSV 同窗口全覆盖
        actions = policy(obs.detach())
        _last_actions = actions.detach()

        # v6 八段调度：按运行时间取当前段指令；段切换打印一次
        _seg_label, _seg_cmd = sched_at(i * PLAY_DT)
        if _seg_label != _prev_seg_label:
            print(f"[sched] t={i * PLAY_DT:5.1f}s  → {_seg_label:8s} cmd={_seg_cmd}")
            _prev_seg_label = _seg_label
        env.commands[:, 0] = _seg_cmd[0]
        env.commands[:, 1] = _seg_cmd[1]
        env.commands[:, 2] = _seg_cmd[2]
        env.commands[:, 3] = 0.0
        # v3 yaw_hold：同步 raw（wz=0 段注入 hold 纠偏；转向段透传）
        env.raw_yaw_cmd[:] = env.commands[:, 2]

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        current_vel_x = env.base_lin_vel[0, 0].item()
        vel_sum += current_vel_x
        step_accum += 1
        avg_vel = vel_sum / step_accum if step_accum > 0 else 0.0
        _vf = _capture_video_frame(i, avg_vel)

        if csv_log_start <= i <= csv_log_end:
            _log_csv_step(i, vf_idx=_vf, actions_t=_last_actions)
            # 拖地风险预检：非接触（fz<=1N）时刻的脚底离地高度
            _fzl, _fzr, _lon, _ron = _foot_contacts()
            _feet_z = (env.rigid_state[robot_index, env.feet_indices, 2]
                       - env.cfg.rewards.feet_to_ankle_distance)
            if not _lon:
                swing_feet_z_log.append(('L', _feet_z[0].item()))
            if not _ron:
                swing_feet_z_log.append(('R', _feet_z[1].item()))
            logger.log_states(dict={
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'command_x': env.commands[robot_index, 0].item(),
                'command_y': env.commands[robot_index, 1].item(),
                'command_yaw': env.commands[robot_index, 2].item(),
                'video_frame': _vf,
            })

        if i == SEG_TOTAL_STEPS - 1:  # 八段全程结束：汇总（CSV 窗口已覆盖全程，独立触发）
            logger.plot_states()
            import numpy as _np
            print("\n" + "=" * 68)
            print("  全关节跟踪误差汇总  (err = act - des, 单位: °)")
            print("=" * 68)
            for _gname, _gidx, _gnames in DOF_SUMMARY_GROUPS:
                print(f"\n{'─' * 20} {_gname} {'─' * 20}")
                print(f"  {'DOF':>3}  {'关节':8}  {'mean':>7}  {'std':>6}  {'min':>7}  {'max':>7}")
                for _di, _dn in zip(_gidx, _gnames):
                    if not all_dof_log[_di]:
                        continue
                    _errs = _np.array([x[2] for x in all_dof_log[_di]]) * 57.3
                    print(f"  {_di:3d}  {_dn:8s}  {_np.mean(_errs):+7.1f}°  {_np.std(_errs):6.1f}°  "
                          f"{_np.min(_errs):+7.1f}°  {_np.max(_errs):+7.1f}°")
            if body_yaw_log:
                _byaws = _np.array([x[1] for x in body_yaw_log])
                print(f"\n  body_yaw: mean={_np.mean(_byaws):+.1f}°  max={_np.max(_byaws):+.1f}°")
            if swing_feet_z_log:
                _szs = _np.array([x[1] for x in swing_feet_z_log]) * 1000  # mm
                _szs_l = _np.array([x[1] for x in swing_feet_z_log if x[0] == 'L']) * 1000
                _szs_r = _np.array([x[1] for x in swing_feet_z_log if x[0] == 'R']) * 1000
                print(f"\n  swing 离地间隙(mm): min={_np.min(_szs):.1f}  p5={_np.percentile(_szs, 5):.1f}  "
                      f"mean={_np.mean(_szs):.1f}  峰值≈p95={_np.percentile(_szs, 95):.1f}")
                print(f"  L: min={_np.min(_szs_l):.1f}  R: min={_np.min(_szs_r):.1f}  "
                      f"(判据: min<5mm 无真机裕度，不装机)")
            print("=" * 68)

        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)

    _csv_file.close()
    print(f"[CSV] 已保存至: {_csv_path}")
    if RENDER:
        print(f"[sync] 视频总帧数={_video_frame_idx}")
        video.release()
        print(f"[VIDEO] 已保存至: {video_filepath}")


if __name__ == '__main__':
    EXPORT_POLICY = False
    RENDER = True
    args = get_args()
    play(args)
