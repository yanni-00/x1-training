"""Add late-swing lateral direction supervision to the stage-one recipe.

The term is a dense per-step bonus on the swing-foot lateral velocity, taken in
the base frame and signed by the lateral command. It is gated to lateral
commands (the same gate as the lateral progress round) and ramped by the square
of normalized swing time, so late swing carries the supervision while the whole
swing still receives a gradient. The term is non-negative inside its own clip;
the pipeline positive-only clip is unchanged.
"""

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from .f1_lateral_swing_direction_metric import swing_direction_step, update_air_time
from .f1_target_speed_ramp import F1TargetSpeedRampCfg, F1TargetSpeedRampEnv


class F1LateralSwingDirectionCfg(F1TargetSpeedRampCfg):
    class env(F1TargetSpeedRampCfg.env):
        reward_semantics = 'target_speed_ramp_with_lateral_swing_direction_v1'

    class rewards(F1TargetSpeedRampCfg.rewards):
        class scales(F1TargetSpeedRampCfg.rewards.scales):
            # Weight derivation: per late swing (0.25 s ramp, 0.3 s airborne)
            # the sum of dt*ramp*progress is about 0.15*0.5*progress. At
            # progress 0.5 that is 0.0375 per swing, 2 swings per 0.7 s cycle,
            # so weight 20 gives about 1.5 per cycle against tracking_lin_vel
            # 1.26 per cycle: the same order as the sparse landing round.
            lateral_swing_direction = 20.0

        lateral_swing_direction_v_ref = 0.4
        lateral_swing_direction_ramp_s = 0.25
        lateral_swing_direction_touchdown_n = 40.0
        lateral_swing_direction_liftoff_n = 5.0


class F1LateralSwingDirectionEnv(F1TargetSpeedRampEnv):
    def _init_buffers(self):
        super()._init_buffers()
        self._swing_air_time = torch.zeros(self.num_envs, len(self.feet_indices),
                                           dtype=torch.float, device=self.device,
                                           requires_grad=False)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, '_swing_air_time') and len(env_ids):
            self._swing_air_time[env_ids] = 0.0

    def _reward_lateral_swing_direction(self):
        cfg = self.cfg.rewards
        forces = self.contact_forces[:, self.feet_indices, 2]
        swing = (forces <= cfg.lateral_swing_direction_liftoff_n).to(torch.float)
        gate = (self.commands[:, 0].abs() <= self.cfg.commands.stand_com_threshold) & \
               (self.commands[:, 1].abs() > self.cfg.commands.stand_com_threshold)
        feet = len(self.feet_indices)
        quat = self.base_quat.unsqueeze(1).repeat(1, feet, 1).reshape(-1, 4)
        vel = self.rigid_state[:, self.feet_indices, 7:10].reshape(-1, 3)
        foot_vel_body = quat_rotate_inverse(quat, vel).reshape(self.num_envs, feet, 3)
        toward = torch.sign(self.commands[:, 1:2]) * foot_vel_body[..., 1]
        reward = swing_direction_step(swing, toward, self._swing_air_time, gate,
                                      cfg.lateral_swing_direction_v_ref,
                                      cfg.lateral_swing_direction_ramp_s, self.dt)
        self._swing_air_time = update_air_time(self._swing_air_time, swing, self.dt)
        return reward