"""Add late-swing lateral direction supervision to the 50 Hz stage-one lateral target recipe.

Same term as the 100 Hz r10/r11 rounds: a dense per-step bonus on the
swing-foot lateral velocity, taken in the base frame and signed by the lateral
command, gated to lateral commands and ramped by the square of normalized swing
time. The term is non-negative inside its own clip; the pipeline positive-only
clip is unchanged.
"""

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from .f1_lateral_swing_direction_metric import swing_direction_step, update_air_time
from .f1_lateral_target import F1LateralTargetCfg, F1LateralTargetEnv


class F1LateralSwingDirection50HzCfg(F1LateralTargetCfg):
    class env(F1LateralTargetCfg.env):
        reward_semantics = 'stage1_sagittal_and_lateral_target_with_late_swing_direction_v1'

    class rewards(F1LateralTargetCfg.rewards):
        class scales(F1LateralTargetCfg.rewards.scales):
            # Weight carried over from the 100 Hz r11 acceptance round (w60).
            # The framework pays scale * dt with dt = 0.02 here (vs 0.01 at
            # 100 Hz), so per-step cost is doubled; the ramp 0.25 s and cycle
            # 0.7 s are unchanged, and the offline smoke test verifies the
            # effective coefficient before launch.
            lateral_swing_direction = 60.0

        lateral_swing_direction_v_ref = 0.4
        lateral_swing_direction_ramp_s = 0.25
        lateral_swing_direction_touchdown_n = 40.0
        lateral_swing_direction_liftoff_n = 5.0


class F1LateralSwingDirection50HzEnv(F1LateralTargetEnv):
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
                                      cfg.lateral_swing_direction_ramp_s)
        self._swing_air_time = update_air_time(self._swing_air_time, swing, self.dt)
        return reward