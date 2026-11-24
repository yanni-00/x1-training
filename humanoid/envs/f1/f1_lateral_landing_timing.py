"""Add the post-clip potential-based timing term on top of the r7 landing reward."""

import torch

from .f1_lateral_landing import F1LateralLandingCfg, F1LateralLandingEnv
from .f1_lateral_landing_timing_metric import new_timing_state, timing_step


class F1LateralLandingTimingCfg(F1LateralLandingCfg):
    class env(F1LateralLandingCfg.env):
        reward_semantics = 'target_speed_ramp_with_lateral_landing_timing_v1'

    class rewards(F1LateralLandingCfg.rewards):
        # Declared: the PBRS uses the same scale as the r7 sparse weight so the
        # prepaid credit and the terminal claw-back are on the r7 settlement
        # scale. gamma matches the 100 Hz PPO effective discount:
        # F1WalkingCfgPPO.algorithm.gamma ** (1 / history_policy_step_stride).
        lateral_timing_gamma = 0.994 ** 0.5
        lateral_timing_scale = 50.0


class F1LateralLandingTimingEnv(F1LateralLandingEnv):
    def _init_buffers(self):
        super()._init_buffers()
        self._timing = new_timing_state(self.num_envs, self.device)

    def _prepare_reward_function(self):
        # episode_sums is created by the base prepare step, after _init_buffers.
        super()._prepare_reward_function()
        self.episode_sums['lateral_landing_timing'] = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, '_timing') and len(env_ids):
            self._timing['armed'][env_ids] = False

    def _timing_term(self):
        cfg = self.cfg.rewards
        terminal = self.reset_buf | self.time_out_buf
        return timing_step(
            self._timing,
            self.rigid_state[:, self.feet_indices, :2],
            self.contact_forces[:, self.feet_indices, 2],
            self.base_euler_xyz[:, 2],
            self.commands[:, 0],
            self.commands[:, 1],
            terminal,
            gamma=cfg.lateral_timing_gamma,
            weight=cfg.lateral_timing_scale,
            dt=self.dt,
            touchdown_n=cfg.lateral_landing_touchdown_n,
            liftoff_n=cfg.lateral_landing_liftoff_n,
            min_air_steps=cfg.lateral_landing_min_airborne_steps,
            max_gap_steps=cfg.lateral_landing_max_gap_steps,
            stand_com_threshold=self.cfg.commands.stand_com_threshold,
            cycle_time=cfg.cycle_time,
            command_scale=self.cfg.commands.ranges.lin_vel_y[1],
        )

    def compute_reward(self):
        # r7 rewards and the positive-only clip first, then the timing term so a
        # claw-back is never clipped away.
        super().compute_reward()
        term = self._timing_term()
        self.rew_buf += term
        self.episode_sums['lateral_landing_timing'] += term
