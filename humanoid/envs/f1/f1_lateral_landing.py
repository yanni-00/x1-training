"""Add the touchdown-settled pure-lateral landing reward to the stage-one recipe."""

from .f1_lateral_landing_metric import landing_step, new_landing_state
from .f1_target_speed_ramp import F1TargetSpeedRampCfg, F1TargetSpeedRampEnv


class F1LateralLandingCfg(F1TargetSpeedRampCfg):
    class env(F1TargetSpeedRampCfg.env):
        reward_semantics = 'target_speed_ramp_with_lateral_landing_v1'

    class rewards(F1TargetSpeedRampCfg.rewards):
        class scales(F1TargetSpeedRampCfg.rewards.scales):
            # Declared design input. Two touch-down settlements per 0.7 s cycle:
            # 2 * 1.0 * 50 * dt = 1.0 per cycle, against tracking_lin_vel at
            # 1.8 * 0.7 = 1.26 per cycle. Sparse by construction.
            lateral_landing = 50.0

        lateral_landing_touchdown_n = 40.0
        lateral_landing_liftoff_n = 5.0
        lateral_landing_min_airborne_steps = 3
        lateral_landing_max_gap_steps = 1


class F1LateralLandingEnv(F1TargetSpeedRampEnv):
    def _init_buffers(self):
        super()._init_buffers()
        self._landing = new_landing_state(self.num_envs, self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, '_landing') and len(env_ids):
            self._landing['armed'][env_ids] = False

    def _reward_lateral_landing(self):
        cfg = self.cfg.rewards
        return landing_step(
            self._landing,
            self.rigid_state[:, self.feet_indices, :2],
            self.contact_forces[:, self.feet_indices, 2],
            self.base_euler_xyz[:, 2],
            self.commands[:, 0],
            self.commands[:, 1],
            touchdown_n=cfg.lateral_landing_touchdown_n,
            liftoff_n=cfg.lateral_landing_liftoff_n,
            min_air_steps=cfg.lateral_landing_min_airborne_steps,
            max_gap_steps=cfg.lateral_landing_max_gap_steps,
            stand_com_threshold=self.cfg.commands.stand_com_threshold,
            cycle_time=cfg.cycle_time,
            command_scale=self.cfg.commands.ranges.lin_vel_y[1],
        )