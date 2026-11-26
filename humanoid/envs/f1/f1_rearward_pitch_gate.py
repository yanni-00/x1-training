"""Add the owner-gated rearward pitch penalty to the 50 Hz swing-direction recipe.

Owner gate for r22 (2026-11-26): penalize sustained rearward lean during PURE
forward walking only — vx commanded, vy AND wz both near zero — so that
forward-with-turn combinations stay outside the penalty. The swing-direction
reward and every other term are unchanged from r21.
"""

import torch

from .f1_lateral_swing_direction_50hz import F1LateralSwingDirection50HzCfg, F1LateralSwingDirection50HzEnv


class F1RearwardPitchGateCfg(F1LateralSwingDirection50HzCfg):
    class env(F1LateralSwingDirection50HzCfg.env):
        reward_semantics = 'stage1_lateral_target_swing_direction_with_rearward_pitch_gate_v1'

    class rewards(F1LateralSwingDirection50HzCfg.rewards):
        # Additional penalty coefficient applied to the rearward gravity
        # component inside _reward_orientation (framework multiplies the
        # orientation scale by dt; this factor scales the extra deduction).
        # r23: 2.0 -> 30. r22 measured ~8.5% gate-on training steps and
        # 0.012/step paid deduction, 20-60x below the swing-direction term;
        # 30 lifts the deduction at the observed -14.7 deg lean to
        # 30*0.254*0.02 = 0.152/step, the same order as the lateral gains.
        rearward_pitch_penalty = 30.0
        rearward_pitch_wz_threshold = 0.1


class F1RearwardPitchGateEnv(F1LateralSwingDirection50HzEnv):
    def _reward_orientation(self):
        original = super()._reward_orientation()
        cfg = self.cfg.rewards
        threshold = self.cfg.commands.stand_com_threshold
        # Pure forward only: vx commanded, vy and wz near zero.
        pure_forward = (self.commands[:, 0] > threshold) & \
                       (self.commands[:, 1].abs() <= threshold) & \
                       (self.commands[:, 2].abs() <= cfg.rearward_pitch_wz_threshold)
        # With gravity along world -z, its body x component is sin(base pitch).
        rearward = (-self.projected_gravity[:, 0]).clamp_min(0.0)
        penalty = cfg.rearward_pitch_penalty * rearward
        return original - torch.where(pure_forward, penalty, torch.zeros_like(penalty))