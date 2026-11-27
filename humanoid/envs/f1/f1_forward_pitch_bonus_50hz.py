"""r26: add a posture-gated pure-forward speed bonus on top of the r21 lateral recipe.

Single reward delta vs r21: `_reward_forward_pitch_bonus`, which re-pays the EXISTING
`_forward_progress_bonus()` (inherited from F1SpeedTrackingEnv, already gating pure
forward with |vy|<=deadband) only when the commanded yaw is ~0 AND the body pitch is
within the registered upright band. Effect: forward-speed reward is withheld the
moment the policy leans past the pitch line or turns, so it must walk forward fast
while staying relatively upright.
"""
import math

import torch

from .f1_lateral_swing_direction_50hz import (
    F1LateralSwingDirection50HzCfg, F1LateralSwingDirection50HzEnv)


class F1ForwardPitchBonus50HzCfg(F1LateralSwingDirection50HzCfg):
    class env(F1LateralSwingDirection50HzCfg.env):
        reward_semantics = (
            'stage1_sagittal_and_lateral_target_with_late_swing_direction_'
            'and_forward_pitch_bonus_v1')

    class rewards(F1LateralSwingDirection50HzCfg.rewards):
        # Bonus paid only when body pitch (base_euler_xyz[:,1], same euler logged as
        # base_pitch) is >= this value in degrees, i.e. the rearward lean is not past
        # the line. -9.0 = the registered straight-walk pitch gate; set -6.0 for a
        # stricter "relatively upright" threshold (owner tunable, single number).
        forward_pitch_gate_deg = -9.0

        class scales(F1LateralSwingDirection50HzCfg.rewards.scales):
            # Pays scale * dt * forward_progress_value on gated steps only.
            forward_pitch_bonus = 1.0


class F1ForwardPitchBonus50HzEnv(F1LateralSwingDirection50HzEnv):
    def _reward_forward_pitch_bonus(self):
        threshold = self.cfg.commands.stand_com_threshold
        # Pure forward yaw: commanded turn near zero (the |vy| proxy is already
        # applied inside _forward_progress_bonus).
        yaw_ok = self.commands[:, 2].abs() <= threshold
        pitch_ok = self.base_euler_xyz[:, 1] >= math.radians(
            self.cfg.rewards.forward_pitch_gate_deg)
        gate = (yaw_ok & pitch_ok).to(torch.float)
        # Reuse the existing forward progress computation (vx relative tracking,
        # pure-forward only); pay it again only on posture-clean straight steps.
        return self._forward_progress_bonus() * gate