"""r27: posture-decay pure-forward speed bonus (continuous, not a hard -9 gate).

r26 used a hard gate: past -9 deg the forward bonus paid exactly 0 (owner frame
recompute: 210/350 at the parent, 0/350 at r26_1750 and r26_2000), so the leaned
policy got no gradient back toward upright. r27 keeps -9 deg as the ACCEPTANCE
line only and changes the training reward to a smooth factor that decays with
rearward lean: standing up even past -9 still raises the return.

Single reward delta vs r21/r25: `_reward_forward_pitch_bonus` = existing
`_forward_progress_bonus()` * (yaw~0) * smooth_upright_factor(pitch).
"""
import math

import torch

from .f1_lateral_swing_direction_50hz import (
    F1LateralSwingDirection50HzCfg, F1LateralSwingDirection50HzEnv)


class F1ForwardPitchDecay50HzCfg(F1LateralSwingDirection50HzCfg):
    class env(F1LateralSwingDirection50HzCfg.env):
        reward_semantics = (
            'stage1_sagittal_and_lateral_target_with_late_swing_direction_'
            'and_forward_pitch_decay_v1')

    class rewards(F1LateralSwingDirection50HzCfg.rewards):
        # Upright factor u = clamp(1 - (lean_deg - ref_lean_deg)/span_deg, 0, 1),
        # lean_deg = -base_pitch_deg (rearward lean magnitude). u=1 at/above ref,
        # decays linearly, 0 at ref+span. Defaults: u=0.85 at -9, 0.55 at -15,
        # 0.37 at -18.6, so the factor stays positive through the observed lean.
        forward_pitch_ref_lean_deg = 6.0
        forward_pitch_span_deg = 20.0

        class scales(F1LateralSwingDirection50HzCfg.rewards.scales):
            forward_pitch_bonus = 1.0


class F1ForwardPitchDecay50HzEnv(F1LateralSwingDirection50HzEnv):
    def _r27_log_init(self):
        if not hasattr(self, '_r27_calls'):
            self._r27_calls = 0
            self._r27_pf_steps = 0        # pure-forward env-steps seen
            self._r27_pf_bonus_nonzero = 0  # pure-forward env-steps where bonus != 0
            self._r27_pf_decay_sum = 0.0   # sum of upright factor over pure-forward steps

    def _reward_forward_pitch_bonus(self):
        cfg = self.cfg.rewards
        threshold = self.cfg.commands.stand_com_threshold
        # Pure forward: same |vy| proxy used inside _forward_progress_bonus.
        pure_fwd = (self.commands[:, 0] > threshold) & (self.commands[:, 1].abs() <= threshold)
        yaw_ok = self.commands[:, 2].abs() <= threshold
        lean_deg = (-self.base_euler_xyz[:, 1]) * (180.0 / math.pi)
        extra = torch.clamp(lean_deg - cfg.forward_pitch_ref_lean_deg, min=0.0)
        upright = torch.clamp(1.0 - extra / cfg.forward_pitch_span_deg, min=0.0, max=1.0)
        gate = yaw_ok.to(torch.float) * upright
        # Existing pure-forward progress (vx relative tracking) re-paid, scaled by the
        # smooth upright factor so leaning costs reward continuously.
        bonus = self._forward_progress_bonus() * gate

        # r27 diagnostic: track trigger rate and average decay on pure-forward steps
        # (r26 could not be derived from the episode reward sum alone). Printed periodically.
        self._r27_log_init()
        self._r27_calls += 1
        self._r27_pf_steps += int(pure_fwd.sum())
        self._r27_pf_bonus_nonzero += int((bonus != 0).sum())
        self._r27_pf_decay_sum += float(upright[pure_fwd].sum())
        if self._r27_calls % 2000 == 0 and self._r27_pf_steps > 0:
            n = self._r27_pf_steps
            print('[R27LOG] calls=%d pure_fwd_steps=%d bonus_nonzero_steps=%d '
                  'trigger_rate=%.4f avg_decay_on_pf=%.4f' % (
                      self._r27_calls, n, self._r27_pf_bonus_nonzero,
                      self._r27_pf_bonus_nonzero / n, self._r27_pf_decay_sum / n),
                  flush=True)
        return bonus