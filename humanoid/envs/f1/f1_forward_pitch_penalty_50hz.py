"""r28: r27 continuous-decay forward bonus + an independent PURE-STRAIGHT rearward-pitch cost,
applied AFTER the episode reward zero-clip (so it actually reaches the policy; r22/r23 added
their penalty before the clip, where it was diluted/clipped away).

Owner-calibrated first weight (from T444_1750 replay): per-step cost =
0.5 * dt * max(-pitch_deg - 9, 0) / 9  ->  ~0.00348/step on the 3-10 s steady straight segment,
comparable to r27's forward reward ~0.00318/step on the same frames.
"""
import math

import torch

from .f1_forward_pitch_decay_50hz import (
    F1ForwardPitchDecay50HzCfg, F1ForwardPitchDecay50HzEnv)


class F1ForwardPitchPenalty50HzCfg(F1ForwardPitchDecay50HzCfg):
    class env(F1ForwardPitchDecay50HzCfg.env):
        reward_semantics = (
            'stage1_sagittal_and_lateral_target_with_late_swing_direction_'
            'and_forward_pitch_decay_and_straight_rearward_cost_v1')

    class rewards(F1ForwardPitchDecay50HzCfg.rewards):
        # Pure-straight rearward pitch cost: 0 at <= band, grows linearly past it,
        # independent of speed. Subtracted from the (already zero-clipped) reward in step().
        rearward_pitch_band_deg = 9.0
        rearward_pitch_cost_scale = 0.5


class F1ForwardPitchPenalty50HzEnv(F1ForwardPitchDecay50HzEnv):
    def _rearward_pitch_cost(self):
        cfg = self.cfg.rewards
        threshold = self.cfg.commands.stand_com_threshold
        pure = ((self.commands[:, 0] > threshold)
                & (self.commands[:, 1].abs() <= threshold)
                & (self.commands[:, 2].abs() <= threshold))
        lean_deg = (-self.base_euler_xyz[:, 1]) * (180.0 / math.pi)
        excess = torch.clamp(lean_deg - cfg.rearward_pitch_band_deg, min=0.0)
        cost = cfg.rearward_pitch_cost_scale * self.dt * (excess / cfg.rearward_pitch_band_deg)
        return torch.where(pure, cost, torch.zeros_like(cost))

    def step(self, actions):
        obs, privileged, rewards, dones, infos = super().step(actions)
        cost = self._rearward_pitch_cost()
        # Apply AFTER the base reward zero-clip so the cost truly reaches the policy.
        rewards = rewards - cost
        # diagnostic: average cost on pure-straight steps (r28 counter)
        if not hasattr(self, '_r28_calls'):
            self._r28_calls = 0
            self._r28_cost_sum = 0.0
            self._r28_pure_steps = 0
        self._r28_calls += 1
        threshold = self.cfg.commands.stand_com_threshold
        pure = ((self.commands[:, 0] > threshold)
                & (self.commands[:, 1].abs() <= threshold)
                & (self.commands[:, 2].abs() <= threshold))
        self._r28_pure_steps += int(pure.sum())
        self._r28_cost_sum += float(cost[pure].sum())
        if self._r28_calls % 2000 == 0 and self._r28_pure_steps > 0:
            print('[R28LOG] calls=%d pure_straight_steps=%d avg_rearward_cost_on_pure=%.6f' % (
                self._r28_calls, self._r28_pure_steps,
                self._r28_cost_sum / self._r28_pure_steps), flush=True)
        return obs, privileged, rewards, dones, infos