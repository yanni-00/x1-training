"""Run the TASK054 stage-one recipe at 100 Hz with matched history timing."""

from collections import deque
import json
from math import isclose

import torch

from humanoid.f1_contract import CONFIG_ROOT
from .f1_equivariant_config import F1EquivariantCfgPPO
from .f1_target_speed_ramp import F1TargetSpeedRampCfg, F1TargetSpeedRampEnv


PROFILE = json.loads((CONFIG_ROOT / 'stage1_100hz_no_filter.json').read_text(encoding='utf-8'))
HISTORY_STRIDE = PROFILE['history_policy_step_stride']
POLICY_DT = PROFILE['sim_dt_s'] * PROFILE['decimation']

if (not isclose(POLICY_DT * PROFILE['policy_hz'], 1.0)
        or not isclose(POLICY_DT * HISTORY_STRIDE, PROFILE['source_observation_period_s'])
        or PROFILE['source_rollout_steps'] != F1EquivariantCfgPPO.runner.num_steps_per_env
        or PROFILE['sim_dt_s'] != F1TargetSpeedRampCfg.sim.dt
        or PROFILE['decimation'] * HISTORY_STRIDE != F1TargetSpeedRampCfg.control.decimation
        or any(PROFILE['filters'].values())):
    raise ValueError('100 Hz profile differs from the matched no-filter timing contract')


class F1TargetSpeedRamp100HzCfg(F1TargetSpeedRampCfg):
    class env(F1TargetSpeedRampCfg.env):
        observation_history_stride = HISTORY_STRIDE

    class control(F1TargetSpeedRampCfg.control):
        decimation = PROFILE['decimation']

    class domain_rand(F1TargetSpeedRampCfg.domain_rand):
        update_step = F1TargetSpeedRampCfg.domain_rand.update_step * HISTORY_STRIDE


class F1TargetSpeedRamp100HzCfgPPO(F1EquivariantCfgPPO):
    class runner(F1EquivariantCfgPPO.runner):
        experiment_name = 'f1_walk_stage1_100hz_054_no_filter'
        num_steps_per_env = F1EquivariantCfgPPO.runner.num_steps_per_env * HISTORY_STRIDE

    class algorithm(F1EquivariantCfgPPO.algorithm):
        gamma = F1EquivariantCfgPPO.algorithm.gamma ** (1 / HISTORY_STRIDE)
        lam = F1EquivariantCfgPPO.algorithm.lam ** (1 / HISTORY_STRIDE)


class F1TargetSpeedRamp100HzEnv(F1TargetSpeedRampEnv):
    def _init_buffers(self):
        super()._init_buffers()
        # Keep the newest frame and the same 20 ms tap spacing as TASK054.
        # The inherited reset clears every stored frame for the reset envs.
        for name in ('obs_history', 'critic_history'):
            history = getattr(self, name)
            length = (history.maxlen - 1) * HISTORY_STRIDE + 1
            expanded = deque(history, maxlen=length)
            for _ in range(length - len(history)):
                expanded.appendleft(torch.zeros_like(history[0]))
            setattr(self, name, expanded)
