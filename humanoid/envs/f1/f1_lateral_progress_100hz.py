"""Run the TASK054 stage-one recipe at 100 Hz with the signed lateral progress reward."""

from .f1_target_speed_ramp_100hz import (
    POLICY_DT,
    F1TargetSpeedRamp100HzCfg,
    F1TargetSpeedRamp100HzCfgPPO,
    F1TargetSpeedRamp100HzEnv,
)
from .f1_lateral_progress import F1LateralProgressCfg, F1LateralProgressEnv


class F1LateralProgress100HzCfg(F1TargetSpeedRamp100HzCfg, F1LateralProgressCfg):
    pass


class F1LateralProgress100HzEnv(F1TargetSpeedRamp100HzEnv, F1LateralProgressEnv):
    pass


__all__ = ['POLICY_DT', 'F1LateralProgress100HzCfg', 'F1TargetSpeedRamp100HzCfgPPO',
           'F1LateralProgress100HzEnv']