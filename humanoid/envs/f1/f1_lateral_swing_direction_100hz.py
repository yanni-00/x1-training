"""Run the TASK054 stage-one recipe at 100 Hz with late-swing lateral direction supervision."""

from .f1_target_speed_ramp_100hz import (
    POLICY_DT,
    F1TargetSpeedRamp100HzCfg,
    F1TargetSpeedRamp100HzCfgPPO,
    F1TargetSpeedRamp100HzEnv,
)
from .f1_lateral_swing_direction import F1LateralSwingDirectionCfg, F1LateralSwingDirectionEnv


class F1LateralSwingDirection100HzCfg(F1LateralSwingDirectionCfg, F1TargetSpeedRamp100HzCfg):
    # Nested classes are redeclared so the 100 Hz timing contract and the new
    # reward scales both survive: outer inheritance alone would shadow one side.
    class env(F1LateralSwingDirectionCfg.env, F1TargetSpeedRamp100HzCfg.env):
        pass

    class rewards(F1LateralSwingDirectionCfg.rewards):
        pass

    class control(F1TargetSpeedRamp100HzCfg.control):
        pass

    class domain_rand(F1TargetSpeedRamp100HzCfg.domain_rand):
        pass


class F1LateralSwingDirection100HzEnv(F1LateralSwingDirectionEnv, F1TargetSpeedRamp100HzEnv):
    pass


__all__ = [
    'POLICY_DT',
    'F1LateralSwingDirection100HzCfg',
    'F1LateralSwingDirection100HzEnv',
    'F1TargetSpeedRamp100HzCfgPPO',
]