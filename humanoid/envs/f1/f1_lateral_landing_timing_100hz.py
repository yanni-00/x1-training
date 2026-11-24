"""Run the TASK054 stage-one recipe at 100 Hz with the lateral landing reward."""

from .f1_lateral_landing_timing import F1LateralLandingTimingCfg, F1LateralLandingTimingEnv
from .f1_target_speed_ramp_100hz import (
    POLICY_DT,
    F1TargetSpeedRamp100HzCfg,
    F1TargetSpeedRamp100HzCfgPPO,
    F1TargetSpeedRamp100HzEnv,
)


class F1LateralLandingTiming100HzCfg(F1LateralLandingTimingCfg, F1TargetSpeedRamp100HzCfg):
    # Nested classes are redeclared so the 100 Hz timing contract and the landing
    # semantics both survive: outer inheritance alone would shadow one of them.
    class env(F1LateralLandingTimingCfg.env, F1TargetSpeedRamp100HzCfg.env):
        pass

    class rewards(F1LateralLandingTimingCfg.rewards):
        pass

    class control(F1TargetSpeedRamp100HzCfg.control):
        pass

    class domain_rand(F1TargetSpeedRamp100HzCfg.domain_rand):
        pass


class F1LateralLandingTiming100HzEnv(F1LateralLandingTimingEnv, F1TargetSpeedRamp100HzEnv):
    pass


__all__ = [
    'POLICY_DT',
    'F1LateralLandingTiming100HzCfg',
    'F1LateralLandingTiming100HzEnv',
    'F1TargetSpeedRamp100HzCfgPPO',
]