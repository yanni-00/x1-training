"""Run the TASK054 stage-one recipe at 100 Hz with the lateral reference profile."""

from .f1_lateral_progress_100hz import F1LateralProgress100HzCfg, F1LateralProgress100HzEnv
from .f1_lateral_reference import F1LateralReferenceCfg, F1LateralReferenceEnv
from .f1_target_speed_ramp_100hz import POLICY_DT, F1TargetSpeedRamp100HzCfgPPO


class F1LateralReference100HzCfg(F1LateralReferenceCfg, F1LateralProgress100HzCfg):
    # Nested classes are redeclared so the 100 Hz timing contract and the lateral
    # reference semantics both survive: outer inheritance alone would shadow one
    # of the two, because each parent defines its own nested class.
    class env(F1LateralReferenceCfg.env, F1LateralProgress100HzCfg.env):
        pass

    class rewards(F1LateralReferenceCfg.rewards):
        pass

    class control(F1LateralProgress100HzCfg.control):
        pass

    class domain_rand(F1LateralProgress100HzCfg.domain_rand):
        pass


class F1LateralReference100HzEnv(F1LateralProgress100HzEnv, F1LateralReferenceEnv):
    pass


__all__ = [
    'POLICY_DT',
    'F1LateralReference100HzCfg',
    'F1LateralReference100HzEnv',
    'F1TargetSpeedRamp100HzCfgPPO',
]