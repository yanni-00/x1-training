"""R37: one existing external arm-posture scale sensitivity, matched to R36."""
from legged_gym.envs.g1.g1r36_config import G1R36Cfg,G1R36CfgPPO


class G1R37Cfg(G1R36Cfg):
    class rewards(G1R36Cfg.rewards):
        class scales(G1R36Cfg.rewards.scales):
            # [OMA] design-hugwbc-g1-r37: explicit twofold sensitivity design,
            # not an estimated sufficient/optimal weight. Keep the same arm
            # reference, coordinate priorities and all other reward components.
            shoulder_deviation = 2 * G1R36Cfg.rewards.scales.shoulder_deviation


class G1R37CfgPPO(G1R36CfgPPO):
    class runner(G1R36CfgPPO.runner):
        experiment_name = 'g1_r37_arm_posture_priority'
