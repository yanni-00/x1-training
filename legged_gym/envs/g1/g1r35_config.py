"""R35: one finite elbow-priority probe in the R34 forward/torso stage."""
from legged_gym.envs.g1.g1r34_config import G1R34Cfg, G1R34CfgPPO


class G1R35Cfg(G1R34Cfg):
    class rewards(G1R34Cfg.rewards):
        # [OMA] design-hugwbc-g1-r35: explicit intermediate design setting,
        # reusing the existing roll/yaw coefficient; not an estimated optimum.
        # Keep the original external component and pitch/reference weights.
        arm_rest_elbow_weight = 2.0


class G1R35CfgPPO(G1R34CfgPPO):
    class runner(G1R34CfgPPO.runner):
        experiment_name = 'g1_r35_bounded_elbow_priority'
