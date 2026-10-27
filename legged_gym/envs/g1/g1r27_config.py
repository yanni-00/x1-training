"""R27 matched-input sensitivity probe for the existing height-cost scale."""
from legged_gym.envs.g1.g1r26_config import G1R26Cfg, G1R26CfgPPO


class G1R27Cfg(G1R26Cfg):
    class rewards(G1R26Cfg.rewards):
        class scales(G1R26Cfg.rewards.scales):
            # Preregistered factor2 sensitivity test after framework reflection;
            # no claim that this is an estimated required coefficient.
            base_height = 2.0 * G1R26Cfg.rewards.scales.base_height


class G1R27CfgPPO(G1R26CfgPPO):
    class runner(G1R26CfgPPO.runner):
        # Match R26's original10800 input,600 updates and curriculum initial state.
        experiment_name = 'g1_r27_height_weight_sensitivity'
