"""R41: explicitly include pelvis in the existing orientation measurement."""
from legged_gym.envs.g1.g1r39_config import G1R39Cfg, G1R39CfgPPO


class G1R41Cfg(G1R39Cfg):
    class rewards(G1R39Cfg.rewards):
        orientation_control_include_pelvis = True


class G1R41CfgPPO(G1R39CfgPPO):
    class runner(G1R39CfgPPO.runner):
        experiment_name = 'g1_r41_worst_body_orientation'
