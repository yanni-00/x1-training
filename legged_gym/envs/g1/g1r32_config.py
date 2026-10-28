"""R32: R30 settings with the existing pitch cost measured on torso_link."""
from legged_gym.envs.g1.g1r30_config import G1R30Cfg,G1R30CfgPPO


class G1R32Cfg(G1R30Cfg):
    class rewards(G1R30Cfg.rewards):
        orientation_control_body='torso_link'


class G1R32CfgPPO(G1R30CfgPPO):
    class runner(G1R30CfgPPO.runner):
        experiment_name='g1_r32_torso_orientation_frame'
