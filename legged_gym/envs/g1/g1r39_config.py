"""R39: owner-confirmed complete G1 arm rest pose, original R36 controller."""
from legged_gym.envs.g1.g1r36_config import G1R36Cfg, G1R36CfgPPO


class G1R39Cfg(G1R36Cfg):
    class rewards(G1R36Cfg.rewards):
        # [OMA] design-hugwbc-g1-r39: project UNITREE_G1_23DOF_CFG source
        # supplies pitch .3/elbow .97. Roll .27 is the tested collision grid
        # minimum .26 plus one tested .01 interval, not a firmware claim.
        arm_rest_shoulder_pitch_rad = 0.3
        arm_rest_elbow_rad = 0.97
        arm_rest_shoulder_roll_rad = 0.27


class G1R39CfgPPO(G1R36CfgPPO):
    class runner(G1R36CfgPPO.runner):
        experiment_name = 'g1_r39_complete_arm_rest_pose'
