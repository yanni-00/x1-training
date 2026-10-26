"""R22 single-variable calibration of the nominal G1 pelvis height."""
import os

from legged_gym.envs.g1.g1r21_config import G1R21Cfg, G1R21CfgPPO


class G1R22Cfg(G1R21Cfg):
    class rewards(G1R21Cfg.rewards):
        # URDF forward kinematics at G1Cfg.init_state.default_joint_angles:
        # both ankle-roll foot collision soles contact z=0 at this pelvis z.
        # Source: .oma/experiments/exp-20261026-002/kinematic-height-reference.json
        # This is a geometric reference, not an assumed dynamic walking height.
        base_height_target = 0.7842021819153314


class G1R22CfgPPO(G1R21CfgPPO):
    class runner(G1R21CfgPPO.runner):
        experiment_name = 'g1_r22_nominal_height'
        resume = True
        resume_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'resources',
            'checkpoints', 'model_8400_g1r21.pt'))
