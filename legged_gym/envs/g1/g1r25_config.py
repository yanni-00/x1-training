"""R25: translate the foot-body clearance reference to G1 collision geometry."""
import os

from legged_gym.envs.g1.g1r24_config import G1R24Cfg, G1R24CfgPPO


class G1R25Cfg(G1R24Cfg):
    class rewards(G1R24Cfg.rewards):
        # g1_29dof_anneal_23dof.urdf: all foot collision spheres have
        # origin.z=-.03 and radius=.005; level foot origin is .035 above sole.
        # Source: .oma/experiments/exp-20261027-002/geometry-reference.json.
        feet_clearance_reference_height = 0.035


class G1R25CfgPPO(G1R24CfgPPO):
    class runner(G1R24CfgPPO.runner):
        experiment_name = 'g1_r25_foot_clearance_reference'
        resume = True
        resume_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'resources',
            'checkpoints', 'model_10800_g1r24.pt'))
        max_iterations = 600
