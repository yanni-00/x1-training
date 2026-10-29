"""R38: G1 asset arm PD only, after full-reference/execution diagnostics."""
from legged_gym.envs.g1.g1_config import G1Cfg
from legged_gym.envs.g1.g1r36_config import G1R36Cfg, G1R36CfgPPO


class G1R38Cfg(G1R36Cfg):
    class control(G1R36Cfg.control):
        # [OMA] design-hugwbc-g1-r38: translate only the arm servo group.
        # Actual existing G1 asset config values, not an estimated gain ladder.
        # Legs/waist, action mapping, torque limits and physics stay inherited.
        stiffness = {name: (G1Cfg.control.stiffness[name] if name in
            ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow') else value)
            for name,value in G1R36Cfg.control.stiffness.items()}
        damping = {name: (G1Cfg.control.damping[name] if name in
            ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow') else value)
            for name,value in G1R36Cfg.control.damping.items()}


class G1R38CfgPPO(G1R36CfgPPO):
    class runner(G1R36CfgPPO.runner):
        experiment_name = 'g1_r38_asset_arm_pd_tracking'
