"""R34: isolate the existing upright forward command stage against R33."""
from legged_gym.envs.g1.g1r33_config import G1R33Cfg, G1R33CfgPPO


class G1R34Cfg(G1R33Cfg):
    # [OMA] design-hugwbc-g1-r34: one command-distribution functional unit.
    # Disable grid expansion so it cannot overwrite the chosen velocity range.
    # The inherited standing sampler remains, as do terrain and randomization.
    class commands(G1R33Cfg.commands):
        curriculum = False

        class ranges(G1R33Cfg.commands.ranges):
            # Source: existing eval_upright.py fixed forward command.
            lin_vel_x = [0.5, 0.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]


class G1R34CfgPPO(G1R33CfgPPO):
    class runner(G1R33CfgPPO.runner):
        # Identical R32 input, optimizer, update count and PPO as matched R33.
        experiment_name = 'g1_r34_forward_command_arm_tracking'
