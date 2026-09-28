from legged_gym.envs.h1.h1interrupt_config import H1InterruptCfg, H1InterruptCfgPPO
from legged_gym.envs.g1.g1_config import G1Cfg, G1CfgPPO, PROPRIOCEPTION_DIM, CMD_DIM, CLOCK_INPUT, PRIVILEGED_DIM, TERRAIN_DIM

class G1InterruptCfg( G1Cfg ):
    # H1InterruptCfg 的 reward/command/disturb 差异全部继承链改为 G1 基座
    class env( G1Cfg.env ):
        num_observations = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM
        num_partial_obs = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT

class G1InterruptCfgPPO( G1CfgPPO ):
    class runner( G1CfgPPO.runner ):
        experiment_name = 'g1_teacher'
