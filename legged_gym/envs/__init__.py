from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .h1.h1 import H1Robot
from legged_gym.envs.h1.h1interrupt import H1InterruptRobot

from legged_gym.envs.h1.h1_config import H1Cfg, H1CfgPPO
from legged_gym.envs.h1.h1interrupt_config import H1InterruptCfg, H1InterruptCfgPPO

from legged_gym.envs.g1.g1interrupt import G1InterruptRobot
from legged_gym.envs.g1.g1_config import G1Cfg, G1CfgPPO, G1InterruptCfg, G1InterruptCfgPPO

from legged_gym.utils.task_registry import task_registry

task_registry.register( "h1int", H1InterruptRobot, H1InterruptCfg(), H1InterruptCfgPPO())
task_registry.register( "g1int", G1InterruptRobot, G1InterruptCfg(), G1InterruptCfgPPO())

