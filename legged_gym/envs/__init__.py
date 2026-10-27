from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .h1.h1 import H1Robot
from legged_gym.envs.h1.h1interrupt import H1InterruptRobot

from legged_gym.envs.h1.h1_config import H1Cfg, H1CfgPPO
from legged_gym.envs.h1.h1interrupt_config import H1InterruptCfg, H1InterruptCfgPPO

from legged_gym.envs.g1.g1interrupt import G1InterruptRobot
from legged_gym.envs.g1.g1_config import G1Cfg, G1CfgPPO
from legged_gym.envs.g1.g1interrupt_config import G1InterruptCfg, G1InterruptCfgPPO
from legged_gym.envs.g1.g1r20 import G1R20Robot
from legged_gym.envs.g1.g1r20_config import G1R20Cfg, G1R20CfgPPO
from legged_gym.envs.g1.g1r21 import G1R21Robot
from legged_gym.envs.g1.g1r21_config import G1R21Cfg, G1R21CfgPPO
from legged_gym.envs.g1.g1r22_config import G1R22Cfg, G1R22CfgPPO
from legged_gym.envs.g1.g1r23 import G1R23Robot
from legged_gym.envs.g1.g1r23_config import G1R23Cfg, G1R23CfgPPO
from legged_gym.envs.g1.g1r24_config import G1R24Cfg, G1R24CfgPPO

from legged_gym.utils.task_registry import task_registry

task_registry.register( "h1int", H1InterruptRobot, H1InterruptCfg(), H1InterruptCfgPPO())
task_registry.register( "g1int", G1InterruptRobot, G1InterruptCfg(), G1InterruptCfgPPO())
task_registry.register( "g1r20", G1R20Robot, G1R20Cfg(), G1R20CfgPPO())
task_registry.register( "g1r21", G1R21Robot, G1R21Cfg(), G1R21CfgPPO())
task_registry.register( "g1r22", G1R21Robot, G1R22Cfg(), G1R22CfgPPO())
task_registry.register( "g1r23", G1R23Robot, G1R23Cfg(), G1R23CfgPPO())
task_registry.register( "g1r24", G1R23Robot, G1R24Cfg(), G1R24CfgPPO())

