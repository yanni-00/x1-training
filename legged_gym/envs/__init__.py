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
from legged_gym.envs.g1.g1r25_config import G1R25Cfg, G1R25CfgPPO
from legged_gym.envs.g1.g1r26_config import G1R26Cfg, G1R26CfgPPO
from legged_gym.envs.g1.g1r27_config import G1R27Cfg, G1R27CfgPPO
from legged_gym.envs.g1.g1r28_config import G1R28Cfg, G1R28CfgPPO
from legged_gym.envs.g1.g1r29 import G1R29Robot
from legged_gym.envs.g1.g1r29_config import G1R29Cfg, G1R29CfgPPO
from legged_gym.envs.g1.g1r30_config import G1R30Cfg, G1R30CfgPPO
from legged_gym.envs.g1.g1r31_config import G1R31Cfg, G1R31CfgPPO
from legged_gym.envs.g1.g1r32 import G1R32Robot
from legged_gym.envs.g1.g1r32_config import G1R32Cfg, G1R32CfgPPO
from legged_gym.envs.g1.g1r33_config import G1R33Cfg, G1R33CfgPPO
from legged_gym.envs.g1.g1r34_config import G1R34Cfg, G1R34CfgPPO
from legged_gym.envs.g1.g1r35_config import G1R35Cfg, G1R35CfgPPO
from legged_gym.envs.g1.g1r36 import G1R36Robot
from legged_gym.envs.g1.g1r36_config import G1R36Cfg, G1R36CfgPPO
from legged_gym.envs.g1.g1r37_config import G1R37Cfg, G1R37CfgPPO
from legged_gym.envs.g1.g1r38_config import G1R38Cfg, G1R38CfgPPO
from legged_gym.envs.g1.g1r39 import G1R39Robot
from legged_gym.envs.g1.g1r39_config import G1R39Cfg, G1R39CfgPPO
from legged_gym.envs.g1.g1r40_config import G1R40Cfg, G1R40CfgPPO
from legged_gym.envs.g1.g1r41 import G1R41Robot
from legged_gym.envs.g1.g1r41_config import G1R41Cfg, G1R41CfgPPO

from legged_gym.utils.task_registry import task_registry

task_registry.register( "h1int", H1InterruptRobot, H1InterruptCfg(), H1InterruptCfgPPO())
task_registry.register( "g1int", G1InterruptRobot, G1InterruptCfg(), G1InterruptCfgPPO())
task_registry.register( "g1r20", G1R20Robot, G1R20Cfg(), G1R20CfgPPO())
task_registry.register( "g1r21", G1R21Robot, G1R21Cfg(), G1R21CfgPPO())
task_registry.register( "g1r22", G1R21Robot, G1R22Cfg(), G1R22CfgPPO())
task_registry.register( "g1r23", G1R23Robot, G1R23Cfg(), G1R23CfgPPO())
task_registry.register( "g1r24", G1R23Robot, G1R24Cfg(), G1R24CfgPPO())
task_registry.register( "g1r25", G1R23Robot, G1R25Cfg(), G1R25CfgPPO())
task_registry.register( "g1r26", G1R23Robot, G1R26Cfg(), G1R26CfgPPO())
task_registry.register( "g1r27", G1R23Robot, G1R27Cfg(), G1R27CfgPPO())
task_registry.register( "g1r28", G1R23Robot, G1R28Cfg(), G1R28CfgPPO())
task_registry.register( "g1r29", G1R29Robot, G1R29Cfg(), G1R29CfgPPO())
task_registry.register( "g1r30", G1R29Robot, G1R30Cfg(), G1R30CfgPPO())
task_registry.register( "g1r31", G1R29Robot, G1R31Cfg(), G1R31CfgPPO())
task_registry.register( "g1r32", G1R32Robot, G1R32Cfg(), G1R32CfgPPO())
task_registry.register( "g1r33", G1R32Robot, G1R33Cfg(), G1R33CfgPPO())
task_registry.register( "g1r34", G1R32Robot, G1R34Cfg(), G1R34CfgPPO())
task_registry.register( "g1r35", G1R32Robot, G1R35Cfg(), G1R35CfgPPO())
task_registry.register( "g1r36", G1R36Robot, G1R36Cfg(), G1R36CfgPPO())
task_registry.register( "g1r37", G1R36Robot, G1R37Cfg(), G1R37CfgPPO())
task_registry.register( "g1r38", G1R36Robot, G1R38Cfg(), G1R38CfgPPO())
task_registry.register( "g1r39", G1R39Robot, G1R39Cfg(), G1R39CfgPPO())
task_registry.register( "g1r40", G1R39Robot, G1R40Cfg(), G1R40CfgPPO())
task_registry.register( "g1r41", G1R41Robot, G1R41Cfg(), G1R41CfgPPO())

