from legged_gym.envs.h1.h1interrupt import H1InterruptRobot
from legged_gym.envs.g1.g1 import G1Robot
from legged_gym.envs.g1.g1interrupt_config import G1InterruptCfg


class G1InterruptRobot(G1Robot, H1InterruptRobot):
    """G1 (23-DOF) + 干预噪声课程（论文主结果路线）。

    MRO: G1Robot._create_envs 先行（waist 索引修正），H1InterruptRobot 提供
    干预管线（interrupt_mask/disturb 课程/上肢正则乘子）——其关节词硬编码
    （waist_roll 命令维度、left/right 干预位）与 G1 命名空间兼容，零改动。
    """
    pass
