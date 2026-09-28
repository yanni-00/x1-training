import torch

from legged_gym.envs.h1.h1 import H1Robot
from legged_gym.envs.g1.g1_config import G1Cfg


class G1Robot(H1Robot):
    """G1 (23-DOF) 移植版 H1Robot。

    差异点（design-hugwbc-g1-001 T2）：
    - waist 3 关节（H1 torso 1）：torso_inds 由 URDF dof 名自动匹配 'torso'→
      G1 无 torso 关节，override 为 waist 三关节索引；waist_roll 命令作用 waist_roll_joint
    - 观测/动作维度跟随 23 DOF（由 config 层决定，本类零硬编码）
    """

    def _create_envs(self):
        # 复用父类全部管线（feet/penalize/termination 名单均由 cfg.asset 驱动）
        super()._create_envs()
        # 修正 waist 索引：父类按 'torso' in dof_name 匹配（G1 无匹配项→空列表→
        # _reward_waist_control 会崩）；G1 用 waist_* 三关节
        dof_names_lower = [n.lower() for n in self.dof_names]
        waist_inds = [i for i, n in enumerate(dof_names_lower) if n.startswith('waist_')]
        waist_roll_inds = [i for i, n in enumerate(dof_names_lower) if n == 'waist_roll_joint']
        self.torso_inds = waist_inds                      # 腰部正则组（deviation 类 reward 用）
        self.waist_roll_inds = waist_roll_inds or waist_inds[:1]
        print('[g1] waist_inds =', waist_inds, '| waist_roll_inds =', self.waist_roll_inds)

    def _reward_waist_control(self):
        # H1: 单 torso_joint 跟踪 waist_roll 命令；G1: waist_roll_joint 跟踪
        waist_commands = self.commands[:, 9]
        reward = torch.square(self.dof_pos[:, self.waist_roll_inds] - waist_commands.unsqueeze(-1))
        return reward.sum(dim=-1)
