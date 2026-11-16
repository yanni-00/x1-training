"""Same stage-one task/rewards; mirror constraint is in the learned actor."""
from .f1_directional_posture import F1DirectionalPostureCfg
from .f1_50hz_config import F1Stage1NoFilter50HzCfgPPO


class F1EquivariantCfgPPO(F1Stage1NoFilter50HzCfgPPO):
    class policy(F1Stage1NoFilter50HzCfgPPO.policy):
        mirror_commands_scale = [F1DirectionalPostureCfg.normalization.obs_scales.lin_vel]*2 + [F1DirectionalPostureCfg.normalization.obs_scales.ang_vel]
        mirror_stand_threshold = F1DirectionalPostureCfg.commands.stand_com_threshold

    class runner(F1Stage1NoFilter50HzCfgPPO.runner):
        policy_class_name = 'ActorCriticF1Equivariant'
        experiment_name = 'f1_walk_stage1_equivariant'
