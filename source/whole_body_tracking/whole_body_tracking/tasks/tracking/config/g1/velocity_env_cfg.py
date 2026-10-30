"""[OMA] Preserve native observations, PPO, robot control and all constraints."""
from isaaclab.utils import configclass
from .flat_env_cfg import G1FlatEnvCfg
from whole_body_tracking.tasks.tracking.mdp.velocity_gait import VelocityGaitCommandCfg


@configclass
class G1VelocityReferenceEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        old=self.commands.motion
        fields=['asset_name','motion_file','anchor_body_name','body_names','resampling_time_range',
                'pose_range','velocity_range','joint_position_range','adaptive_kernel_size',
                'adaptive_lambda','adaptive_uniform_ratio','adaptive_alpha']
        self.commands.motion=VelocityGaitCommandCfg(**{key:getattr(old,key) for key in fields},debug_vis=False)
