"""[OMA] Preserve native robot, observations, rewards and terminations."""
from isaaclab.utils import configclass
from .flat_env_cfg import G1FlatEnvCfg
from whole_body_tracking.tasks.tracking.mdp.walk_stop import WalkStopCommandCfg


@configclass
class G1WalkStopEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        old = self.commands.motion
        self.commands.motion = WalkStopCommandCfg(
            asset_name=old.asset_name,
            motion_file=old.motion_file,
            anchor_body_name=old.anchor_body_name,
            body_names=old.body_names,
            resampling_time_range=old.resampling_time_range,
            pose_range=old.pose_range,
            velocity_range=old.velocity_range,
            joint_position_range=old.joint_position_range,
            adaptive_kernel_size=old.adaptive_kernel_size,
            adaptive_lambda=old.adaptive_lambda,
            adaptive_uniform_ratio=old.adaptive_uniform_ratio,
            adaptive_alpha=old.adaptive_alpha,
            debug_vis=False,
        )
