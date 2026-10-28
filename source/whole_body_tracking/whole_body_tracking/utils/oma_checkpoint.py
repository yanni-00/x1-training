"""[OMA] Exact motion/control state needed by the existing MuJoCo play path."""
import hashlib
from pathlib import Path
import numpy as np
import torch


def collect_runtime(env, policy):
    base = env.unwrapped
    command = base.command_manager.get_term("motion")
    robot = command.robot
    data = robot.data
    with np.load(command.cfg.motion_file, allow_pickle=False) as source:
        joint_names = source["joint_names"].tolist()
        body_names = source["body_names"].tolist()
        motion = {key: torch.from_numpy(source[key].copy()) for key in
                  ["fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                   "body_lin_vel_w", "body_ang_vel_w", "support_frames"]}
    if joint_names != list(robot.joint_names):
        raise RuntimeError("Checkpoint reference/articulation joint order mismatch")
    if motion["joint_pos"].shape[1] != 29 or len(joint_names) != 29:
        raise RuntimeError("Checkpoint reference joint dimension mismatch")
    if any(not torch.isfinite(value).all() for value in motion.values()):
        raise RuntimeError("Non-finite runtime reference")
    if len(body_names) != motion["body_pos_w"].shape[1]:
        raise RuntimeError("Checkpoint reference body order mismatch")
    action = base.action_manager.get_term("joint_pos")
    scale = action._scale
    scale = scale[0].detach().cpu().clone() if isinstance(scale, torch.Tensor) else torch.full((29,), float(scale))
    nominal = getattr(data, "default_joint_pos_nominal", None)
    if nominal is None:
        raise RuntimeError("Nominal pre-randomization joint positions were not recorded")
    control = {
        "default_joint_pos": nominal.detach().cpu().clone(),
        "kp": data.default_joint_stiffness[0].detach().cpu().clone(),
        "kd": data.default_joint_damping[0].detach().cpu().clone(),
        "armature": data.default_joint_armature[0].detach().cpu().clone(),
        "effort_limit": data.joint_effort_limits[0].detach().cpu().clone(),
        "action_scale": scale,
    }
    if any(value.shape != (29,) or not torch.isfinite(value).all() for value in control.values()):
        raise RuntimeError("Invalid runtime control parameters")
    normalizer = getattr(policy, "actor_obs_normalizer", None)
    eps = getattr(normalizer, "eps", None)
    return {
        "format": "beyondmimic_g1_runtime_v1",
        "joint_names": joint_names, "body_names": body_names,
        "tracked_body_names": list(command.cfg.body_names),
        "anchor_body_name": command.cfg.anchor_body_name,
        "physics_dt": float(base.cfg.sim.dt), "decimation": int(base.cfg.decimation),
        "policy_dt": float(base.cfg.sim.dt * base.cfg.decimation),
        "phase_acceleration": float(command.cfg.phase_acceleration),
        "normalizer_epsilon": float(eps) if eps is not None else None,
        "actor_obs_normalization": bool(getattr(policy, "actor_obs_normalization", False)),
        "contact_force_threshold": float(base.scene["contact_forces"].cfg.force_threshold),
        "reference_sha256": hashlib.sha256(Path(command.cfg.motion_file).read_bytes()).hexdigest(),
        "motion": motion, "control": control,
    }
