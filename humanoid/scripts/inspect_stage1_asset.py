"""Load the selected F1 asset and exercise one held-target policy period; no learning."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import isaacgym  # noqa: F401 -- required before torch
    from isaacgym import gymapi
    import numpy as np
    import torch
    from humanoid.f1_owner_baseline import apply_owner_baseline
    from humanoid.envs.f1.f1_50hz_config import F1Stage1NoFilter50HzCfg

    cfg = apply_owner_baseline(F1Stage1NoFilter50HzCfg())
    model = cfg.owner_physical_model
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = cfg.sim.dt
    params.substeps = cfg.sim.substeps
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(*cfg.sim.gravity)
    params.physx.use_gpu = True
    params.use_gpu_pipeline = False
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError('Isaac Gym create_sim failed')
    try:
        options = gymapi.AssetOptions()
        for name in ('collapse_fixed_joints', 'replace_cylinder_with_capsule',
                     'flip_visual_attachments', 'fix_base_link', 'armature'):
            setattr(options, name, getattr(cfg.asset, name))
        options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        asset = gym.load_asset(sim, str(ROOT), model['urdf_relative_path'], options)
        if asset is None:
            raise RuntimeError('Isaac Gym failed to load F1 asset')
        names = list(gym.get_asset_dof_names(asset))
        if names != model['joint_names']:
            raise ValueError('Native joint order differs from the F1 contract: '+repr(names))
        props = gym.get_asset_dof_properties(asset)
        for key, expected in [('friction', 0), ('damping', 0.05), ('armature', 0)]:
            if not np.allclose(props[key], expected):
                raise ValueError('Unexpected imported '+key+': '+repr(props[key]))
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(sim, plane)
        env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 1)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*cfg.init_state.pos)
        actor = gym.create_actor(env, asset, pose, 'f1_asset_inspection', 0, cfg.asset.self_collisions)
        gym.set_actor_dof_properties(env, actor, props)
        state = np.zeros(len(names), dtype=gymapi.DofState.dtype)
        target = np.array([cfg.init_state.default_joint_angles[n] for n in names])
        state['pos'] = target
        gym.set_actor_dof_states(env, actor, state, gymapi.STATE_ALL)
        kp = np.array([next(v for k, v in cfg.control.stiffness.items() if k in n) for n in names])
        kd = np.array([next(v for k, v in cfg.control.damping.items() if k in n) for n in names])
        limits = props['effort'] * cfg.safety.torque_limit
        gym.prepare_sim(sim)
        for _ in range(cfg.control.decimation):
            state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            torque = np.clip(kp*(target-state['pos'])-kd*state['vel'], -limits, limits)
            gym.apply_actor_dof_efforts(env, actor, torque.astype(np.float32))
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        if not np.isfinite(state['pos']).all() or not np.isfinite(state['vel']).all():
            raise ValueError('Non-finite native response')
        report = {'status': 'passed', 'scope': 'native_asset_and_one_policy_period_no_training',
                  'environment': {'python': sys.version, 'torch': torch.__version__,
                                  'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)},
                  'joint_names': names, 'imported_dynamics': {k: props[k].tolist() for k in
                      ('damping', 'friction', 'armature', 'effort')},
                  'kp': kp.tolist(), 'kd': kd.tolist(), 'policy_hz': 1/(cfg.sim.dt*cfg.control.decimation),
                  'sim_steps': cfg.control.decimation, 'final_q': state['pos'].tolist(),
                  'final_v': state['vel'].tolist(), 'urdf_sha256': model['urdf_sha256'],
                  'learning_run': False, 'cloud_readiness_claimed': False}
        output = Path(args.output).resolve()
        if ROOT not in output.parents:
            raise ValueError('Inspection output must remain inside the copied training directory')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(report))
    finally:
        gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
