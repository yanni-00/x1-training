"""Apply the identified actuator configuration to the r11 walking task."""

from math import isclose

from humanoid.f1_contract import ROBOT_CONTRACT, nonnegative
from .f1_lateral_swing_direction_100hz import (
    POLICY_DT,
    F1LateralSwingDirection100HzCfg,
    F1LateralSwingDirection100HzEnv,
    F1TargetSpeedRamp100HzCfgPPO,
)

PROFILE = 'stage1_100hz_identified'


class F1Identified100HzCfg(F1LateralSwingDirection100HzCfg):
    class env(F1LateralSwingDirection100HzCfg.env):
        physical_model = 'identified_100hz'


def delay_range(steps):
    if (not isinstance(steps, list) or len(steps) != 2
            or any(type(step) is not int or step < 0 for step in steps)
            or steps[0] > steps[1]):
        raise ValueError('Delay range must contain ordered nonnegative physics steps')
    return list(steps)


def identified_configs(physical):
    if physical.get('profile') != PROFILE:
        raise ValueError('Expected the dedicated identified 100 Hz training profile')
    cfg = F1Identified100HzCfg()
    ppo = F1TargetSpeedRamp100HzCfgPPO()
    cfg.asset.file = '{LEGGED_GYM_ROOT_DIR}/' + physical['urdf_relative_path']
    groups = {name.split('_', 1)[1] for name in ROBOT_CONTRACT['joint_names']}
    for field in ('stiffness', 'damping'):
        gains = physical['control'][field]
        if set(gains) != groups:
            raise ValueError(f'{field} must cover all six F1 joint groups')
        setattr(cfg.control, field, {name: nonnegative(value, f'{field}.{name}')
                                     for name, value in gains.items()})
    delay = physical['action_and_feedback_delay']
    if not isclose(cfg.sim.dt, delay['physics_step_s'], rel_tol=1e-7):
        raise ValueError('Delay ranges do not match the simulation step')
    dr = cfg.domain_rand
    dr.add_lag = True
    dr.randomize_lag_timesteps = True
    dr.randomize_lag_timesteps_perstep = False
    dr.lag_timesteps_range = delay_range(delay['action_steps'])
    dr.add_dof_lag = True
    dr.randomize_dof_lag_timesteps = True
    dr.randomize_dof_lag_timesteps_perstep = False
    dr.dof_lag_timesteps_range = delay_range(delay['joint_feedback_steps'])
    delivery = physical['delivery']
    if (not delivery['enabled']
            or set(delivery['joint_names']) != {name for name in ROBOT_CONTRACT['joint_names'] if '_ankle_' in name}):
        raise ValueError('The delivery filter must cover both axes of both ankles')
    train = physical['training']
    cfg.env.num_envs = train['num_envs']
    ppo.seed = train['seed']
    ppo.runner.max_iterations = train['max_iterations']
    ppo.runner.experiment_name = PROFILE
    return cfg, ppo


def register_identified_task(physical):
    from humanoid.algo.ppo.actor_critic_f1_equivariant import register_policy_class
    from humanoid.envs import task_registry

    cfg, ppo = identified_configs(physical)
    register_policy_class()
    task_registry.register(physical['training']['task'], F1LateralSwingDirection100HzEnv, cfg, ppo)
    return task_registry
