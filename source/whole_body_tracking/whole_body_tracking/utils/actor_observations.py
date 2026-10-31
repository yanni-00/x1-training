"""[OMA] Deployment-observable actor contract; critic/reward truth is separate."""
import numpy as np

PROPRIO_FORMAT = "beyondmimic_g1_velocity_proprio_runtime_v1"
FULL_TERMS = [("command",58),("motion_anchor_pos_b",3),("motion_anchor_ori_b",6),
              ("base_lin_vel",3),("base_ang_vel",3),("joint_pos",29),("joint_vel",29),("actions",29)]
PROPRIO_TERMS = [(name,size) for name,size in FULL_TERMS if name not in {"motion_anchor_pos_b","base_lin_vel"}]


def contract(proprioceptive):
    terms=PROPRIO_TERMS if proprioceptive else FULL_TERMS
    return {"version":"beyondmimic_actor_proprio_v1" if proprioceptive else "beyondmimic_actor_full_v1",
            "terms":[name for name,_ in terms],"term_dimensions":[size for _,size in terms],
            "dimension":sum(size for _,size in terms),"history_length":1,
            "requires_position_estimate":not proprioceptive,
            "requires_linear_velocity_estimate":not proprioceptive}


def audit_native_observations(env, policy=None):
    base=env.unwrapped
    proprio=bool(getattr(base.cfg,"oma_proprioceptive_actor",False))
    expected=contract(proprio)
    manager=base.observation_manager
    names=list(manager.active_terms["policy"])
    shape=tuple(manager.group_obs_dim["policy"])
    critic_shape=tuple(manager.group_obs_dim["critic"])
    if names!=expected["terms"] or shape!=(expected["dimension"],) or critic_shape!=(286,):
        raise RuntimeError("Native actor/critic observation contract differs: "+str((names,shape,critic_shape)))
    if proprio:
        cfg=base.cfg.observations.policy
        if cfg.motion_anchor_pos_b is not None or cfg.base_lin_vel is not None:
            raise RuntimeError("Forbidden actor truth/estimator term still active")
    if policy is not None and policy.actor[0].in_features!=expected["dimension"]:
        raise RuntimeError("Native actor network differs from audited observation manager")
    result={**expected,"native_active_terms":names,"native_actor_shape":list(shape),
            "native_critic_shape":list(critic_shape),"passed":True}
    if proprio:
        import hashlib
        from pathlib import Path
        package=Path(__file__).resolve().parents[1]
        paths=["utils/actor_observations.py","utils/runtime_policy.py",
               "tasks/tracking/config/g1/proprio_velocity_env_cfg.py"]
        result["implementation_sha256"]={path:hashlib.sha256(
            (package/path).read_text(encoding="utf-8").encode("utf-8")).hexdigest() for path in paths}
    return result


def proprioceptive_observation(q_ref,dq_ref,relative_rotation,omega,q,dq,last_action,default_q):
    """No actual root position or linear velocity argument; row-major first 2 columns."""
    obs=np.concatenate([q_ref,dq_ref,np.asarray(relative_rotation)[:,:2].reshape(-1),
                        omega,np.asarray(q)-default_q,dq,last_action]).astype(np.float32)
    if obs.shape!=(154,) or not np.isfinite(obs).all():
        raise ValueError("Invalid deployment-observable actor input")
    return obs
