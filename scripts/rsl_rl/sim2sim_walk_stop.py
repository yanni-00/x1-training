#!/usr/bin/env python3
"""BeyondMimic task adapter for the established G1 server MuJoCo play workflow.
Reuses G1's EGL rendering, CPU PyTorch inference and per-substep clipped PD loop.
Only simulation is used. Exact references/control parameters come from the model.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import argparse, csv, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import torch
import mujoco

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


# Same quaternion convention and formula as unitree_rl_lab_train/sim2sim_play.py.
def matrix(q):
    w,x,y,z = q
    return np.asarray([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                       [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                       [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def sample(data, phase, quaternion=False):
    lo = min(int(phase),len(data)-1); hi = min(lo+1,len(data)-1); a = phase-lo
    left,right = data[lo],data[hi]
    if quaternion:
        right = np.where(np.sum(left*right,axis=-1,keepdims=True)<0,-right,right)
        result = left+(right-left)*a
        return result/np.linalg.norm(result,axis=-1,keepdims=True)
    return left+(right-left)*a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True);p.add_argument("--protocol",required=True)
    p.add_argument("--output_dir",required=True);p.add_argument("--mjcf")
    p.add_argument("--no-video",action="store_true")
    a = p.parse_args()
    checkpoint = torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    runtime = checkpoint["infos"]["oma_runtime"]
    module = ROOT/"source/whole_body_tracking/whole_body_tracking"
    actor = load_module("runtime_policy",module/"utils/runtime_policy.py").RuntimeActor(checkpoint).eval()
    Clock = load_module("reference_clock",module/"tasks/tracking/mdp/reference_clock.py").SupportPhaseClock
    motion = {k:v.cpu().numpy() for k,v in runtime["motion"].items()}
    control = {k:v.cpu().numpy() for k,v in runtime["control"].items()}
    names = runtime["joint_names"];body_names = runtime["body_names"]
    protocol = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    dt = runtime["policy_dt"]; physics_dt = runtime["physics_dt"]
    if abs(float(motion["fps"].reshape(-1)[0])*dt-1)>1e-7:raise ValueError("Reference/control rate mismatch")
    counts = [round(s["duration_s"]/dt) for s in protocol["stages"]]
    if any(n<=0 or abs(n*dt-s["duration_s"])>1e-7 for n,s in zip(counts,protocol["stages"])):
        raise ValueError("Protocol duration is not aligned to policy time")
    total = sum(counts)
    clock = Clock(1,1/dt,motion["support_frames"],acceleration=runtime["phase_acceleration"])
    support_index = protocol["initial_support_index"]
    if not 0<=support_index<len(clock.support):raise ValueError("Invalid initial support")
    clock.reset(torch.tensor([0]),clock.support[support_index].reshape(1))
    mjcf = Path(a.mjcf) if a.mjcf else module/"assets/unitree_description/mjcf/g1.xml"
    model = mujoco.MjModel.from_xml_path(str(mjcf.resolve()));model.opt.timestep=physics_dt
    data = mujoco.MjData(model)
    joint_ids = np.asarray([model.joint(n).id for n in names])
    qids = model.jnt_qposadr[joint_ids];vids = model.jnt_dofadr[joint_ids]
    if len(set(qids.tolist()))!=29:raise ValueError("MuJoCo joint map is not bijective")
    model.dof_armature[vids] = control["armature"]
    pelvis = body_names.index("pelvis");anchor = body_names.index(runtime["anchor_body_name"])
    anchor_id = model.body(runtime["anchor_body_name"]).id
    foot_names = ["left_ankle_roll_link","right_ankle_roll_link"]
    foot_ids = [model.body(n).id for n in foot_names]
    phase = float(clock.phase[0])
    data.qpos[:3] = sample(motion["body_pos_w"],phase)[pelvis]
    data.qpos[3:7] = sample(motion["body_quat_w"],phase,True)[pelvis]
    data.qpos[qids] = sample(motion["joint_pos"],phase);data.qvel[:]=0
    mujoco.mj_forward(model,data)
    out = Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    header = ["step","time_s","stage","requested_go","reference_phase","reference_rate","reference_exhausted","terminated","root_x","root_y","root_z","root_vx","root_vy","root_vz","root_tilt_rad","reference_vx","reference_vy","reference_vz"]
    for side in ["left","right"]:header += [f"{side}_foot_{k}" for k in ["x","y","z","vx","vy","vz","fx","fy","fz"]]
    for prefix in ["q","dq","q_ref","dq_ref","action"]:header += [f"{prefix}__{n}" for n in names]
    # Abort limits are the existing BeyondMimic offline verifier settings.
    abort_height = protocol["abort_pelvis_height_m"];abort_tilt = protocol["abort_tilt_rad"]
    manifest = {"checkpoint":str(Path(a.checkpoint).resolve()),"checkpoint_sha256":hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        "protocol":protocol,"policy_dt":dt,"physics_dt":physics_dt,"planned_steps":total,"joint_names":names,"body_names":body_names,
        "observation_dim":160,"state_source":"MuJoCo truth; kinematic-state-estimator transfer remains a separate check",
        "reference_source":"Exact training motion tensors serialized with model","reference_sha256":runtime["reference_sha256"],
        "contact_force_threshold_source":runtime["contact_force_threshold"],"mjcf_sha256":hashlib.sha256(mjcf.read_bytes()).hexdigest(),
        "mujoco_version":mujoco.__version__,"video_requested":not a.no_video,"completed_steps":0,"completed_protocol":False,"terminated":False,"behavior_acceptance":False,
        "automatic_reset_policy":"Initialize once; abort at first physical failure; no reset during any GO/STOP switch.",
        "abort_limits_source":"1_G1_deploy-main/tools/verify_beyondmimic.py; explicit values frozen in protocol"}
    manifest_path = out/"evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    renderer=writer=None
    if not a.no_video:
        import imageio.v2 as imageio
        (out/"videos").mkdir(exist_ok=True)
        model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
        renderer=mujoco.Renderer(model,720,1280)
        camera=mujoco.MjvCamera();camera.distance=3.2;camera.azimuth=130.;camera.elevation=-12.
        writer=imageio.get_writer(str(out/"videos/play.mp4"),fps=1/dt,codec="libx264")
    step=0;last_action=np.zeros(29,dtype=np.float32);terminated=False;reason=None
    try:
        with (out/"joints.csv").open("w",newline="",encoding="utf-8") as f:
            csv_writer=csv.writer(f);csv_writer.writerow(header)
            for stage,nsteps in zip(protocol["stages"],counts):
                for _ in range(nsteps):
                    mujoco.mj_forward(model,data)
                    phase=float(clock.phase[0]);rate=float(clock.rate[0])
                    q_ref=sample(motion["joint_pos"],phase);dq_ref=sample(motion["joint_vel"],phase)*rate
                    pos_ref=sample(motion["body_pos_w"],phase);quat_ref=sample(motion["body_quat_w"],phase,True)
                    anchor_r=matrix(data.xquat[anchor_id]);root_r=matrix(data.qpos[3:7])
                    obs=np.concatenate([q_ref,dq_ref,anchor_r.T@(pos_ref[anchor]-data.xpos[anchor_id]),
                        (anchor_r.T@matrix(quat_ref[anchor]))[:,:2].reshape(-1),root_r.T@data.qvel[:3],data.qvel[3:6],
                        data.qpos[qids]-control["default_joint_pos"],data.qvel[vids],last_action]).astype(np.float32)
                    if obs.shape!=(160,) or not np.isfinite(obs).all():raise ValueError("Invalid actual-state observation")
                    with torch.inference_mode():action=actor(torch.from_numpy(obs[None]))[0].numpy()
                    if not np.isfinite(action).all():raise ValueError("Non-finite model action")
                    tilt=math.acos(float(np.clip(root_r[2,2],-1,1)))
                    row=[step,step*dt,stage["name"],int(stage["go"]),phase,rate,int(clock.exhausted[0]),0,
                         *data.qpos[:3].tolist(),*data.qvel[:3].tolist(),tilt,*((sample(motion["body_lin_vel_w"],phase)[pelvis])*rate).tolist()]
                    forces=np.zeros((2,3));force=np.zeros(6)
                    for index in range(data.ncon):
                        contact=data.contact[index];b1=int(model.geom_bodyid[contact.geom1]);b2=int(model.geom_bodyid[contact.geom2])
                        for foot_i,foot in enumerate(foot_ids):
                            if (b1==foot and b2==0) or (b2==foot and b1==0):
                                mujoco.mj_contactForce(model,data,index,force)
                                forces[foot_i] += contact.frame.reshape(3,3).T@force[:3]*(1 if b2==foot else -1)
                    for foot_i,foot in enumerate(foot_ids):
                        velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,foot,velocity,0)
                        row += [*data.xpos[foot].tolist(),*velocity[3:].tolist(),*forces[foot_i].tolist()]
                    row += [*data.qpos[qids].tolist(),*data.qvel[vids].tolist(),*q_ref.tolist(),*dq_ref.tolist(),*action.tolist()]
                    if writer is not None:
                        camera.lookat[:]=data.qpos[:3];renderer.update_scene(data,camera);writer.append_data(renderer.render())
                    target=control["default_joint_pos"]+control["action_scale"]*action
                    for _ in range(runtime["decimation"]):
                        torque=np.clip(control["kp"]*(target-data.qpos[qids])-control["kd"]*data.qvel[vids],-control["effort_limit"],control["effort_limit"])
                        data.qfrc_applied[:]=0;data.qfrc_applied[vids]=torque;mujoco.mj_step(model,data)
                    last_action=action.copy();clock.step(torch.tensor([bool(stage["go"])]),dt)
                    tilt_after=math.acos(float(np.clip(matrix(data.qpos[3:7])[2,2],-1,1)))
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():terminated=True;reason="non_finite_physics"
                    elif data.qpos[2]<abort_height:terminated=True;reason="pelvis_height_abort"
                    elif tilt_after>abort_tilt:terminated=True;reason="pelvis_tilt_abort"
                    row[7]=int(terminated);csv_writer.writerow(row);step+=1
                    if terminated:break
                f.flush()
                if terminated:break
    finally:
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
        manifest.update(completed_steps=step,completed_protocol=step==total and not terminated,terminated=terminated,termination_reason=reason)
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output_dir":str(out),"completed_steps":step,"completed_protocol":manifest["completed_protocol"],"terminated":terminated,"reason":reason}))


if __name__=="__main__":main()
