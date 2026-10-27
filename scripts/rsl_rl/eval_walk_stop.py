"""[OMA] Continuous actual-state CSV + video protocol; no post-failure restart."""
import argparse, json, math, sys
from pathlib import Path
from isaaclab.app import AppLauncher
import cli_args

p=argparse.ArgumentParser()
p.add_argument('--checkpoint_file',required=True);p.add_argument('--motion_file',required=True)
p.add_argument('--protocol_file',required=True);p.add_argument('--output_dir',required=True)
p.add_argument('--video',action='store_true');p.add_argument('--task',default='Tracking-WalkStop-G1-v0')
cli_args.add_rsl_rl_args(p);AppLauncher.add_app_launcher_args(p)
a,hydra_args=p.parse_known_args()
if a.video:a.enable_cameras=True
sys.argv=[sys.argv[0]]+hydra_args
app=AppLauncher(a);simulation_app=app.app

import csv, os, torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import whole_body_tracking.tasks


def as_list(x):return x.detach().cpu().reshape(-1).tolist()


@hydra_task_config(a.task,'rsl_rl_cfg_entry_point')
def main(env_cfg,agent_cfg):
    protocol=json.loads(Path(a.protocol_file).read_text(encoding='utf-8'))
    stages=protocol['stages']
    if len({s['name'] for s in stages})!=len(stages):raise ValueError('Stage names must be unique')
    if any(s['duration_s']<=0 for s in stages):raise ValueError('Stage durations must be positive')
    agent_cfg=cli_args.update_rsl_rl_cfg(agent_cfg,a)
    env_cfg.scene.num_envs=1;env_cfg.seed=int(protocol['seed']);agent_cfg.seed=env_cfg.seed
    env_cfg.sim.device=a.device if a.device is not None else env_cfg.sim.device
    dt=env_cfg.sim.dt*env_cfg.decimation
    stage_steps=[round(s['duration_s']/dt) for s in stages]
    if any(abs(n*dt-s['duration_s'])>1e-7 for n,s in zip(stage_steps,stages)):
        raise ValueError('Protocol stages must contain a whole number of policy steps')
    total_steps=sum(stage_steps);env_cfg.episode_length_s=(total_steps+2)*dt
    env_cfg.commands.motion.motion_file=str(Path(a.motion_file).resolve())
    env_cfg.commands.motion.evaluation_start_support=protocol['initial_support_index']
    if protocol.get('nominal',False):
        env_cfg.observations.policy.enable_corruption=False
        env_cfg.commands.motion.pose_range={};env_cfg.commands.motion.velocity_range={}
        env_cfg.commands.motion.joint_position_range=(0.,0.)
        for name in ['physics_material','add_joint_default_pos','base_com','push_robot']:
            setattr(env_cfg.events,name,None)
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    raw=gym.make(a.task,cfg=env_cfg,render_mode='rgb_array' if a.video else None)
    if a.video:
        raw=gym.wrappers.RecordVideo(raw,video_folder=str(out/'videos'),step_trigger=lambda step:step==0,video_length=total_steps+1,disable_logger=True)
    env=RslRlVecEnvWrapper(raw);base=env.unwrapped
    runner=OnPolicyRunner(env,agent_cfg.to_dict(),log_dir=None,device=agent_cfg.device)
    runner.load(str(Path(a.checkpoint_file).resolve()))
    policy=runner.get_inference_policy(device=base.device)
    motion=base.command_manager.get_term('motion');robot=motion.robot
    sensor=base.scene['contact_forces'];foot_names=['left_ankle_roll_link','right_ankle_roll_link']
    foot_ids=robot.find_bodies(foot_names,preserve_order=True)[0]
    sensor_ids=sensor.find_bodies(foot_names,preserve_order=True)[0]
    motion.request(False)
    obs,_=env.get_observations()
    if tuple(obs.shape)!=(1,160):raise ValueError(f'Unexpected policy observation shape: {tuple(obs.shape)}')
    header=['step','time_s','stage','requested_go','reference_phase','reference_rate','reference_exhausted','terminated','root_x','root_y','root_z','root_vx','root_vy','root_vz','root_tilt_rad','reference_vx','reference_vy','reference_vz']
    for side in ['left','right']:
        header += [f'{side}_foot_{v}' for v in ['x','y','z','vx','vy','vz','fx','fy','fz']]
    for prefix in ['q','dq','q_ref','dq_ref','action']:
        header += [f'{prefix}__{name}' for name in robot.joint_names]
    manifest={'checkpoint':str(Path(a.checkpoint_file).resolve()),'motion_file':str(Path(a.motion_file).resolve()),'protocol':protocol,'policy_dt':dt,'planned_steps':total_steps,'video_requested':a.video,'observation_dim':160,'joint_names':robot.joint_names,'body_names':robot.body_names,'contact_force_threshold_source':float(sensor.cfg.force_threshold),'automatic_reset_policy':'Abort at first returned done. CSV records PRE-step physical state so reset poses cannot masquerade as recovery.','posture_success':None,'behavior_acceptance':False}
    (out/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    completed=0;terminated=False
    try:
        with (out/'joints.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(header)
            for stage,nsteps in zip(stages,stage_steps):
                motion.request(bool(stage['go']))
                for _ in range(nsteps):
                    with torch.inference_mode():
                        action=policy(obs)
                        root=robot.data.root_pos_w[0];velocity=robot.data.root_lin_vel_w[0]
                        quat=robot.data.root_quat_w[0]
                        # R_zz for WXYZ quaternion: physical pelvis tilt from world vertical.
                        tilt=math.acos(float((1-2*(quat[1]**2+quat[2]**2)).clamp(-1,1)))
                        row=[completed,completed*dt,stage['name'],int(stage['go']),float(motion.clock.phase[0]),float(motion.clock.rate[0]),int(motion.clock.exhausted[0]),0,*as_list(root),*as_list(velocity),tilt,*as_list(motion.body_lin_vel_w[0,0])]
                        for b,c in zip(foot_ids,sensor_ids):row += [*as_list(robot.data.body_pos_w[0,b]),*as_list(robot.data.body_lin_vel_w[0,b]),*as_list(sensor.data.net_forces_w[0,c])]
                        row += [*as_list(robot.data.joint_pos[0]),*as_list(robot.data.joint_vel[0]),*as_list(motion.joint_pos[0]),*as_list(motion.joint_vel[0]),*as_list(action[0])]
                        obs,_,done,info=env.step(action)
                        terminated=bool(done[0]);row[7]=int(terminated);writer.writerow(row);completed+=1
                    if terminated:break
                f.flush()
                if terminated:break
    finally:
        env.close()
        manifest.update(completed_steps=completed,terminated=terminated,completed_protocol=completed==total_steps and not terminated)
        (out/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('[OMA] Evaluation saved: '+str(out),flush=True)

if __name__=='__main__':
    try:main()
    finally:simulation_app.close()
