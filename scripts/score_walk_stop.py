"""[OMA] Score measured robot motion. Reference hold alone never passes behavior."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

REQUIRED_GATES=['stop_speed_max_mps','stop_drift_max_m','tilt_max_rad','go_velocity_rmse_max_mps','go_path_min_m','both_contact_fraction_min','contact_foot_speed_max_mps']


def score(directory,gates=None):
    directory=Path(directory);manifest=json.loads((directory/'evaluation_manifest.json').read_text(encoding='utf-8'))
    protocol=manifest['protocol'];dt=manifest['policy_dt']
    with (directory/'joints.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Empty actual-state CSV')
    numeric=[k for k in rows[0] if k!='stage']
    finite=all(math.isfinite(float(row[k])) for row in rows for k in numeric)
    if not finite:raise ValueError('Non-finite actual-state telemetry')
    expected=[]
    for stage in protocol['stages']:
        count=round(stage['duration_s']/dt)
        if abs(count*dt-stage['duration_s'])>1e-7:raise ValueError('Stage duration not aligned to policy dt')
        expected += [stage['name']]*count
    if gates is not None:
        for name in REQUIRED_GATES:
            if name in gates and (not math.isfinite(float(gates[name])) or gates[name]<0):
                raise ValueError('Invalid frozen gate: '+name)
        for key in ['stop_yaw_rate_max_radps','go_yaw_mean_rate_error_max_radps']:
            if key in gates and (not math.isfinite(float(gates[key])) or gates[key]<0):raise ValueError('Invalid frozen angular gate: '+key)
        if 'both_contact_fraction_min' in gates and gates['both_contact_fraction_min']>1:
            raise ValueError('Invalid contact fraction gate')
    checks={'row_count_matches_protocol':len(rows)==len(expected),'stage_schedule_matches_protocol':[r['stage'] for r in rows]==expected,'step_sequence_is_continuous':[int(r['step']) for r in rows]==list(range(len(rows))),'timestamps_match_policy_clock':all(abs(float(r['time_s'])-i*dt)<=1e-9 for i,r in enumerate(rows)),'completed_protocol':bool(manifest['completed_protocol']),'no_environment_termination':all(int(row['terminated'])==0 for row in rows),'all_stages_recorded':set(s['name'] for s in protocol['stages']).issubset(set(r['stage'] for r in rows)),'no_reference_exhaustion_during_go':all(not(int(r['requested_go']) and int(r['reference_exhausted'])) for r in rows)}
    metrics={};gate_checks={};force_limit=float(manifest['contact_force_threshold_source'])
    for spec in protocol['stages']:
        data=[r for r in rows if r['stage']==spec['name']]
        if not data:continue
        t=np.asarray([float(r['time_s']) for r in data])
        # STOP quality is assessed over a terminal stable-hold window. Its
        # response latency is reported separately, without a fixed deadline.
        hold_s=spec.get('stability_window_s')
        if hold_s is not None:
            if spec['go'] or hold_s<=0 or hold_s>spec['duration_s']:raise ValueError('Invalid stable-hold window')
            tail=t>=t[0]+spec['duration_s']-hold_s-1e-9
        else:
            tail=t>=t[0]+spec.get('reporting_settle_s',protocol['reporting_settle_s'])
        checks[spec['name']+'_has_post_settle_samples']=bool(tail.any())
        if not tail.any():
            metrics[spec['name']]={'recorded_duration_s':float(len(data)*dt),'tail_samples':0,'status':'No post-settle data; the run cannot pass'}
            continue
        pos=np.asarray([[float(r['root_x']),float(r['root_y'])] for r in data]);v=np.asarray([[float(r['root_vx']),float(r['root_vy'])] for r in data]);ref=np.asarray([[float(r['reference_vx']),float(r['reference_vy'])] for r in data]);speed=np.linalg.norm(v,axis=1)
        tilt=np.asarray([float(r['root_tilt_rad']) for r in data]);rates=np.asarray([float(r['reference_rate']) for r in data])
        forces=np.asarray([[[float(r[f'{s}_foot_{k}']) for k in ['fx','fy','fz']] for s in ['left','right']] for r in data]);contact=np.linalg.norm(forces,axis=-1)>force_limit
        foot_v=np.asarray([[[float(r[f'{s}_foot_{k}']) for k in ['vx','vy']] for s in ['left','right']] for r in data]);slip=np.linalg.norm(foot_v,axis=-1)
        contact_slip=slip[tail][contact[tail]]
        m={'recorded_duration_s':float(len(data)*dt),'tail_samples':int(tail.sum()),'actual_speed_mean_mps':float(speed[tail].mean()),'actual_speed_max_mps':float(speed[tail].max()),'actual_planar_drift_m':float(np.linalg.norm(pos[tail]-pos[tail][0],axis=1).max()),'actual_planar_path_m':float(np.linalg.norm(np.diff(pos,axis=0),axis=1).sum()),'velocity_tracking_rmse_mps':float(np.sqrt(np.mean(np.sum((v-ref)**2,axis=1)))),'tilt_max_rad':float(tilt.max()),'both_contact_fraction':float(contact[tail].all(axis=1).mean()),'contact_foot_speed_max_mps':float(contact_slip.max()) if len(contact_slip) else None,'reference_rate_tail_max':float(rates[tail].max())}
        yaw_rate=None
        if gates is not None and 'stop_yaw_rate_max_radps' in gates:
            yaw_rate=np.asarray([abs(float(r['actual_yaw_rate'])) for r in data])
            m['actual_yaw_rate_tail_max_radps']=float(yaw_rate[tail].max())
        if not spec['go']:
            m['reference_hold_response_s']=None
            moving=np.flatnonzero(rates!=0)
            held_from=int(moving[-1]+1) if len(moving) else 0
            if held_from<len(data):m['reference_hold_response_s']=float(t[held_from]-t[0])
            m['stable_stop_response_s']=None
            m['response_definition']='Earliest suffix satisfying all frozen stop gates and reference hold through the stage end, with the required minimum stable-hold duration. Null means not observed; no hard latency threshold.'
            if hold_s is not None and gates is not None and all(k in gates for k in REQUIRED_GATES):
                minimum=round(hold_s/dt)
                for j in range(max(0,len(data)-minimum+1)):
                    touching_slip=slip[j:][contact[j:]]
                    stable=(np.all(rates[j:]==0) and speed[j:].max()<=gates['stop_speed_max_mps']
                        and np.linalg.norm(pos[j:]-pos[j],axis=1).max()<=gates['stop_drift_max_m']
                        and tilt[j:].max()<=gates['tilt_max_rad']
                        and contact[j:].all(axis=1).mean()>=gates['both_contact_fraction_min']
                        and len(touching_slip)>0 and touching_slip.max()<=gates['contact_foot_speed_max_mps'])
                    if yaw_rate is not None:stable=stable and yaw_rate[j:].max()<=gates['stop_yaw_rate_max_radps']
                    if stable:
                        m['stable_stop_response_s']=float(t[j]-t[0]);break
        metrics[spec['name']]=m
        if not spec['go']:checks[spec['name']+'_reference_hold_reached']=bool(np.all(rates[tail]==0))
        if gates is not None and all(k in gates for k in REQUIRED_GATES):
            stage_gates={'tilt':m['tilt_max_rad']<=gates['tilt_max_rad']}
            if spec['go']:
                stage_gates.update(go_velocity=m['velocity_tracking_rmse_mps']<=gates['go_velocity_rmse_max_mps'],go_progress=m['actual_planar_path_m']>=gates['go_path_min_m'])
            else:
                stage_gates.update(stop_speed=m['actual_speed_max_mps']<=gates['stop_speed_max_mps'],stop_drift=m['actual_planar_drift_m']<=gates['stop_drift_max_m'],both_contact=m['both_contact_fraction']>=gates['both_contact_fraction_min'],contact_slip=m['contact_foot_speed_max_mps'] is not None and m['contact_foot_speed_max_mps']<=gates['contact_foot_speed_max_mps'])
            if not spec['go'] and yaw_rate is not None:
                stage_gates['stop_yaw_rate']=m['actual_yaw_rate_tail_max_radps']<=gates['stop_yaw_rate_max_radps']
            gate_checks[spec['name']]=stage_gates
    complete_gates=gates is not None and all(k in gates for k in REQUIRED_GATES)
    quantitative_pass=bool(complete_gates and all(checks.values()) and len(gate_checks)==len(protocol['stages']) and all(all(g.values()) for g in gate_checks.values()))
    video_review_path=directory/'video_review.json';video_pass=False
    if video_review_path.is_file():
        review=json.loads(video_review_path.read_text(encoding='utf-8'))
        videos=list((directory/'videos').glob('*.mp4'))
        matching=any(hashlib.sha256(v.read_bytes()).hexdigest()==review.get('source_video_sha256') for v in videos)
        video_pass=bool(review.get('reviewed') and review.get('passed') and matching)
    return {'scope':'one uninterrupted evaluation run; not proof of independent training reproducibility or full-loop completion','csv_sha256':hashlib.sha256((directory/'joints.csv').read_bytes()).hexdigest(),'protocol':protocol,'structural_checks':checks,'stage_metrics':metrics,'frozen_gates_supplied':complete_gates,'quantitative_gate_checks':gate_checks,'quantitative_pass':quantitative_pass,'video_review_pass':video_pass,'run_acceptance':quantitative_pass and video_pass}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation_dir',required=True);p.add_argument('--gates');a=p.parse_args()
    gates=json.loads(Path(a.gates).read_text(encoding='utf-8')) if a.gates else None
    result=score(a.evaluation_dir,gates);out=Path(a.evaluation_dir)/'metrics.json';out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
