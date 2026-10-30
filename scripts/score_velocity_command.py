"""[OMA] Score actual robot velocity against user command, never braked reference."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from score_walk_stop import score


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation_dir',required=True);p.add_argument('--gates',required=True)
    a=p.parse_args();directory=Path(a.evaluation_dir)
    gates=json.loads(Path(a.gates).read_text(encoding='utf-8'))
    manifest=json.loads((directory/'evaluation_manifest.json').read_text())
    if not manifest.get('velocity_command_mode'):
        raise ValueError('Legacy replay is not velocity-command evidence')
    result=score(directory,gates)
    with (directory/'joints.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No actual-state telemetry')
    for stage in manifest['protocol']['stages']:
        data=[row for row in rows if row['stage']==stage['name']]
        if not data:continue
        user=np.array([[float(row['raw_vx']),float(row['raw_vy'])] for row in data])
        enabled=np.array([int(row['walk_enabled']) for row in data],dtype=bool)
        user[~enabled]=0
        actual=np.array([[float(row['actual_body_vx']),float(row['actual_body_vy'])] for row in data])
        rmse=float(np.sqrt(np.mean(np.sum((actual-user)**2,axis=1))))
        yaw=np.array([float(row['actual_yaw'])-float(row['target_yaw']) for row in data])
        yaw=np.arctan2(np.sin(yaw),np.cos(yaw))
        m=result['stage_metrics'][stage['name']]
        if 'velocity_tracking_rmse_mps' in m:
            m['reference_velocity_tracking_rmse_mps']=m.pop('velocity_tracking_rmse_mps')
        m['user_command_velocity_rmse_mps']=rmse
        m['heading_error_max_rad']=float(np.abs(yaw).max())
        m['reference_cycle_count_start']=int(data[0]['reference_cycles'])
        m['reference_cycle_count_end']=int(data[-1]['reference_cycles'])
        checks=result['quantitative_gate_checks'].get(stage['name'])
        if checks is not None:
            checks['heading']=m['heading_error_max_rad']<=gates['heading_error_max_rad']
            if stage['go']:
                checks['go_velocity']=rmse<=gates['go_velocity_rmse_max_mps']
        expected=np.asarray(stage['velocity'],dtype=np.float32)
        result['structural_checks'][stage['name']+'_raw_command_matches_protocol']=bool(all(
            np.allclose([float(row[k]) for k in ['raw_vx','raw_vy','raw_yaw_rate']],expected,rtol=0,atol=1e-7) for row in data))
    result['structural_checks']['no_silent_command_saturation']=all(int(r['command_saturated'])==0 for r in rows)
    result['quantitative_pass']=bool(result['frozen_gates_supplied'] and all(result['structural_checks'].values()) and
        len(result['quantitative_gate_checks'])==len(manifest['protocol']['stages']) and
        all(all(values.values()) for values in result['quantitative_gate_checks'].values()))
    result['run_acceptance']=result['quantitative_pass'] and result['video_review_pass']
    result['scope']='Forward velocity/stop integration only; full joystick directions, naturalness review and independent training remain open'
    result['velocity_target_source']='Raw user command, masked only by explicitly logged walking-mode enable; never the braked reference'
    result['main_joystick_goal_accepted']=False
    result['gates_sha256']=hashlib.sha256(Path(a.gates).read_bytes()).hexdigest()
    (directory/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
