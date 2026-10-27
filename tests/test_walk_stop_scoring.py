"""Negative control: frozen reference with a sliding robot must fail."""
import csv,importlib.util,json,tempfile
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'scripts/score_walk_stop.py';spec=importlib.util.spec_from_file_location('score',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
stages=[{'name':'stand','go':False,'duration_s':2},{'name':'walk','go':True,'duration_s':2},{'name':'stop','go':False,'duration_s':2},{'name':'restart','go':True,'duration_s':2}]
protocol={'stages':stages,'reporting_settle_s':0}
# Synthetic fixture thresholds only; these are not G1 acceptance requirements.
gates={'stop_speed_max_mps':.1,'stop_drift_max_m':.1,'tilt_max_rad':.2,'go_velocity_rmse_max_mps':.1,'go_path_min_m':.1,'both_contact_fraction_min':1,'contact_foot_speed_max_mps':.1}
with tempfile.TemporaryDirectory() as d:
 d=Path(d);manifest={'protocol':protocol,'policy_dt':.5,'completed_protocol':True,'contact_force_threshold_source':10};(d/'evaluation_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
 rows=[]
 for i,s in enumerate(stages):
  for j in range(4):
   moving=s['go'];r={'step':len(rows),'time_s':len(rows)*.5,'stage':s['name'],'requested_go':int(moving),'reference_exhausted':0,'reference_rate':int(moving),'terminated':0,'root_x':j*.25 if moving else 0,'root_y':0,'root_vx':.5 if moving else 0,'root_vy':0,'reference_vx':.5 if moving else 0,'reference_vy':0,'root_tilt_rad':0}
   for side in ['left','right']:
    for k in ['fx','fy','fz','vx','vy']:r[f'{side}_foot_{k}']=100 if k=='fz' else 0
   rows.append(r)
 def save():
  with (d/'joints.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save();r=m.score(d,gates);assert r['quantitative_pass'] and not r['run_acceptance'], 'Unreviewed video must not pass'
 rows[2]['time_s']+=.01
 save();r=m.score(d,gates);assert not r['structural_checks']['timestamps_match_policy_clock'] and not r['quantitative_pass'], 'Bad timebase must not pass'
 rows[2]['time_s']-=.01
 save()
 r=m.score(d);assert not r['quantitative_pass'] and not r['run_acceptance'], 'Missing frozen gates must not pass'
 for row in rows:
  if row['stage']=='stop':row['root_vx']=.8;row['root_x']=row['step']*.4
 save();r=m.score(d,gates);assert r['structural_checks']['stop_reference_hold_reached'] and not r['quantitative_pass'], 'Reference hold hid actual sliding'
 for row in rows:
  if row['stage']=='restart':row['reference_exhausted']=1
 save();r=m.score(d,gates);assert not r['structural_checks']['no_reference_exhaustion_during_go'], 'Finite playlist exhaustion masqueraded as locomotion'
print(json.dumps({'synthetic_good_motion_metrics':True,'video_required_for_run_acceptance':True,'frozen_gates_required':True,'actual_sliding_rejected_despite_reference_hold':True,'reference_exhaustion_during_go_rejected':True,'malformed_timebase_rejected':True}))
