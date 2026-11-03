"""Real-parent continuation protection and live-manager checks for one scalar reweight."""
from pathlib import Path
import ast,copy,importlib.util,json,os,subprocess,sys,types,torch
R=Path(__file__).resolve().parents[1];P=R/'source/whole_body_tracking/whole_body_tracking'
sp=importlib.util.spec_from_file_location('oma_checkpoint',P/'utils/oma_checkpoint.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
parent=torch.load(sys.argv[1],map_location='cpu',weights_only=True)['infos']['oma_runtime']
assert parent['reward_parameters']['feet_contact_planar_speed']['weight']==-.2
before=copy.deepcopy(parent['reward_parameters']['feet_contact_planar_speed']);after=dict(before,weight=-.3)
delta={'feet_contact_planar_speed':{'kind':'strengthen_existing_contact_cost_v1','before':before,'after':after}}
current=copy.deepcopy(parent);current['reward_parameters']['feet_contact_planar_speed']=after
m.validate_runtime_continuation(parent,current,delta)
def reject(fn):
 try:fn()
 except RuntimeError:return
 raise AssertionError('Unsafe continuation/audit accepted')
reject(lambda:m.validate_runtime_continuation(parent,current))
# The old addition-only request must not authorize changing an existing coefficient.
reject(lambda:m.validate_runtime_continuation(parent,current,{'feet_contact_planar_speed':after}))
for key,value in [('weight',-.1),('force_threshold',20.),('body_names',['left_ankle_roll_link']),('statistic','other')]:
 d=copy.deepcopy(delta);c=copy.deepcopy(current);d['feet_contact_planar_speed']['after'][key]=value;c['reward_parameters']['feet_contact_planar_speed'][key]=value
 reject(lambda:m.validate_runtime_continuation(parent,c,d))
d=copy.deepcopy(delta);d['feet_contact_planar_speed']['before']['weight']=-.1;reject(lambda:m.validate_runtime_continuation(parent,current,d))
c=copy.deepcopy(current);c['reward_parameters']['motion_body_lin_vel']['std']=.5;reject(lambda:m.validate_runtime_continuation(parent,c,delta))
c=copy.deepcopy(current);c['control']['kp']*=2;reject(lambda:m.validate_runtime_continuation(parent,c,delta))
c=copy.deepcopy(current);c['reward_parameters']['motion_pelvis_lin_vel']={'weight':.5};reject(lambda:m.validate_runtime_continuation(parent,c,delta))
# Normal same-parent continuation remains valid with no authorization.
m.validate_runtime_continuation(parent,copy.deepcopy(parent))
# Audit the actual function code/source and live manager; no simulator install needed.
f=P/'tasks/tracking/mdp/rewards.py';tree=ast.parse(f.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='feet_contact_planar_speed');mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[]);ast.fix_missing_locations(mod);ns={};exec(compile(mod,str(f),'exec'),ns);S=types.SimpleNamespace
# A manager owns a deepcopy whose body IDs are resolved; declarations stay unresolved.
class Scene(dict):
 pass
asset=S(body_names=['torso_link','right_ankle_roll_link','left_ankle_roll_link'],data=S(body_lin_vel_w=torch.zeros(2,3,3)))
sensor=S(body_names=['left_ankle_roll_link','torso_link','right_ankle_roll_link'],data=S(net_forces_w=torch.zeros(2,3,3)))
asset.data.body_lin_vel_w[0,2,0]=.1;asset.data.body_lin_vel_w[1,1,1]=.2
sensor.data.net_forces_w[0,0,2]=20.;sensor.data.net_forces_w[1,2,2]=20.
scene=Scene(robot=asset,contact_forces=sensor);scene.sensors={'contact_forces':sensor}
params={'force_threshold':10.,'asset_cfg':S(name='robot',body_names=before['body_names'],preserve_order=True,body_ids=slice(None)),'sensor_cfg':S(name='contact_forces',body_names=before['body_names'],preserve_order=True,body_ids=slice(None))}
term=S(weight=-.3,params=params,func=ns['feet_contact_planar_speed']);live=copy.deepcopy(term)
live.params['asset_cfg'].body_ids=[2,1];live.params['sensor_cfg'].body_ids=[0,2]
assert live.params!=term.params,'Regression must reproduce the formerly rejected valid lifecycle'
manager=S(active_terms=['feet_contact_planar_speed'],get_term_cfg=lambda name:live)
base=S(scene=scene,cfg=S(rewards=S(motion_body_lin_vel=S(weight=1.,params={'std':1.}),feet_contact_planar_speed=term,motion_pelvis_lin_vel=S(weight=0))),reward_manager=manager)
helper_name='whole_body_tracking.utils.contact_quality';helper_spec=importlib.util.spec_from_file_location(helper_name,P/'utils/contact_quality.py');helper=importlib.util.module_from_spec(helper_spec);sys.modules[helper_name]=helper;helper_spec.loader.exec_module(helper)
os.environ['OMA_VELOCITY_REWARD_DELTA']=json.dumps(delta)
resolved,audits=m.audit_reward_parameters(base,None);audit=audits['feet_contact_planar_speed']
assert audit['resolved_weight']==-.3 and resolved['feet_contact_planar_speed']==after
assert audit['resolved_entities']['asset_cfg']['body_ids']==[2,1]
assert audit['resolved_entities']['sensor_cfg']['body_ids']==[0,2]
assert audit['native_cost_shape']==[2]and audit['native_cost_finite_nonnegative']
torch.testing.assert_close(live.func(base,**live.params),torch.tensor([.1,.2]))
assert term.params['asset_cfg'].body_ids==slice(None),'Audit must not mutate unresolved declarations'
manager.active_terms=[];reject(lambda:m.audit_reward_parameters(base,None));manager.active_terms=['feet_contact_planar_speed']
wrong=copy.deepcopy(live);wrong.weight=-.4;manager.get_term_cfg=lambda name:wrong;reject(lambda:m.audit_reward_parameters(base,None));manager.get_term_cfg=lambda name:live
for selector,wrong_ids in [('asset_cfg',[1,2]),('sensor_cfg',[2,0]),('asset_cfg',slice(None))]:
 good=live.params[selector].body_ids;live.params[selector].body_ids=wrong_ids
 reject(lambda:m.audit_reward_parameters(base,None));live.params[selector].body_ids=good
live.params['force_threshold']=20.;reject(lambda:m.audit_reward_parameters(base,None));live.params['force_threshold']=10.
live.params['sensor_cfg'].body_names=['left_ankle_roll_link'];reject(lambda:m.audit_reward_parameters(base,None));live.params['sensor_cfg'].body_names=before['body_names']
live.func=lambda:None;reject(lambda:m.audit_reward_parameters(base,None));live.func=term.func
asset.data.body_lin_vel_w[0,2,0]=float('nan');reject(lambda:m.audit_reward_parameters(base,None));asset.data.body_lin_vel_w[0,2,0]=.1
os.environ.pop('OMA_VELOCITY_REWARD_DELTA')
r=subprocess.run([sys.executable,str(R/'scripts/oma_bootstrap.py'),'--velocity_feet_contact_reweight','-.3'],capture_output=True,text=True);assert r.returncode and 'both new and previous' in r.stderr
# Reject weakening before simulator/network startup.
r=subprocess.run([sys.executable,str(R/'scripts/oma_bootstrap.py'),'--proprioceptive_actor','--velocity_reference','--velocity_resume','--proprio_resume','--deployment_target_limits','--gm_resume_url','invalid-not-fetched','--velocity_feet_contact_reweight','-.1','--velocity_feet_contact_previous_weight','-.2'],capture_output=True,text=True);assert r.returncode and 'strictly stronger' in r.stderr
print('PASS actual5997 parent exact scalar strengthening; all unauthorized changes rejected; actual deepcopy/resolved lifecycle with separate asset/sensor IDs; native function execution and all unsafe selectors rejected; legacy nochange; CLI negatives before network/simulator')
