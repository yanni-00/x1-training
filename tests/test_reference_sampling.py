"""Check command sample shape/velocity semantics without claiming simulator success."""
import ast,json,types
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/walk_stop.py'
tree=ast.parse(p.read_text(encoding='utf-8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WalkStopCommand');scope={'torch':torch,'MotionCommand':object};exec(compile(ast.Module(body=[cls],type_ignores=[]),str(p),'exec'),scope);c=object.__new__(scope['WalkStopCommand'])
c.num_envs=2;c.clock=types.SimpleNamespace(phase=torch.tensor([.5,1.]),rate=torch.tensor([0.,1.]))
q=torch.tensor([[[1.,0,0,0]], [[-1.,0,0,0]], [[1.,0,0,0]]]);pos=torch.tensor([[[0.,0,0]],[[1.,0,0]],[[2.,0,0]]]);j=torch.tensor([[0.,0.],[1.,2.],[2.,4.]])
c.motion=types.SimpleNamespace(time_step_total=3,joint_pos=j,joint_vel=torch.ones(3,2)*2,body_pos_w=pos,body_quat_w=q,body_lin_vel_w=torch.ones(3,1,3),body_ang_vel_w=torch.ones(3,1,3));c._env=types.SimpleNamespace(scene=types.SimpleNamespace(env_origins=torch.tensor([[10.,0,0],[20.,0,0]])))
assert torch.allclose(c.joint_pos,torch.tensor([[.5,1.],[1.,2.]]))
assert torch.equal(c.joint_vel[0],torch.zeros(2)) and torch.equal(c.joint_vel[1],torch.ones(2)*2)
assert torch.equal(c.body_lin_vel_w[0],torch.zeros(1,3)) and torch.equal(c.body_ang_vel_w[0],torch.zeros(1,3))
assert torch.allclose(c.body_quat_w.norm(dim=-1),torch.ones(2,1)), 'Antipodal quaternion interpolation failed'
assert torch.allclose(c.body_pos_w[:,0,0],torch.tensor([10.5,21.]))
assert torch.equal(pos,torch.tensor([[[0.,0,0]],[[1.,0,0]],[[2.,0,0]]])), 'Reference arrays mutated'
update=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_update_command')
for node in ast.walk(update):
 if isinstance(node,ast.Attribute):assert not node.attr.startswith('write_') and node.attr!='_resample_command', 'GO/STOP update may reset/write robot state'
print(json.dumps({'fractional_joint_interpolation':True,'zero_velocity_at_hold':True,'body_velocity_scaling':True,'antipodal_quaternion_handling':True,'environment_origin_offset':True,'source_arrays_immutable':True,'no_robot_write_or_resample_in_command_update':True}))
