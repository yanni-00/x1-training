"""Source-driven offline F1 actuation primitives; no simulator or training entry.

Every hardware parameter is supplied explicitly. The variable-inertia model is
the relative rotor-spin contribution only, not a calibrated full actuator model.
No automatic integration into Isaac Gym or inference of firmware gain units.
"""
from itertools import product
import math

import numpy as np


def named_values(source_names, values, target_names):
    """Reorder by names, accepting extra upper-body channels, rejecting gaps."""
    if len(source_names) != len(set(source_names)) or len(target_names) != len(set(target_names)):
        raise ValueError('Duplicate joint name')
    if np.asarray(values).shape[-1] != len(source_names):
        raise ValueError('Joint value/name size mismatch')
    index = {name:i for i,name in enumerate(source_names)}
    if not set(target_names) <= index.keys():
        raise ValueError('Missing required joint name')
    return np.asarray(values)[..., [index[n] for n in target_names]]


class MitCodec:
    """Float32 clamp/truncate and bit layout from EncosMotor::SetMitCmd.

    decode() is a software inverse of the transmitter mapping, not a claim about
    measured firmware quantization, special-zero conventions or physical torque.
    """
    fields = ('kp','kd','pos','vel','toq')

    def __init__(self, parameters, bits):
        self.parameters, self.bits = parameters, bits
        if tuple(bits[f] for f in self.fields) != (12,9,16,12,12):
            raise ValueError('Unsupported ENC0S packet source layout')

    def integers(self, command):
        result = {}
        for field in self.fields:
            value = np.float32(command[field])
            if not np.isfinite(value):
                raise ValueError('Non-finite motor command')
            lo,hi = [np.float32(self.parameters[field+'_'+k]) for k in ('min','max')]
            value = np.clip(value,lo,hi)
            span = np.float32(hi-lo)
            scaled = np.float32(np.float32(value-lo)*np.float32((1<<self.bits[field])-1))
            result[field] = int(np.float32(scaled/span))
        return result

    def encode(self, command):
        x = self.integers(command)
        p,d,q,v,t = [x[f] for f in self.fields]
        return bytes((p>>7, ((p&0x7f)<<1)|((d&0x100)>>8), d&0xff,
                      q>>8, q&0xff, v>>4, ((v&0xf)<<4)|(t>>8), t&0xff))

    def decode(self, packet):
        if len(packet)!=8 or packet[0]&0xe0:
            raise ValueError('Invalid MIT packet/mode')
        a = packet
        x = {'kp':(a[0]<<7)|(a[1]>>1), 'kd':((a[1]&1)<<8)|a[2],
             'pos':(a[3]<<8)|a[4], 'vel':(a[5]<<4)|(a[6]>>4),
             'toq':((a[6]&0xf)<<8)|a[7]}
        result = {}
        for field in self.fields:
            lo,hi = [np.float32(self.parameters[field+'_'+k]) for k in ('min','max')]
            span = np.float32(hi-lo)
            decoded = np.float32(np.float32(np.float32(x[field])*span)/np.float32((1<<self.bits[field])-1))
            result[field] = float(np.float32(decoded+lo))
        return result


class ParallelAnkle:
    """Deployed branch geometry, exact implicit first/second derivatives."""
    def __init__(self, geometry, reflected_rotor_inertias, torque_bounds, numerical):
        self.geometry = geometry
        self.inertias = np.asarray(reflected_rotor_inertias,dtype=float)
        self.bounds = np.asarray(torque_bounds,dtype=float)
        self.numerical = numerical
        if len(geometry)!=2 or self.inertias.shape!=(2,) or not np.isfinite(self.inertias).all() or np.any(self.inertias<=0) or self.bounds.shape!=(2,) or not np.isfinite(self.bounds).all() or np.any(self.bounds<=0):
            raise ValueError('Incomplete ankle data')

    def kinematics(self, q):
        pitch,roll = np.asarray(q,dtype=float)
        if not np.isfinite([pitch,roll]).all():
            raise ValueError('Non-finite ankle position')
        sp,cp,sr,cr = math.sin(pitch),math.cos(pitch),math.sin(roll),math.cos(roll)
        motors,jacobians,hessians = [],[],[]
        for branch in self.geometry:
            x,y,z = branch['foot_attachment']
            center = np.asarray(branch['motor_center'])
            radius,length = branch['radius'],branch['rod_length']
            yr,zr = y*cr-z*sr,y*sr+z*cr
            p = np.array([x*cp+zr*sp,yr,-x*sp+zr*cp])
            dx,dy,dz = p[0]-center[0],p[1]-center[1],center[2]-p[2]
            a,c = 2*radius*dy,2*radius*dz
            rho = math.hypot(a,c)
            if rho<=self.numerical['hypot_min']:
                raise ValueError('Degenerate crank constraint')
            argument = (length**2-dx**2-dy**2-dz**2-radius**2)/rho
            tolerance = self.numerical['inverse_argument_tolerance']
            if not -1-tolerance<=argument<=1+tolerance:
                raise ValueError('Unreachable ankle pose; no silent clamping')
            theta = math.atan2(a,c)+branch['assembly_sign']*math.acos(np.clip(argument,-1.,1.))
            direction = branch['motor_direction']
            motor = math.remainder((theta-branch['zero_angle'])/direction,2*math.pi)
            theta = branch['zero_angle']+direction*motor
            crank = center+np.array([0.,-radius*math.sin(theta),radius*math.cos(theta)])
            d = p-crank
            cm = np.array([0.,-direction*radius*math.cos(theta),-direction*radius*math.sin(theta)])
            cmm = direction**2*np.array([0.,radius*math.sin(theta),-radius*math.cos(theta)])
            pq = np.array([[p[2],0.,-p[0]],[sp*yr,-zr,cp*yr]])
            pqq = np.empty((2,2,3))
            pqq[0,0] = [-p[0],0.,-p[2]]
            pqq[0,1] = pqq[1,0] = [cp*yr,0.,-sp*yr]
            pqq[1,1] = [-sp*zr,-yr,-cp*zr]
            denominator = float(d@cm)
            if abs(denominator)<=self.numerical['crank_relative_tolerance']*length*radius:
                raise ValueError('Singular ankle transmission')
            j = pq@d/denominator
            h = np.empty((2,2))
            for i,k in product(range(2),repeat=2):
                h[i,k] = (pq[i]@pq[k]+d@pqq[i,k]-(pq[i]@cm)*j[k]
                          -(pq[k]@cm)*j[i]+(cm@cm-d@cmm)*j[i]*j[k])/denominator
            motors.append(motor);jacobians.append(j);hessians.append(h)
        return np.asarray(motors),np.asarray(jacobians),np.asarray(hessians)

    def inertia_and_bias(self,q,velocity):
        _,j,h = self.kinematics(q)
        v = np.asarray(velocity)
        matrix = j.T@(self.inertias[:,None]*j)
        convective_motor_acceleration = np.einsum('kij,i,j->k',h,v,v)
        bias = j.T@(self.inertias*convective_motor_acceleration)
        # dM/dq[a] for independent kinetic-energy/Christoffel verification.
        derivative = np.array([h[:,:,a].T@(self.inertias[:,None]*j)
                               +j.T@(self.inertias[:,None]*h[:,:,a]) for a in range(2)])
        return matrix,bias,derivative

    def measured_jacobian(self,q,motors):
        """C++ feedback/command Jacobian uses FK q and the measured m snapshot.

        Do not replace measured m with inverse_position(FK(m)); table
        interpolation means they are not numerically identical.
        """
        pitch,roll = q
        sp,cp,sr,cr = math.sin(pitch),math.cos(pitch),math.sin(roll),math.cos(roll)
        rows = []
        for b,m in zip(self.geometry,motors):
            x,y,z = b['foot_attachment']
            yr,zr = y*cr-z*sr,y*sr+z*cr
            p = np.array([cp*x+sp*zr,yr,-sp*x+cp*zr])
            theta = b['zero_angle']+b['motor_direction']*m
            crank = np.asarray(b['motor_center'])+np.array([0.,-b['radius']*math.sin(theta),b['radius']*math.cos(theta)])
            cm = np.array([0.,-b['motor_direction']*b['radius']*math.cos(theta),-b['motor_direction']*b['radius']*math.sin(theta)])
            d = p-crank
            denominator = d@cm
            if abs(denominator)<=self.numerical['crank_relative_tolerance']*b['rod_length']*b['radius']:
                raise ValueError('Singular measured crank constraint')
            rows.append(np.array([[p[2],0.,-p[0]],[sp*yr,-zr,cp*yr]])@d/denominator)
        j = np.asarray(rows)
        if abs(np.linalg.det(j))<=self.numerical['inverse_relative_tolerance']*np.max(abs(j))**2:
            raise ValueError('Singular measured motor/joint Jacobian')
        return j

    def limit_torque(self,q,joint_torque):
        _,j,_ = self.kinematics(q)
        if abs(np.linalg.det(j))<=self.numerical['inverse_relative_tolerance']*np.max(abs(j))**2:
            raise ValueError('Singular motor/joint Jacobian')
        requested = np.linalg.solve(j.T,np.asarray(joint_torque))
        actual = np.clip(requested,-self.bounds,self.bounds)
        return j.T@actual,requested,actual

    def constraint_residual(self,q,motors):
        pitch,roll = q
        sp,cp,sr,cr = math.sin(pitch),math.cos(pitch),math.sin(roll),math.cos(roll)
        residual = []
        for b,m in zip(self.geometry,motors):
            x,y,z = b['foot_attachment']
            yr,zr = y*cr-z*sr,y*sr+z*cr
            p = np.array([cp*x+sp*zr,yr,-sp*x+cp*zr])
            theta = b['zero_angle']+b['motor_direction']*m
            crank = np.asarray(b['motor_center'])+np.array([0.,-b['radius']*math.sin(theta),b['radius']*math.cos(theta)])
            residual.append(np.linalg.norm(p-crank)-b['rod_length'])
        return np.asarray(residual)


class AnkleLookup:
    """Current C++ bilinear FK, including its left/right mirror convention."""
    def __init__(self,table,settings):
        self.table,self.settings = np.asarray(table),settings
        assert self.table.shape==(settings['rows'],settings['columns'],2)

    def lookup(self,motors,side,ankle):
        if side not in ('left','right'):
            raise ValueError('Explicit ankle side required')
        m = np.asarray(motors) * (-1 if side=='left' else 1)
        u = (m-self.settings['motor_origin_rad'])/self.settings['motor_step_rad']
        last = np.array(self.table.shape[:2])-1
        epsilon = self.settings['coordinate_epsilon']
        if np.any(u < -epsilon) or np.any(u > last+epsilon):
            raise ValueError('Ankle motor outside FK lookup domain')
        u = np.clip(u,0,last)
        base = np.minimum(np.floor(u).astype(int),last-1)
        a,b = u-base
        i,j = base
        q = (1-a)*(1-b)*self.table[i,j]+a*(1-b)*self.table[i+1,j]+(1-a)*b*self.table[i,j+1]+a*b*self.table[i+1,j+1]
        if side=='left':
            q = q*np.array([1,-1])
        if np.max(abs(ankle.constraint_residual(q,motors))) > self.settings['interpolated_length_tolerance_m']:
            raise ValueError('Ankle interpolation violates geometry tolerance')
        return q


class SecondOrderFilter:
    def __init__(self,input_coefficients,feedback_coefficients,initial_value):
        self.b = np.asarray(input_coefficients,dtype=float)
        self.c = np.asarray(feedback_coefficients,dtype=float)
        self.previous_input = np.tile(np.asarray(initial_value,dtype=float),(2,1))
        self.previous_output = self.previous_input.copy()

    def update(self,value):
        value = np.asarray(value,dtype=float)
        out = self.b[0]*value+self.b[1]*self.previous_input[0]+self.b[2]*self.previous_input[1]+self.c[0]*self.previous_output[0]+self.c[1]*self.previous_output[1]
        self.previous_input[1],self.previous_input[0] = self.previous_input[0].copy(),value.copy()
        self.previous_output[1],self.previous_output[0] = self.previous_output[0].copy(),out.copy()
        return out


class ObservationHistory:
    def __init__(self,length,frame_size,initialization):
        if initialization not in ('zero_prefix','repeat_first'):
            raise ValueError('Explicit history initialization required')
        self.buffer = np.zeros((length,frame_size))
        self.initialization = initialization
        self.first = True

    def append(self,frame):
        frame = np.asarray(frame)
        if frame.shape!=self.buffer.shape[1:]:
            raise ValueError('Observation frame size mismatch')
        if self.first and self.initialization=='repeat_first':
            self.buffer[:] = frame
        self.buffer[:-1] = self.buffer[1:].copy()
        self.buffer[-1] = frame
        self.first = False
        return self.buffer.reshape(-1).copy()


def integer_period_steps(host_dt,frequency_hz):
    steps = 1/(host_dt*frequency_hz)
    if steps<=0 or not math.isclose(steps,round(steps),rel_tol=0,abs_tol=np.finfo(float).eps*max(1,abs(steps))*8):
        raise ValueError('Noninteger host/policy period')
    return round(steps)


def implicit_friction_step(matrix,velocity,torque_minus_bias,damping,friction,dt):
    """SPD velocity step with exact two-DOF Coulomb active-set solve.

    Numerical integrator: (M/dt+B)v_next + f = M v/dt + tau - c;
    f in friction*Sign(v_next). No smoothing-velocity parameter is invented.
    Joint limits/contact are deliberately outside this fixed-base solver.
    """
    matrix = np.asarray(matrix)
    if matrix.shape!=(2,2) or not np.isfinite(matrix).all() or np.min(np.linalg.eigvalsh(matrix))<=0:
        raise ValueError('Expected positive definite two-axis inertia')
    friction = np.asarray(friction)
    if dt<=0 or not np.isfinite(dt) or not np.isfinite(damping).all() or not np.isfinite(friction).all() or np.any(friction<0) or np.any(np.asarray(damping)<0):
        raise ValueError('Invalid integration or passive dynamics input')
    a = matrix/dt+np.diag(damping)
    rhs = matrix@velocity/dt+torque_minus_bias
    tolerance = np.finfo(float).eps*max(1.,float(np.max(abs(rhs))),float(np.max(abs(a))))*64
    for status in product((0,-1,1),repeat=2):
        status = np.asarray(status)
        active = np.flatnonzero(status)
        inactive = np.flatnonzero(status==0)
        result = np.zeros(2)
        if len(active):
            result[active] = np.linalg.solve(a[np.ix_(active,active)],rhs[active]-friction[active]*status[active])
        force = rhs-a@result
        if np.all(result[active]*status[active]>=-tolerance) and np.all(abs(force[inactive])<=friction[inactive]+tolerance):
            return result,force
    raise ValueError('Friction active-set solve failed')


def require_backend_support(backend,rotor_mode,identified_armature_active):
    if rotor_mode not in ('constant_nominal_diagonal','posture_full'):
        raise ValueError('Unknown rotor mode')
    if rotor_mode=='posture_full' and identified_armature_active:
        raise ValueError('Reject double counting: identified armature plus relative rotor model')
    if rotor_mode=='posture_full' and backend!='reduced_full_matrix':
        raise ValueError('Full state-dependent inertia and bias require a full-matrix solver; scalar DOF armature cannot represent them')
