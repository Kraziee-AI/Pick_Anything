"""Read-only, opt-in Newton/MJWarp evidence; never used by control or scoring.

MJWarp Contact.pos is a world-frame midpoint, frame[0] points geom0->geom1.
ImplicitActuator computed/applied effort is approximate; solver qfrc_actuator
is a separate solver quantity. Snapshots cover the last physics substep only.
"""
import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = 'small_grasp_physics_diagnostic_v1'
PHASES = {'close_centered', 'hold', 'lift', 'lift_hold'}


def array(value):
    """Copy a NumPy, Warp or Isaac ProxyArray without writing backend buffers."""
    value = getattr(value, 'torch', value)
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value.numpy() if hasattr(value, 'numpy') else value).copy()


def contacts_snapshot(data, geom_map, shape_body, shape_labels, body_labels, body_q,
                      robot_root, target_root, maximum=256):
    count = int(array(data.nacon).reshape(-1)[0]); capacity = int(data.naconmax)
    if not 0 <= count <= capacity:
        raise ValueError(f'Invalid solver detected contact count {count}/{capacity}')
    robot_bodies = [i for i,label in enumerate(body_labels) if str(label).startswith(robot_root+'/') and
                    str(label).rsplit('/',1)[-1] in ('gripper','moving_jaw_so101_v1')]
    target_shapes = [i for i,label in enumerate(shape_labels) if str(label).startswith(target_root+'/')]
    if len(robot_bodies)!=2 or not target_shapes:
        raise ValueError('Expected two robot jaw bodies and target shapes not resolved')
    contact = data.contact
    fields = {key: array(getattr(contact, key))[:count] for key in
              ('geom', 'worldid', 'pos', 'frame', 'dist', 'dim', 'efc_address')}
    records = []; included = 0; selected = 0
    for i in range(count):
        active = bool((np.asarray(fields['efc_address'][i]) >= 0).any())
        included += active
        world = int(fields['worldid'][i]); geoms = np.asarray(fields['geom'][i], dtype=int)
        if world < 0 or world >= len(geom_map) or (geoms < 0).any(): continue
        if (geoms >= geom_map.shape[1]).any(): raise ValueError('Invalid solver geom id')
        shapes = np.asarray(geom_map[world, geoms], dtype=int)
        if (shapes < 0).any(): continue
        bodies = np.asarray(shape_body[shapes], dtype=int)
        labels = [str(shape_labels[s]) for s in shapes]
        robots = [j for j, b in enumerate(bodies) if b >= 0 and
                  str(body_labels[b]).startswith(robot_root + '/') and
                  str(body_labels[b]).rsplit('/', 1)[-1] in ('gripper', 'moving_jaw_so101_v1')]
        if len(robots) != 1: continue
        side = robots[0]; other = 1-side
        if not labels[other].startswith(target_root + '/'): continue
        selected += 1
        if len(records) >= maximum: continue
        point = np.asarray(fields['pos'][i], dtype=float)
        normal = np.asarray(fields['frame'][i], dtype=float)[0]
        pose = np.asarray(body_q[bodies[side]], dtype=float)
        if not all(np.isfinite(x).all() for x in (point, normal, pose)):
            raise ValueError('Nonfinite contact geometry')
        if not np.isclose(np.linalg.norm(normal), 1., atol=1e-4):
            raise ValueError('Solver contact normal is not unit length')
        rotation = Rotation.from_quat(pose[3:]).as_matrix()
        records.append({'solver_contact_index':i, 'world_id':world,
            'geom_ids':geoms.tolist(), 'newton_shape_ids':shapes.tolist(),
            'shape_labels':labels, 'newton_body_ids':bodies.tolist(),
            'body_labels':[str(body_labels[b]) if b>=0 else '<world>' for b in bodies],
            'robot_side':side, 'robot_body':str(body_labels[bodies[side]]),
            'point_world_m':point.tolist(),
            'point_robot_body_at_sample_m':((point-pose[:3]) @ rotation).tolist(),
            'normal_world_geom0_to_geom1':normal.tolist(),
            'normal_world_robot_to_target':(normal if side==0 else -normal).tolist(),
            'signed_distance_m':float(fields['dist'][i]), 'dimension':int(fields['dim'][i]),
            'constraint_active':active, 'efc_address':np.asarray(fields['efc_address'][i]).tolist()})
    return {'status':'available', 'source':'NewtonManager._solver.mjw_data.contact',
        'point_semantics':'solver contact midpoint between geoms; world metres',
        'normal_semantics':'world unit normal geom0 to geom1; not a force or pressure',
        'local_frame_semantics':'world midpoint reexpressed in sampled post-step robot body frame; collision generation pose may be earlier',
        'detected_count':count, 'constraint_included_count':included, 'solver_capacity':capacity,
        'expected_robot_body_ids':robot_bodies,'target_shape_ids':target_shapes,
        'selected_pair_count':selected, 'record_limit':maximum,
        'truncated':selected>len(records), 'records':records}


def solver_joint_effort(solver, model, joint_names, robot_root):
    """Map solver generalized forces to these scalar robot DOFs explicitly."""
    starts = array(model.joint_qd_start); mapping = array(solver.mjc_dof_to_newton_dof)
    forces = array(solver.mjw_data.qfrc_actuator)
    constraints = array(solver.mjw_data.qfrc_constraint)
    rows = []
    for name in joint_names:
        matches = [i for i, label in enumerate(model.joint_label)
                   if str(label).startswith(robot_root+'/') and str(label).rsplit('/',1)[-1]==name]
        if len(matches)!=1: raise ValueError(f'Ambiguous/unavailable robot joint mapping: {name}')
        joint = matches[0]; dof = int(starts[joint])
        if int(np.sum(array(model.joint_dof_dim)[joint])) != 1:
            raise ValueError('Diagnostic mapping requires scalar robot joints')
        mapped = np.argwhere(mapping==dof)
        if len(mapped)!=1: raise ValueError(f'Ambiguous/unavailable solver DOF: {name}')
        world, mj_dof = map(int, mapped[0])
        a, c = float(forces[world,mj_dof]), float(constraints[world,mj_dof])
        if not np.isfinite([a,c]).all(): raise ValueError('Nonfinite solver joint effort')
        rows.append({'joint':name, 'newton_joint_id':joint, 'newton_dof_id':dof,
            'world_id':world, 'mujoco_dof_id':mj_dof,
            'actuator_effort_nm':a, 'all_constraint_effort_nm':c})
    return {'status':'available','source':'MJWarp qfrc_actuator/qfrc_constraint mapped by mjc_dof_to_newton_dof',
        'semantics':'last-substep solver generalized actuator force; constraints include all contacts, friction and limits, not target-specific',
        'rows':rows}


def unavailable(error):
    return {'status':'unavailable','reason':f'{type(error).__name__}: {error}'}


class PhysicsDiagnostics:
    def __init__(self, target_root, robot_root='/World/envs/env_0/Robot'):
        self.target_root, self.robot_root = target_root, robot_root

    def sample(self, env):
        row = {'step':int(env.episode_length_buf[0]), 'schema':SCHEMA,
            'sample_time':'after env.step; last solver physics substep snapshot, not substep history',
            'control_dt_s':float(env.step_dt)}
        robot = env.scene['robot']; joint = {'joint_names':list(robot.joint_names),
            'implicit_effort_semantics':'computed/applied torque are approximate actuator telemetry, not solver-measured effort',
            'implicit_effort_sample_time':'last actuator computation; may precede last solver snapshot'}
        for field in ('joint_pos','joint_vel','joint_pos_target','joint_effort_limits','joint_vel_limits',
                      'computed_torque','applied_torque'):
            try:
                value = array(getattr(robot.data,field))[0]
                if not np.isfinite(value).all(): raise ValueError('nonfinite values')
                units = ('rad/s' if field in ('joint_vel','joint_vel_limits') else
                         'rad' if field in ('joint_pos','joint_pos_target') else 'Nm')
                joint[field] = {'status':'available','source':f'robot.data.{field}',
                                'units':units,'values':value.tolist()}
            except (AttributeError,ValueError,RuntimeError,IndexError) as error:
                joint[field] = unavailable(error)
        row['joint_telemetry'] = joint
        try:
            from isaaclab_newton.physics import NewtonManager
            model = NewtonManager.get_model(); solver = NewtonManager._solver
            row['solver_dt_s'] = float(NewtonManager.get_solver_dt())
            row['contacts'] = contacts_snapshot(solver.mjw_data,
                array(solver.mjc_geom_to_newton_shape),array(model.shape_body),model.shape_label,
                model.body_label,array(NewtonManager.get_state_0().body_q),self.robot_root,self.target_root)
        except (AttributeError,ValueError,RuntimeError,IndexError,ImportError) as error:
            row['contacts'] = unavailable(error)
        try:
            row['solver_joint_effort'] = solver_joint_effort(solver,model,robot.joint_names,self.robot_root)
        except (AttributeError,ValueError,RuntimeError,IndexError,UnboundLocalError) as error:
            row['solver_joint_effort'] = unavailable(error)
        return row


def phase_rows(snapshots, trace):
    """Attach the actual issued phase AFTER execution; never feed it to control."""
    phases = {s['step']:s['phase'] for s in trace if 'joint_target' in s}
    return [{**s,'phase':phases[s['step']]} for s in snapshots if phases.get(s['step']) in PHASES]
