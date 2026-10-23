"""Exact one-joint collision-vertex height envelope at a fixed pad midpoint.

Uses only robot USD geometry and a proposed grasp pose, never object geometry.
All collision hull vertices count as over the table: this is conservative if a
vertex sweeps beyond the table boundary. Existing runtime checks remain exact
to their existing sampled model and table bounds.
"""
import numpy as np

from .gripper_geometry import FIXED_REFERENCE_M, MOVING_REFERENCE_M


class FixedMidpointClosureEnvelope:
    def __init__(self, geometry, parts):
        self.labels=[]
        samples=[]
        for angle in (0.,np.pi/2,np.pi):
            midpoint=(FIXED_REFERENCE_M+geometry.moving_to_fixed(MOVING_REFERENCE_M,angle))/2
            vectors=[]
            for body,name,vertices,_ in parts:
                transformed=geometry.moving_to_fixed(vertices,angle) if body=='moving' else vertices
                vectors.append(transformed-midpoint)
                if angle==0.:
                    self.labels.extend((str(body),str(name)) for _ in vertices)
            if not vectors:
                raise ValueError('Closure envelope requires robot collision vertices')
            samples.append(np.concatenate(vectors))
        self.constant=(samples[0]+samples[2])/2
        self.cosine=(samples[0]-samples[2])/2
        self.sine=samples[1]-self.constant

    def minimum(self, centre_world_m, rotation_world, jaw_start_rad, jaw_end_rad):
        centre=np.asarray(centre_world_m,dtype=float)
        rotation=np.asarray(rotation_world,dtype=float)
        lower,upper=sorted((float(jaw_start_rad),float(jaw_end_rad)))
        if (centre.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(centre).all() or
                not np.isfinite(rotation).all() or not np.isfinite([lower,upper]).all() or
                upper-lower>2*np.pi or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-4) or
                not np.isclose(np.linalg.det(rotation),1.,atol=1e-4)):
            raise ValueError('Invalid world pose or jaw interval for closure envelope')
        # Every vertex height is c+a*cos(theta)+b*sin(theta). Its minimum on
        # the closed interval occurs at an endpoint or an enclosed stationary
        # minimum. No sampling gap or per-angle IK is required.
        c=centre[2]+self.constant@rotation[2]
        a=self.cosine@rotation[2]
        b=self.sine@rotation[2]
        lo=c+a*np.cos(lower)+b*np.sin(lower)
        hi=c+a*np.cos(upper)+b*np.sin(upper)
        heights=np.minimum(lo,hi)
        angles=np.where(lo<=hi,lower,upper)
        stationary=np.arctan2(b,a)+np.pi
        for turn in (-2*np.pi,0.,2*np.pi):
            theta=stationary+turn
            value=c-np.hypot(a,b)
            use=(theta>=lower)&(theta<=upper)&(value<heights)
            heights=np.where(use,value,heights)
            angles=np.where(use,theta,angles)
        index=int(np.argmin(heights));theta=float(angles[index])
        vector=self.constant[index]+self.cosine[index]*np.cos(theta)+self.sine[index]*np.sin(theta)
        vertex=centre+rotation@vector
        body,name=self.labels[index]
        return {'minimum_mesh_z_m':float(heights[index]),'minimum_vertex_world_m':vertex.tolist(),
            'critical_jaw_joint_rad':theta,'body':body,'mesh':name,
            'jaw_interval_rad':[lower,upper], 'vertex_count':len(self.labels),
            'method':'analytic constant-pad-midpoint revolute-joint collision-hull vertex envelope',
            'table_xy_assumption':'all vertices conservatively treated over table; runtime bounds unchanged',
            'reference_semantics':'robot pad-reference midpoint at candidate point; no object center or GT'}
