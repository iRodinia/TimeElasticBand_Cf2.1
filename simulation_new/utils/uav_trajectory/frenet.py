import os
import numpy as np
from scipy import interpolate
from collections import Iterable
from simulation_new.utils.uav_trajectory.trajectory import Trajectory

def default_divide(a: float, b: float):
    if b == 0:
        if a > 0: return np.Inf
        elif a == 0: return 0
        else: return -np.Inf
    else:
        return a / b

def box_analysis(x, y, threshold=3.):
    assert len(x) == len(y)

    up_percent = np.percentile(y, 75)
    low_percent = np.percentile(y, 25)
    iqr = up_percent - low_percent
    x_new = []
    y_new = []
    for k in range(len(y)):
        if y[k] >= low_percent - threshold*iqr and y[k] <= up_percent + threshold*iqr:
            x_new.append(x[k])
            y_new.append(y[k])
        elif k == len(y)-1:
            x_new.append(x[k])
            y_new.append(y_new[-1])
        elif k == 0:
            x_new.append(x[k])
            y_new.append(0.)
    return x_new, y_new

class FrenetTrajectory(Trajectory):
    
    def __init__(self, coef_mat=None, order=10, path=None):
        super().__init__(coef_mat=coef_mat, order=order)
        if path:
            self.loadcsv(path)
        self.discrete_accuracy = 0.05
        self.derivative1 = [poly.derivative() for poly in self.polynomials]
        self.derivative2 = [poly.derivative() for poly in self.derivative1]
        self.derivative3 = [poly.derivative() for poly in self.derivative2]
        self._frenet_formula()
        self._length_formula()
    
    def _eval_ex(self, polynomials_4d, t):
        assert t >= 0
        assert t <= self.duration
        current_t = 0.0
        for p in polynomials_4d:
            if t <= current_t + p.duration:
                return np.array([p.px.eval(t - current_t), p.py.eval(t - current_t), p.pz.eval(t - current_t), p.pyaw.eval(t - current_t)])
            current_t = current_t + p.duration

    def _frenet_formula(self):
        time_accuracy = self.discrete_accuracy
        curvature = []
        torsion = []
        timestamps = np.linspace(0, self.duration, num=int(self.duration/time_accuracy+1), endpoint=True)
        for t in timestamps:
            d1 = self._eval_ex(self.derivative1, t)[0:3]
            d2 = self._eval_ex(self.derivative2, t)[0:3]
            d3 = self._eval_ex(self.derivative3, t)[0:3]
            mid_vec = np.cross(d1, d2)
            curvature.append(default_divide(np.linalg.norm(mid_vec), np.linalg.norm(d1)**3))
            torsion.append(default_divide(np.dot(mid_vec, d3), np.linalg.norm(mid_vec)**2))

        # self.curvature_fnc = interpolate.interp1d(timestamps, curvature, kind='linear')
        # self.torsion_fnc = interpolate.interp1d(timestamps, torsion, kind='linear')
        timestamps1, curvature = box_analysis(timestamps, curvature)
        timestamps2, torsion = box_analysis(timestamps, torsion)
        self.curvature_fnc = interpolate.interp1d(timestamps1, curvature, kind='quadratic')
        self.torsion_fnc = interpolate.interp1d(timestamps2, torsion, kind='quadratic')
    
    def _length_formula(self):
        timestamps = np.linspace(0, self.duration, num=int(self.duration/self.discrete_accuracy+1), endpoint=True)
        positions = np.array([self.eval(t).pos for t in timestamps])
        pos_err = np.diff(positions, axis=0)
        dist_err = np.concatenate(([0.], [np.linalg.norm(d) for d in pos_err]))
        dist = np.cumsum(dist_err)
        self.length_fnc = interpolate.interp1d(timestamps, dist, kind='linear')

    def eval_curvature(self, timestamps):
        if not isinstance(timestamps, Iterable):
            if timestamps > self.duration:
                timestamps = self.duration
            return self.curvature_fnc(timestamps)

        result = []
        for t in timestamps:
            if t > self.duration:
                print("Warning: evaluate point {0:.3f} exceeds maximum duration {1:.3f}".format(t, self.duration))
                result.append(0.)
            else:
                result.append(self.curvature_fnc(t))
        return result
    
    def eval_torsion(self, timestamps):
        if not isinstance(timestamps, Iterable):
            if timestamps > self.duration:
                timestamps = self.duration
            return self.torsion_fnc(timestamps)

        result = []
        for t in timestamps:
            if t > self.duration:
                print("Warning: evaluate point {0:.3f} exceeds maximum duration {1:.3f}".format(t, self.duration))
                result.append(0.)
            else:
                result.append(self.torsion_fnc(t))
        return result
    
    def eval_length(self, timestamps):
        if not isinstance(timestamps, Iterable):
            if timestamps > self.duration:
                timestamps = self.duration
            return self.length_fnc(timestamps)

        result = []
        for t in timestamps:
            if t > self.duration:
                t = self.duration
            result.append(self.length_fnc(t))
        return result
    
    def closest_point_from_cartesian(self, crd, ref_time=None, accuracy: float=None):
        """compute the closest position frenet coordinate s on the curve from a cartesian coordinate [x,y,z]
        Args:
            crd (1*3 array): The cartesian coordinate
            ref_time (float > 0): The current time for more accurate evaluation
        Return:
            t_ref, s, position
        """
        crd = np.array(crd)
        if accuracy != None and accuracy > 0:
            dt = accuracy
        else:
            dt = self.duration / 500

        if ref_time == None:
            timestamps = np.linspace(0, self.duration, num=int(self.duration/dt+1), endpoint=True)
            positions = np.array([self.eval(t).pos for t in timestamps])
            dists = np.array([np.linalg.norm(p-crd) for p in positions])
            index = np.unravel_index(dists.argmin(), dists.shape)
            t_ref = timestamps[index]
        else:
            assert isinstance(ref_time, float) and ref_time >= 0
            ref_idx = self.time_at_which_piece(ref_time)
            nearby_range = 2 # define which polynomial pieces to search around ref_idx
            idxes = range(max(0,ref_idx-nearby_range), min(ref_idx+nearby_range+1,self.n_pieces()))
            time_before = self.total_time_before_n_piece(idxes[0])
            seg_time = 0.
            for i in idxes:
                seg_time += self.polynomials[i].duration
            timestamps = np.linspace(0, seg_time, num=int(seg_time/dt+1), endpoint=True) + time_before
            positions = np.array([self.eval(t).pos for t in timestamps])
            dists = np.array([np.linalg.norm(p-crd) for p in positions])
            index = np.unravel_index(dists.argmin(), dists.shape)
            if dists[index] > 1.:
                return self.closest_point_from_cartesian(crd, ref_time=None)
            t_ref = timestamps[index]

        return t_ref, self.length_fnc(t_ref), positions[index]

    def print_traj_status(self):
        import matplotlib.pyplot as plt

        timestamps = np.linspace(0, self.duration, num=int(self.duration/self.discrete_accuracy+1), endpoint=True)

        fig1 = plt.figure("shape", figsize=(5,3.2))
        ax1 = plt.axes(projection='3d')
        start = self._eval_ex(self.polynomials, 0)[0:3]
        ax1.scatter(start[0], start[1], start[2], label="start position")
        pos = np.array([self._eval_ex(self.polynomials, t)[0:3] for t in timestamps])
        ax1.plot(pos[:,0], pos[:,1], pos[:,2])
        ax1.legend()

        fig2 = plt.figure("frenet", figsize=(5,3.2))
        ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])
        curv = [self.curvature_fnc(t) for t in timestamps]
        l1 = ax2.plot(timestamps, curv, 'g-')
        tors = [self.torsion_fnc(t) for t in timestamps]
        l2 = ax2.plot(timestamps, tors, 'm-')
        ax2.legend(labels = ('curvature', 'torsion'), loc = 'lower right')

        fig3 = plt.figure("reference")
        yaw = [self._eval_ex(self.polynomials, t)[3] for t in timestamps]
        ax3_1 = fig3.add_subplot(4,1,1)
        ax3_1.plot(timestamps, pos[:,0])
        ax3_1.grid()
        ax3_2 = fig3.add_subplot(4,1,2)
        ax3_2.plot(timestamps, pos[:,1])
        ax3_2.grid()
        ax3_3 = fig3.add_subplot(4,1,3)
        ax3_3.plot(timestamps, pos[:,2])
        ax3_3.grid()
        ax3_4 = fig3.add_subplot(4,1,4)
        ax3_4.plot(timestamps, yaw)
        ax3_4.grid()

        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = os.path.abspath(os.path.dirname(__file__)) + '/generated_trajectory/test.csv'
    tr = FrenetTrajectory(path=path)
    _,s,p = tr.closest_point_from_cartesian([1,2,0.5])
    tr.print_traj_status()
