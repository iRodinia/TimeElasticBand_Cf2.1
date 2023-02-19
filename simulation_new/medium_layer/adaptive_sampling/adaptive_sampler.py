import numpy as np
from collections import deque

def saturation_mapping(x: float, p = 1.5, adjust_input = False):
    """using Gaussian mapping
        satisfy following conditions:
        1. x = 0, y = p
        2. x = 1, y = 1
        3. x -> Inf, y -> 1/p

        a*exp(-k*x)+b

        Predict the slowing-down ratio brought by the change of curvature and torsion
    """
    assert p > 1. and x >= 0
    if adjust_input:
        sigma = np.sqrt(1. / (2*np.log(p)))
        k = 2. * sigma
        x = k * (1 - np.exp(-x/k))
    b = 1/p
    a = p - b
    k = np.log(p**2 - 1) - np.log(p - p*b)
    return a*np.exp(-k*x) + b
    


class AdaptiveSampler:
    def __init__(self, params: dict):
        self.max_acc_pred = params['max_acc_pred']
        self.max_acc_curr = params['max_acc_curr']
        self.time_ahead = params['time_ahead']
        self.dt = params['dt']
        self.w_l = params['l_error_weight'] # relative
        self.w_tors = params['torsion_weight'] # relative
        self.k_D = params['k_D']
        self.error_threshold = params['error_threshold']
        self.k_curr = (self.max_acc_curr - 1) / self.error_threshold / self.time_ahead

        self.last_ratio = 1.
        self.buffer_size = 5
        self.slow_mode = False

        self.pred_drate_buffer = deque([], maxlen=self.buffer_size)   # store the cost of future curvature and torsion
        self.curr_drate_buffer = deque([], maxlen=self.buffer_size)   # store the cost of current curvature and torsion
        self.timestamp_buffer = deque([], maxlen=self.buffer_size)
    
    def refresh_error(self, time, frenet_error, curr_ref_curv, curr_ref_tors, pred_ref_curv, pred_ref_tors):
        """compute and store the time_flow_ratio

        Args:
            time (float >= 0): current timestamp
            frenet_error (1*2 array): positional error in Frenet coordinates
                [s_error, l_error],
                s_error is path length error, s_error = s_ref - s_curr
                l_error is the deviation pos from path
            curr_ref_curv (float): The reference trajecotry curvature at (t)
            curr_ref_tors (float): The reference trajecotry torsion at (t)
            pred_ref_curv (float): The reference trajecotry curvature at (t + t_ahead)
            pred_ref_tors (float): The reference trajecotry torsion at (t + t_ahead)
        """
        assert time >= 0

        D_curr = (curr_ref_curv**2 + self.w_tors*curr_ref_tors**2)**0.5
        D_pred = (pred_ref_curv**2 + self.w_tors*pred_ref_tors**2)**0.5
        if D_pred == 0:
            dece_rate = self.max_acc_pred
        elif D_curr == 0:
            dece_rate = 0.
        else:
            _r = D_pred / D_curr
            dece_rate = saturation_mapping(_r, p=self.max_acc_pred)
        
        rate_acc = (dece_rate-1) * (self.last_ratio) / self.time_ahead
        rate_acc = np.exp(-abs(1 - self.last_ratio)) * rate_acc
        self.pred_drate_buffer.append(rate_acc)

        assert len(frenet_error) == 2
        s_error = frenet_error[0]
        l_error = frenet_error[1]

        dist = self.k_curr * (s_error + self.w_l*l_error - self.error_threshold) * (1 + self.k_D*D_curr) # F = k * L
        self.curr_drate_buffer.append(-dist)
        
        self.timestamp_buffer.append(time)
    
    def get_ratio_output(self):
        """compute PID output of the control output
            Attention! The error here is "ref - real", which is contrary to the adjustment
            of Reference signals!

        Args:
        """
        pdrate= self.pred_drate_buffer[-1]
        cdrate = self.curr_drate_buffer[-1]
        # if self.slow_mode:
        #     drate = cdrate
        #     if pdrate > cdrate:
        #         self.slow_mode = False
        # else:
        #     drate = min(pdrate, cdrate)
        #     if pdrate <= 0.01:
        #         self.slow_mode = True
        drate = min(pdrate, cdrate) if pdrate <= -0.05 else cdrate
        ratio = self.last_ratio + drate*self.dt
        ratio = max(0.05, ratio)
        self.last_ratio = ratio
        return ratio

if __name__ == '__main__':
    print(saturation_mapping(0.))
    print(saturation_mapping(1.))
    print(saturation_mapping(33.33))