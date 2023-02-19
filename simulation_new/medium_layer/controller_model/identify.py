import os
import numpy as np
from collections import deque
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial

class ControllerIndetifier:

    def __init__(self, model_degree=2):
        self.poly_deg = model_degree
        self.buffer_size = 1000
        self.data_lag = 3
        self.predict_step = 20
        self.x_data = deque([], maxlen=self.buffer_size)
        self.y_data = deque([], maxlen=self.buffer_size)

        self.x_dim = 18
        # [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r, x_ref, y_ref, z_ref, x_d_ref, y_d_ref, z_d_ref]
        self.y_dim = 4
        # [f1, f2, f3, f4]

        xlags = [list(range(1, self.data_lag+1)) for _ in range(self.x_dim)]
        self.models = [FROLS(
            order_selection=True,
            ylag=self.data_lag, xlag=xlags,
            # n_terms=nterms,
            info_criteria='aic',
            estimator='least_squares',
            basis_function=Polynomial(degree=model_degree)
        ) for _ in range(self.y_dim)]
    
    def train_model(self, x_train, y_train):
        assert len(x_train) == len(y_train)
        
        self.x_data.extend(x_train)
        self.y_data.extend(y_train)
        xdata = np.array(self.x_data)
        ydata = np.array(self.y_data)
        for i in range(self.y_dim):
            self.models[i].fit(X=xdata, y=ydata[:,i].reshape(-1, 1))
    
    def predict(self, x_data):
        assert len(self.y_data) >= self.data_lag
        xdata = np.array(self.x_data)
        ydata = np.array(self.y_data)
        xdata_extend = np.concatenate((x_data,xdata[-self.data_lag:]), axis=0)
        result = []
        for i in range(self.y_dim):
            p = self.models[i].predict(X=xdata_extend, y=ydata[-self.data_lag:])
            result.append(p.flatten(order='C'))
        pred = np.array(result).T
        return pred[self.data_lag:]

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) + '/results/save-flight-12.18.2022_17.03.49.npy'
    with np.load(data_path) as d:
        timestamps = d['timestamps']
        states = d['states']
        targets = d['targets']
        controls = d['controls']
    
    states[0][[1,2,3,4],:] = states[0][[3,1,4,2],:]

    train_rate = 0.6
    train_sample = int(train_rate*targets.shape[2])
    train_x = np.hstack([states[0][:12,:train_sample].T, targets[0][:6,:train_sample].T])
    train_y = (controls[0][:,:train_sample].T - 0.084623) * 1000
    
    label_x = np.hstack([states[0][:12,train_sample:].T, targets[0][:6,train_sample:].T])
    label_y = controls[0][:,train_sample:].T

    idt = ControllerIndetifier()
    idt.train_model(train_x, train_y)
    pred_y = idt.predict(label_x)
    
    pred_y = pred_y / 1000 + 0.084623
    ts = range(pred_y.shape[0])

    fig = plt.figure('result')
    ax1 = fig.add_subplot(4,1,1)
    ax1.plot(ts, label_y[:,0], '-', ts, pred_y[:,0], ':')
    ax1.set_ylim([0.,0.2])
    ax2 = fig.add_subplot(4,1,2)
    ax2.plot(ts, label_y[:,1], '-', ts, pred_y[:,1], ':')
    ax2.set_ylim([0.,0.2])
    ax3 = fig.add_subplot(4,1,3)
    ax3.plot(ts, label_y[:,2], '-', ts, pred_y[:,2], ':')
    ax3.set_ylim([0.,0.2])
    ax4 = fig.add_subplot(4,1,4)
    ax4.plot(ts, label_y[:,3], '-', ts, pred_y[:,3], ':')
    ax4.set_ylim([0.,0.2])

    plt.show()
