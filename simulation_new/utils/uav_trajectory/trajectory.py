#!~/anaconda3/envs/teb/bin/python

import numpy as np

def normalize(v):
  norm = np.linalg.norm(v)
  assert norm > 0
  return v / norm


class Polynomial:
  def __init__(self, p):
    self.p = p

  # evaluate a polynomial using horner's rule
  def eval(self, t):
    assert t >= 0
    x = 0.0
    for i in range(0, len(self.p)):
      x = x * t + self.p[len(self.p) - 1 - i]
    return x

  # compute and return derivative
  def derivative(self):
    return Polynomial([(i+1) * self.p[i+1] for i in range(0, len(self.p) - 1)])


class TrajectoryOutput:
  def __init__(self):
    self.pos = None   # position [m]
    self.vel = None   # velocity [m/s]
    self.acc = None   # acceleration [m/s^2]
    self.omega = None # angular velocity [rad/s]
    self.yaw = None   # yaw angle [rad]


# 4d single polynomial piece for x-y-z-yaw, includes duration.
class Polynomial4D:
  def __init__(self, duration, px, py, pz, pyaw):
    self.duration = duration
    self.px = Polynomial(px)
    self.py = Polynomial(py)
    self.pz = Polynomial(pz)
    self.pyaw = Polynomial(pyaw)

  # compute and return derivative
  def derivative(self):
    return Polynomial4D(
      self.duration,
      self.px.derivative().p,
      self.py.derivative().p,
      self.pz.derivative().p,
      self.pyaw.derivative().p)

  def eval(self, t):
    result = TrajectoryOutput()
    # flat variables
    result.pos = np.array([self.px.eval(t), self.py.eval(t), self.pz.eval(t)])
    result.yaw = self.pyaw.eval(t)

    # 1st derivative
    derivative = self.derivative()
    result.vel = np.array([derivative.px.eval(t), derivative.py.eval(t), derivative.pz.eval(t)])
    dyaw = derivative.pyaw.eval(t)

    # 2nd derivative
    derivative2 = derivative.derivative()
    result.acc = np.array([derivative2.px.eval(t), derivative2.py.eval(t), derivative2.pz.eval(t)])

    # 3rd derivative
    derivative3 = derivative2.derivative()
    jerk = np.array([derivative3.px.eval(t), derivative3.py.eval(t), derivative3.pz.eval(t)])

    thrust = result.acc + np.array([0, 0, 9.81]) # add gravity

    z_body = normalize(thrust)
    x_world = np.array([np.cos(result.yaw), np.sin(result.yaw), 0])
    y_body = normalize(np.cross(z_body, x_world))
    x_body = np.cross(y_body, z_body)

    jerk_orth_zbody = jerk - (np.dot(jerk, z_body) * z_body)
    h_w = jerk_orth_zbody / np.linalg.norm(thrust)

    result.omega = np.array([-np.dot(h_w, y_body), np.dot(h_w, x_body), z_body[2] * dyaw])
    return result


class Trajectory:
  def __init__(self, coef_mat=None, order=10):
    self.polynomials = None
    self.duration = None
    self.order = order
    if coef_mat is not None:
      self.loadmat(np.array(coef_mat), order)

  def n_pieces(self):
    return len(self.polynomials)
  
  def total_time_before_n_piece(self, n_piece: int):
    """
    Determine the total time before polynomial[n_piece]
    :param n_piece (int): the index of polynomials
    return:
      total time
    """
    assert n_piece < self.n_pieces()
    total_time = 0.
    for i in range(n_piece):
      total_time += self.polynomials[i].duration
    return total_time
  
  def time_at_which_piece(self, time):
    """
    Determine which piece is the current reference signal in
    :param time (float > 0)
    return:
      piece_index
    """
    assert time >= 0
    current_t = 0.0
    for i in range(len(self.polynomials)):
      current_t += self.polynomials[i].duration
      if time <= current_t:
        return i
    return i
  
  def loadmat(self, mat: np.ndarray, order=10):
    """
    Construct polynomial trajectory from matrix
    :param mat (ndarray, ndim==2): matrix containing duration and coefficients of polynomials
      [[duration0, x0_0, x0_1, ..., x0_9, y0_0, y0_1, ..., y0_9, z0_0, z0_1, ..., z0_9, (yaw0_0, ...)]
      [duration1, x1_0, x1_1, ..., x1_9, y1_0, y1_1, ..., y1_9, z1_0, z1_1, ..., z1_9, (yaw1_0, ...)]
      ...]
    :param order: order of the polynomials, highest coeff is x^(order-1)
    """
    if mat.ndim == 1: mat = np.array([mat])
    if mat.shape[1] == 3*order+1:
      self.polynomials = [Polynomial4D(row[0], row[1:order+1], row[order+1:2*order+1], row[2*order+1:3*order+1], np.zeros(order)) for row in mat]
    elif mat.shape[1] == 4*order+1:
      self.polynomials = [Polynomial4D(row[0], row[1:order+1], row[order+1:2*order+1], row[2*order+1:3*order+1], row[3*order+1:4*order+1]) for row in mat]
    else:
      print("Invalid matrix shape!")
      exit(-1)
    
    self.duration = np.sum(mat[:,0])
    self.order = 10

  def loadcsv(self, filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=range(33))
    if data.ndim == 1: data = np.array([data])
    self.polynomials = [Polynomial4D(row[0], row[1:9], row[9:17], row[17:25], row[25:33]) for row in data]
    self.duration = np.sum(data[:,0])
    self.order = 8

  def eval(self, t, maximum_allowed=True):
    assert t >= 0
    if maximum_allowed:
      t = min(self.duration, t)
    else:
      assert t <= self.duration

    current_t = 0.0
    for p in self.polynomials:
      if t <= current_t + p.duration:
        return p.eval(t - current_t)
      current_t = current_t + p.duration
