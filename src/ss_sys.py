
''' control systems - ode simulation
  @link https://www.youtube.com/watch?v=yp5x8RMNi7o
'''
import numpy as np
from matplotlib import pyplot as plt
import control


def ss_sys(x, t):


  # compute state first derivative
  dx1 = x[1]
  dx2 = (F - c*x[1] - k*x[0])/m

  return [dx1, dx2]

def sim():
  # set constants
  t_0 = 0
  t_f = 60
  dt = 0.1

  # set a discrete time stamp
  t = np.arange(t_0, t_f, dt)

  # set system constants
  c = 4 # damping constant
  k = 2 # spring stiffness constant
  m = 20 # point-mass
  F = 5 # input force into the system

  # set state initial condition
  x_init = [0, 0]

  # set up state space matrices describing the control system
  A = [[0, 1], [-k/m, c/m]]
  B = [[0], [1/m]]
  C = [[1, 0]]
  D = [0] # feedforward vector

  # instantiate the system and run the simulation
  sys = control.ss(A, B, C, D, dt)
  t, y, x = control.forced_respone(sys, t, F)
  return t, y, x
