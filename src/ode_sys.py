
''' control systems - ode simulation
  @link https://www.youtube.com/watch?v=yp5x8RMNi7o
'''
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt



def sys_ode(x, t):
  # set system constants
  c = 4 # damping constant
  k = 2 # spring stiffness constant
  m = 20 # point-mass
  F = 5 # input force into the system

  # compute state first derivative
  dx1 = x[1]
  dx2 = (F - c*x[1] - k*x[0])/m

  return [dx1, dx2]

def sim():
  # set constants
  t_0 = 0
  t_f = 60
  period = 0.1

  # set state initial condition
  x_init = [0, 0]

  # set a discrete time stamp
  t = np.arange(t_0, t_f, period)
  x = odeint(sys_ode, x_init, t)

  x1 = x[:,0]
  x2 = x[:,1]

  plt.plot(t,x1)
  plt.plot(t,x2)
  plt.title('Mass-Spring-Damper System')
  plt.xlabel('t')
  plt.ylabel('x(t)')
  plt.legend(['x1', 'x2'])
  plt.grid()
  plt.show()
