import matplotlib.animation as animation
import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

''' NBUG
'''

''' MOD CONFIG
'''

''' MODEL CONFIG
'''
_M1 = 10 # the cart
_M2 = 1  # mass 2
_T_END = 6.2

def main():

  sys = GEKKO() # declare an empty model
  sys.time = np.linspace(0,8,100)
  t_end = int(100.0*_T_END/8.0)

  # set model parameters
  m1 = sys.Param(value=_M1)
  m2 = sys.Param(value=_M2)
  final = np.zeros(len(sys.time))
  for i in range(len(sys.time)):
    if sys.time[i] < _T_END:
      final[i] = 0
    else:
      final[i] = 1
  final = sys.Param(value=final)

  # input var
  u = sys.Var(value=0)

  # state variables and initial conditions
  theta = sys.Var(value=0)
  q = sys.Var(value=0)
  y = sys.Var(value=-1)
  v = sys.Var(value=0)

  epsi = sys.Intermediate(m2/(m1+m2))

  # define state space model
  sys.Equation(y.dt() == v)
  sys.Equation(v.dt() == -epsi*theta + u)
  sys.Equation(theta.dt() == q)
  sys.Equation(q.dt() == theta - u)

  # final conditions
  sys.Obj(final*y**2)
  sys.Obj(final*v**2)
  sys.Obj(final*theta**2)
  sys.Obj(final*q**2)

  # constraints
  sys.fix(y,t_end,0.0)
  sys.fix(v,t_end,0.0)
  sys.fix(theta,t_end,0.0)
  sys.fix(q,t_end,0.0)

  # minimize input
  sys.Obj(0.0001*u**2)

  sys.options.IMODE = 6 #MPC
  sys.solve()

  # plot results
  plt.figure(figsize=(12,10))

  plt.subplot(221)
  plt.plot(sys.time, u.value, 'm', lw=1)
  plt.legend([r'$u$'],loc=1)
  plt.ylabel('force')
  plt.xlabel('time')
  plt.xlim(sys.time[0],sys.time[-1])

  plt.subplot(222)
  plt.plot(sys.time, v.value, 'g', lw=1)
  plt.legend([r'$v$'],loc=1)
  plt.ylabel('velocity')
  plt.xlabel('time')
  plt.xlim(sys.time[0],sys.time[-1])

  plt.subplot(223)
  plt.plot(sys.time, y.value, 'r', lw=1)
  plt.legend([r'$y$'],loc=1)
  plt.ylabel('position')
  plt.xlabel('time')
  plt.xlim(sys.time[0],sys.time[-1])

  plt.subplot(224)
  plt.plot(sys.time, theta.value, 'y', lw=1)
  plt.plot(sys.time, q.value, 'c', lw=1)
  plt.legend([r'$\theta$', r'$q$'], loc=1)
  plt.ylabel('angle')
  plt.xlabel('time')
  plt.xlim(sys.time[0],sys.time[-1])

  # set animation codec
  plt.rcParam['animation.html'] = 'html5'

  return # end of main

if __name__ == '__main__':
  main()
  print('\n\n\n--- end of main ---\n\n')
