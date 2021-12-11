# -*- coding: utf-8 -*-
''' control systems - double pendulum simulation
  @link https://www.youtube.com/watch?v=8ZZDNd4eyVI&t=1115s
'''
import numpy as np
import sympy as smp
import matplotlib
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from matplotlib.collections import LineCollection
import pandas as pd

plt.rcParams['animation.ffmpeg_path'] = '/home/smerx/anaconda3/envs/dpend/bin/ffmpeg' # Add the path of ffmpeg here!!


matplotlib.use("Agg")

''' NBUG
'''
from pprint import pprint as pp
from nbug import *


''' custom modules
'''
# from dvm import *

''' CONFIG
'''
trc_en = True
author = 'B.Mojra'
out_dir = '../out/'
out_dpend_gif = 'dpend.gif'
anim_coord_labels = ['t', 'x1','y1','x2','y2']
ode_out_labels = ['th1','dth1','th2','dth2']

''' ANIMATION CONFIG
'''
hist_lstyle = [('dotted')]
hist_lcolor = [('grey')]
hist_lwidth = [1]

def get_res_df(t, dX, L1, L2):
  t = t.reshape(-1,1)
  th1 = dX['th1'].to_numpy()
  th1 = th1.reshape(-1,1)
  th2 = dX['th2'].to_numpy()
  th2 = th2.reshape(-1,1)
  x1 =  L1*np.sin(th1)
  y1 = -L1*np.cos(th1)
  x2 =  L1*np.sin(th1) + L2*np.sin(th2)
  y2 = -L1*np.cos(th1) - L2*np.cos(th2)
  st()
  df = pd.concat([t,x1,y1,x2,y2], axis=1)
  df.columns=anim_coord_labels
  st()
  df['X'] = pd.concat([df.x1, df.x2], axis=1)
  df['Y'] = pd.concat([df.y1, df.y2], axis=1)
  return df

# now define the intended system of ode's
def dSdt(sys, t, g, m1, m2, L1, L2):
  th1, z1, th2, z2 = sys
  return [ dth1dt_fn(z1),
           dz1dt_fn(t,g,m1,m2,L1,L2,th1,th2,z1,z2),
           dth2dt_fn(z2),
           dz2dt_fn(t,g,m1,m2,L1,L2,th1,th2,z1,z2)]

def update(i, lines, dX:pd.DataFrame, lc):
  ''' anim config
  '''
  hist_lstyle = [('dotted')]
  hist_lcolor = [('grey')]
  hist_lwidth = [1]
  # set pendulum lines coord - solid black for l1 & l2
  lines = [[0, dX.x1[i], dX.x2[i]], [0, dX.y1[i], dX.y2[i]]]
  # set line style
  lstyles = [('solid')] * len(lines)
  lcolors = ['k'] * len(lines)
  lwidths = [2] * len(lines)

  if trc_en == True:
    # first obtain the hist points
    nprint('in update()')
    nprint('X', dX.X)
    nprint('Y', dX.Y)
    st()

    # set dashed grey lines for m1 & m2 pos hist and append to line collection
    new_x = dX.X[:i,:]
    new_y = dX.Y[:i,:]
    hlines = [np.column_stack([xi, yi]) for xi, yi in zip(new_x, new_y)]

    lstyles.append([hist_lstyle * len(hlines)])
    lcolors.append([hist_lcolor * len(hlines)])
    lwidths.append([hist_lwidth * len(hlines)])
    lines.append(hlines)
    # --- end if trc_en == True:

  # Now we update the lines in our LineCollection
  lc.set_segments(lines)
  lc.set_linestyles(lstyles)
  lc.set_colors(lcolors)
  lc.set_linewidths(lwidths)
  return lc

if __name__ == '__main__':

  # setup symbolic math
  t, g = smp.symbols('t g')
  m1, m2 = smp.symbols('m1 m2')
  L1, L2 = smp.symbols('L1 L2')

  # setup \theta_1 and \theta_2 but they will be functions of time
  th1, th2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)
  # declare symbolic time-variant variables
  th1 = th1(t)
  th2 = th2(t)

  # define symbolic first and second derivatives
  th1_d = smp.diff(th1, t)
  th2_d = smp.diff(th2, t)
  th1_dd = smp.diff(th1_d, t)
  th2_dd = smp.diff(th2_d, t)

  # define state variables
  x1 = L1*smp.sin(th1)
  y1 = -L1*smp.cos(th1)
  x2 = L1*smp.sin(th1)+L2*smp.sin(th2)
  y2 = -L1*smp.cos(th1)-L2*smp.cos(th2)

  # setup kinetic and potential energy equations
  # kinetic energy
  T1 = 1/2 * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2 )
  T2 = 1/2 * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2 )
  T = T1 + T2

  # potential energy
  v1 = m1 * g * y1
  v2 = m2 * g * y2
  v = v1 + v2

  # setup the La'Grangian
  L = T - v

  # print(L)

  # get LaGragian's first derivative
  LE1 = smp.diff(L, th1) - smp.diff(smp.diff(L, th1_d), t).simplify()
  LE2 = smp.diff(L, th2) - smp.diff(smp.diff(L, th2_d), t).simplify()

  # the system derivative is already in standard form and we proceed with solving it
  sols = smp.solve([LE1, LE2], (th1_dd, th2_dd), simplify=False, rational=False)

  # this code solves the system for second order ode
  # sols[th2_dd]

  # now, we convert symbolic expressions (i.e. second order derivative) to
  # a numerical functions so they can be used to calculate state dynamics, using
  # smp.lambdify() function.
  # we also convert two second order state variables to four first order state
  # variables, using z1 and z2.
  dz1dt_fn = smp.lambdify((t,g,m1,m2,L1,L2,th1,th2,th1_d,th2_d), sols[th1_dd])
  dz2dt_fn = smp.lambdify((t,g,m1,m2,L1,L2,th1,th2,th1_d,th2_d), sols[th2_dd])
  dth1dt_fn = smp.lambdify(th1_d, th1_d)
  dth2dt_fn = smp.lambdify(th2_d, th2_d)

  # setup system constants
  t = np.linspace(0, 40, 1001)
  g = 9.81
  m1 = 4
  m2 = 2
  L1 = 2
  L2 = 1

  # run the sim
  dX = odeint(dSdt, y0=[1,-3,-1,5], t=t, args=(g,m1,m2,L1,L2))
  dX = pd.DataFrame(dX, columns=ode_out_labels, dtype=float)
  # t = pd.DataFrame(t, columns=['t'], dtype=float)
  # dX = pd.concat([t,dX], axis=1)

  plt.plot(dX)
  plt.legend(dX.columns)
  plt.savefig('system_dynamics.png')
  # plt.show()
  plt.close()
  # st()

  # Set up formatting for the movie files
  Writer = animation.writers['ffmpeg']
  writer = Writer(fps=25, metadata=dict(artist=author), bitrate=1800)
  fig, ax = plt.subplots(1,1, figsize=(8,8))
  ax.set_facecolor('w')
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([]) # enable to hide ticks



  lines = []
  lc = LineCollection(lines)
  ax.add_collection(lc)
  ax.set_xlim(-4, 4)
  ax.set_ylim(-4, 4)
  ax.set(xlabel="x", ylabel="y", aspect="equal")

  st()
  m12_xy = get_res_df(t, dX, L1, L2)

  st()

  # set line to be plotted
  ani = animation.FuncAnimation(fig, update, frames=m12_xy.shape[0], interval=50,\
    fargs=(lines, m12_xy, trc_en, lc,))

  # canvas.draw()


  if os.path.exists(out_dir+out_dpend_gif):
    os.remove(out_dir+out_dpend_gif)
    print(shorthead+'removed existing dpend.gif file...')
  ani.save(out_dir+out_dpend_gif, writer=writer)
  print(shorthead+'saved new dpend.gif to file...')

# https://youtu.be/8ZZDNd4eyVI?t=640

  # pp(th1_dd)
  st()

  print('--- end of main ---')
