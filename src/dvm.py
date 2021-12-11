''' data visualization mod (viz)
  @author Bardia Mojra - 1000766739
  @brief ee-5323 - nonlinear systems - final project
  @date 11/11/21
'''

import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib import cm
# import seaborn as sns
# sns.set()
import os


''' NBUG
'''
from pdb import set_trace as st
from pprint import pprint as pp
from nbug import *

''' matplotlib config
'''
matplotlib.pyplot.ion()
plt.style.use('ggplot')
colors=['yellowgreen','firebrick', 'royalblue', 'goldenrod', 'mediumorchid']
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }

def plot_df(df:pd.DataFrame,
            plot_title:str,
            labels:str,
            test_id:str,
            # data_path:str,
            out_dir:str,
            start=0,
            end=None,
            fig_fname=None,
            show=False,
            save=True,
            range_padding:float=0.5,
            figsize=[6,6]):
  if test_id is None:
    eprint('plot_df(): no test_id given... ')
  if out_dir is None:
    eprint('plot_df(): no output dirctory given... ')
  if end is None:
    end = df.index.size-1
  if labels is None:
    labels = df.columns
  else:
    df = df[labels].copy()
    labels = df.columns

  # set plot colors
  # cmap = cm.get_cmap('plasma', 15)
  # plot_colors = iter(cmap(np.linspace(0, 1, 15)))
  # plot_colors = iter([plt.cm.tab100(i) for i in range(20)])
  # print(df.head(5))


  # plot
  fig = plt.figure(figsize=figsize)
  xy = df.to_numpy(copy=True)
  for i, label in enumerate(labels):
    if i == 0: pass
    else:
      plt.plot(xy[:,0], xy[:,i],label=label, color=colors[i])
  plt.legend(loc='best')
  plt.grid(True)
  plt.xlabel('time', fontdict=font)
  plt.ylabel('magnitude', fontdict=font)
  # set title
  if plot_title != '_':
    tit = test_id+' - '+plot_title
    plt.title(tit, y=0.95, fontdict=font)
  plot_title = plot_title.replace(' ', '_')
  plot_title = plot_title.replace('.', '-')
  # save and show image
  if save==True and out_dir is not None:
    fig_name = out_dir+test_id+'_'+plot_title+'.png'
    plt.savefig(fig_name, bbox_inches='tight',dpi=400)
    print(longhead+'saving figure: '+fig_name)
    csv_name = out_dir+test_id+'_'+plot_title+'.csv'
    df.to_csv(csv_name, columns=df.columns, index=False)
    print(longhead+'saving figure: '+csv_name)
  if show==True:
    plt.show()
  else:
    pass
    # plt.close()
  return


def get_losses(df:pd.DataFrame,
               dataPath:str,
               lossPath:str,
               save_en:bool=True,
               prt_en:bool=True):
    L1 = list()
    L2 = list()
    loss_df = df.iloc[:,[3,5]].copy()
    for i in range(len(loss_df.index)):
      state_l1 = 0.0
      state_l2 = 0.0
      # nprint('row', i)
      # st()

      for j in range(len(loss_df.columns)):
        l1 = abs(180*loss_df.iloc[i,j]) # convert to degrees
        l2 = (180*loss_df.iloc[i,j]) ** 2
        # nprint('l1, l2', l1, l2)
        state_l1 += l1
        state_l2 += l2
      # nprint('state_l1, state_l2', state_l1, state_l2)
      L1.append(state_l1)
      L2.append(state_l2)
    # concatenate
    L1_df = pd.DataFrame(L1, columns=['L1'])
    L2_df = pd.DataFrame(L2, columns=['L2'])
    df = pd.concat([df,L1_df, L2_df], axis=1)
    if dataPath is not None and os.path.exists(dataPath):
      os.remove(dataPath)
      df.to_csv(dataPath, columns=df.columns, index=False)
      print(longhead+'saving figure: '+dataPath)
    if save_en==True and lossPath is not None:
      with open(lossPath, 'a+') as f:
        L1_str = shorthead+f"L1 (total): {df['L1'].sum()}"
        L2_str = shorthead+f"L2 (total): {df['L2'].sum()}"
        f.write(L1_str)
        f.write(L2_str+'\n\n')
        f.close()
    return df

def print_losses(df: pd.DataFrame):
  print(shorthead+"L1 (total): ", df['L1'].sum())
  print(shorthead+"L2 (total): ", df['L2'].sum())

  return
