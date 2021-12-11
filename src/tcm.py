''' test config module
'''
import os
import pickle as pk
import pandas as pd

''' NBUG
'''
from nbug import *

class test_configuration(object):
  def __init__(self,
               TEST_ID:str,
               OUT_DIR:str,
               OUT_DATA:str,
               CONF_DIR:str,
               SIM_DUR:float,
               output_labels:list,
               all_friction:float,
               cart_mass:float,
               pend_1_length:float,
               pend_1_mass:float,
               pend_1_moment:float,
               pend_2_length:float,
               pend_2_mass:float,
               pend_2_moment:float,
               K:list,
               Nbar:float):
    self.id = TEST_ID
    self.out_dir = OUT_DIR
    self.out_data = OUT_DATA
    self.data_path = OUT_DATA
    self.conf_dir = CONF_DIR
    self.sim_dur = SIM_DUR
    self.conf_path = CONF_DIR+TEST_ID+'.config'
    self.loss_path = OUT_DIR+TEST_ID+'_losses.log'
    self.out_labels = output_labels
    self.fric = all_friction
    self.m_c = cart_mass
    self.l_1 = pend_1_length
    self.m_1 = pend_1_mass
    self.pm_1 = pend_1_moment
    self.l_2 = pend_2_length
    self.m_2 = pend_2_mass
    self.pm_2 = pend_2_moment
    self.k = K
    self.nbar = Nbar

  # def load_data(self):
    # return pd.read_csv(self.out_data,)
# global functions
def pkl(t:test_configuration):
  if os.path.exists(t.conf_path):
    os.remove(t.conf_path)
  with open(t.conf_path, 'wb') as f:
    pk.dump(t, f)
    f.close()
  return

def unpkl(test_id:str, conf_dir:str):
  path = conf_dir+test_id+'.config'
  with open(path, 'rb') as f:
    test_config = pk.load(f)
    f.close()
  return test_config

def unpkl_file(path:str):
  with open(path, 'rb') as f:
    test_config = pk.load(f)
    f.close()
  return test_config





# EOF
