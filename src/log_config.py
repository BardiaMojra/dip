
import os


# custom mods
from nbug import *
from tcm import *

CONF_DIR = '../config/'
CONF_LOG_PATH = '../config/test_configs.log'


def get_confList(c):
  return [c.id,
          c.sim_dur,
          c.fric,
          c.m_c,
          c.l_1,
          c.m_1,
          c.pm_1,
          c.l_2,
          c.m_2,
          c.pm_2]

def get_cStr(cl:list):
  b = ', '
  tempstr = cl[0]+b+str(cl[1])+b+str(cl[2])+b+str(cl[3])+b+str(cl[4])+b+str(cl[5])+b+str(cl[6])+b+str(cl[7])+b+str(cl[8])+b+str(cl[9])+'\n'
  print(tempstr)
  return tempstr


def get_header():
  return str('id, sim_dur, fric, m_c, l_1, m_1, pm_1, l_2, m_2, pm_2 \n')


if __name__ == '__main__':

  if os.path.exists(CONF_LOG_PATH):
    os.remove(CONF_LOG_PATH)
  configs = os.listdir(CONF_DIR)
  configs = sorted(configs)
  header = get_header()
  with open(CONF_LOG_PATH, 'w') as clog:
    clog.write(header)
    clog.close()
  print(header)
  for f in configs:
    TEST_ID = f.replace('.config', '')
    c = unpkl(TEST_ID, CONF_DIR)
    cList = get_confList(c)
    cStr = get_cStr(cList)
    with open(CONF_LOG_PATH, 'a') as clog:
      clog.write(cStr)
  clog.close()
