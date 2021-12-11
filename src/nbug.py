''' nbug mod
  @author Bardia Mojra - 1000766739
'''

import sys
from pdb import set_trace as st
# enable nbug print
nprt_enComp     = True
nprt_en         = True
nprt_en2        = True
nst_en          = True

longhead = '\n  \--->> '
shorthead = '\--->> '
longtail = '\n\n'
shorttail = '\n'
attn = 'here ----------- <<<<<\n\n'  #print(longhead+attn)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def nprt(string:str, *args):
  if nprt_enComp is True:
    print(shorthead+(string+str(*args)));

def nprint(string:str, *args):
  if nprt_en is True:
    print(shorthead+(string+': ')); print(*args);

def nprint_2(string:str, *args):
  if nprt_en2 is True:
    print(shorthead+(string+': ')); print(*args);

def lh():
  print(longhead);

def nst(fname:str):
  if nst_en:
    print(fname)
    st()

# EOF
