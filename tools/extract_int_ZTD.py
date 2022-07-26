import sys
import os
current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *
from extract_func.Extract_int_wm_function import *

df = pd.read_csv('NewPTE_vert_fixed_hgtlvs.csv')
df = df.dropna()

extract_inter_param(df)

print('Finished extraction')
