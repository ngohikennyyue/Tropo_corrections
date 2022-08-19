import sys
import os

current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
parent = os.path.dirname(parent)
sys.path.append(parent)

from extract_func.Extract_PTE_function import *
hawaii = pd.read_csv('../Hawaii/Hawaii_train_ifg_PTE_fixed_hgtlvs.csv').dropna()
west = pd.read_csv('../West/West_train_ifg_PTE_fixed_hgtlvs.csv').dropna()
east = pd.read_csv('../East/East_train_ifg_PTE_fixed_hgtlvs.csv').dropna()
print('Length Hawaii:', len(hawaii))
print('Length West:', len(west))
print('Length East:', len(east))

pd.concat([hawaii, west, east]).reset_index().to_feather('train_data.ftr')