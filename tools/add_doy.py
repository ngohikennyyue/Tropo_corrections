import os
import sys
import numpy as np
import glob
import rasterio
import matplotlib.pyplot as plt
import math
import pandas as pd
from datetime import datetime
current = os.path.dirname(os.path.realpath('extract_func'))
parent = os.path.dirname(current)
sys.path.append(parent)
from extract_func.Extract_PTE_function import *

addDOY('PTE_vert_fixed_hgtlvs.csv')
addDOY('PTE_vert_fixed_hgtlvs_slope.csv')
addDOY('PTE_vert_fixed_hgtlvs_cloud.csv')