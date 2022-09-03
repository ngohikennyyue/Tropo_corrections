import os
import skgstat as skg
import glob
from extract_func.Extract_PTE_function import *

ifg_files = glob.glob('InSAR/Large_scale/Hawaii/Extracted/unwrappedPhase/*[0-9]')

for file in ifg_files[:1]:
    name = file.split('/')[-1]
    ifg, grid = focus_bound(file, -155.9, 18.9, -154.9, 19.9)
    V = skg.Variogram(grid[::100], ifg.ravel()[::100])
    print(V)
    fig = V.plot()
    plt.savefig('{}'.format(name))

