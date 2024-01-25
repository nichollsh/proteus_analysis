import glob, pathlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sci_colormaps = {}
for g in glob.glob(str(pathlib.Path(__file__).parent.absolute())+"/colormaps/*.txt"):
    cm_data = np.loadtxt(g)
    name = g.split('/')[-1].split('.')[0]
    sci_colormaps[name]      = LinearSegmentedColormap.from_list(name, cm_data)
    sci_colormaps[name+"_r"] = LinearSegmentedColormap.from_list(name, cm_data[::-1])
    