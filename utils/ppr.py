import netCDF4 as nc
import numpy as np

def read_postproc(f):
    data = {}
    ds = nc.Dataset(f+"/ppr.nc")

    vars = ["ba_D_LW", "ba_U_LW", "ba_N_LW", "ba_D_SW", "ba_U_SW", "ba_N_SW"]
    for k in vars:
        data[k] = np.array(ds.variables[k][:,:], dtype=float)

    vars = ["time", "bandmin", "bandmax"]
    for k in vars:
        data[k] = np.array(ds.variables[k][:], dtype=float)

    data["nsamps"] = len(ds.dimensions["nsamps"])
    data["nbands"] = len(ds.dimensions["nbands"])

    data["original_model"] = str(ds.original_model)

    ds.close()
    return data
