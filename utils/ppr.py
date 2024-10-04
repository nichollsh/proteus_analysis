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

h_pl  = 6.626075540e-34    #Planck's constant
c_vac = 2.99792458e8       #Speed of light
k_B   = 1.38065812e-23     #Boltzman thermodynamic constant

def planck(wav, tmp):
    # Output value
    flx = 0.0

    # Convert nm to m
    wav = wav * 1.0e-9

    # Optimisation variables
    wav5 = wav*wav*wav*wav*wav
    hc   = h_pl * c_vac

    # Calculate planck function value [W m-2 sr-1 m-1]
    # http://spiff.rit.edu/classes/phys317/lectures/planck.html
    flx = 2.0 * hc * (c_vac / wav5) / ( np.exp(hc / (wav * k_B * tmp)) - 1.0)

    # Integrate solid angle (hemisphere), convert units
    flx = flx * np.pi * 1.0e-9 # [W m-2 nm-1]

    return flx

