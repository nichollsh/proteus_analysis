import numpy as np
import glob, os, re
import netCDF4 as nc 
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# List of volatiles
volatile_species = ["H2O", "CO2", "H2", "CO", "CH4", "N2", "O2", "S", "He"]
volatile_colors  = {"H2O": "#C720DD",
                    "CO2": "#D24901",
                    "H2" : "#008C01",
                    "CH4": "#027FB1",
                    "CO" : "#D1AC02",
                    "N2" : "#870036",
                    "S"  : "#FF8FA1",
                    "O2" : "#00008B",
                    "He" : "#30FF71"
                    }

# Get paths to case outputs for a given grid parent folder
def get_cases(pgrid_dir:str):
    # Case folders
    p = os.path.abspath(pgrid_dir)
    case_dirs = glob.glob(p + "/case_*")
    if len(case_dirs) == 0:
        print("WARNING: Case folders not found - check your pgrid path!")
    
    # Sort by case index
    idxs = []
    for c in case_dirs:
        idxs.append(int(c.split("/")[-1].split("_")[-1]))
    mask = np.argsort(idxs)
    return [os.path.abspath(case_dirs[i]) for i in mask]

# Get simulation output years for which we have json data
def get_json_years(case_dir:str):
    output_files = glob.glob(os.path.join(case_dir,"data")+"*.json")
    output_times = [ int(str(f).split('/')[-1].split('.')[0]) for f in set(output_files)]
    mask = np.argsort(output_times)
    return np.array(output_times)[mask]

# Get simulation output years for which we have netcdf data
def get_nc_years(case_dir:str):
    output_files = glob.glob(case_dir+"/data/*_atm.nc")
    output_times = [ int(str(f).split('/')[-1].split('_')[0]) for f in set(output_files)]
    mask = np.argsort(output_times)
    return np.array(output_times)[mask]

# Get simulation output years for which we have both json and netcdf data
def get_common_years(json_years:list, nc_years:list):
    both = set(json_years).intersection(nc_years)
    mask = np.argsort(list(both))
    return np.array(both)[mask]

# Get status of cases
def get_statuses(pgrid_dir:str):
    p = os.path.abspath(pgrid_dir)
    case_dirs = glob.glob(p + "/case_*")
    statuses = []
    for c in case_dirs:
        status_file = os.path.join(os.path.abspath(c),"status")
        with open(status_file,'r') as hdl:
            statuses.append(int(hdl.readlines()[0]))
    return statuses

def read_nc(nc_file:str):

    ds = nc.Dataset(nc_file)
    vars = list(ds.variables.keys())

    nlev_c = len(ds.variables["p"][:])
    nlev_l = len(ds.variables["pl"][:])
    tsurf  = float(ds.variables["tstar"][:])
    psurf  = float(ds.variables["pl"][-1])
    gases  = [str(bytearray(s).decode()).strip() for s in ds.variables["gases"][:]]
    x_raw  = np.array(ds.variables["x_gas"][:]).T
    x_gas  = {}
    for i,g in enumerate(gases):
        x_gas[g] = x_raw[i]

    data = {
        "nlev_c":       nlev_c,
        "nlev_l":       nlev_l,
        "tstar":        tsurf,
        "psurf":        psurf,
        "gases":        gases,
        "x_gas":        x_gas,
    }

    # All array-like variables
    for k in vars:
        if k in data.keys():
            continue 
        data[k] = np.array(ds.variables[k][:], dtype=float)
    
    ds.close()
    return data

# Is value numeric?
def is_float(v):
    try:
        _=float(v)
    except ValueError:
        return False
    return True

# Get case configuration file (value as strings)
def read_config(case_dir:str):
    f = case_dir+"/init_coupler.cfg"
    cfg = {}
    with open(f,'r') as hdl:
        lines = hdl.readlines()
    for l in lines:
        ll = l[:-1] # remove new line char
        ll =  re.sub(r"\s+", "", ll) # remove whitespace
        if (len(ll) == 0) or (ll[0] == "#"):
            continue
        if "#" in ll:
            ll = ll.split("#")[0]  # remove comment
        ll = ll.split("=")
        k = str(ll[0]); v = str(ll[1])
        if is_float(v):
            v = float(v)
        cfg[k] = v
    return cfg

def read_helpfile(case_dir:str, atm=True):
    f = case_dir+"/runtime_helpfile.csv"
    df_raw = pd.read_csv(f,sep='\s+')
    if atm:
        df = df_raw.loc[df_raw['Input']=='Atmosphere'].drop_duplicates(subset=['Time'], keep='last')
    else:
        df = df_raw.loc[df_raw['Input']=='Interior'].drop_duplicates(subset=['Time'], keep='last')
    return df

# Function to access nested dictionaries
def recursive_get(d, keys):
    if len(keys) == 1:
        return d[keys[0]]
    return recursive_get(d[keys[0]], keys[1:])

# Get the scaled values for a particular quantity
def get_dict_values( self, keys, fmt_o='' ):
    dict_d = recursive_get( self.data_d, keys )
    scaling = float(dict_d['scaling'])
    if len( dict_d['values'] ) == 1:
        values_a = float( dict_d['values'][0] )
    else:
        values_a = np.array( [float(value) for value in dict_d['values']] )
    scaled_values_a = scaling * values_a
    if fmt_o:
        scaled_values_a = fmt_o.ascale( scaled_values_a )
    return scaled_values_a

def load_cvars(cases):
    ncases = len(cases)
    pbar = tqdm(desc="Configs", total=ncases)
    cfgs = []
    for i in range(ncases):
        cfgs.append(read_config(cases[i]))
        pbar.update(1)
    pbar.close()
    keys = cfgs[0].keys()
    cvars = {}
    for k in keys:
        values = []
        for i in range(ncases):
            values.append(cfgs[i][k])
        if len(values) > 0:
            cvars[k] = np.array(values,dtype=type(values[0]))
    return cvars 
    
def load_helpfiles(cases):
    helps = []
    hvars = {}
    ncases = len(cases)
    pbar = tqdm(desc="Helpfiles", total=ncases)
    for i in range(ncases):
        helps.append(read_helpfile(cases[i]))
        pbar.update(1)
    pbar.close()

    for k in helps[0].keys():
        tmp_arr = []
        for h in helps:
            tmp_arr.append(np.array(h.loc[:,k]))
        hvars[k] = np.array(tmp_arr,dtype=object)
    return helps, hvars

# Get value of 'key' at 'idx' for each case in 'hvars', which is a ragged array
def access_hvars(hvars,key,idx):
    rag = hvars[key]

    # for each case...
    vals = []
    for i in range(np.shape(rag)[0]):
        vals.append(rag[i][idx])

    # return array value of 'key' for each case, at 'idx'
    return np.array(vals,dtype=type(rag[0][0]))


def load_netcdfs_end(cases):
    ncases = len(cases)
    endn = []
    pbar = tqdm(desc="NetCDFs", total=ncases)
    for i in range(ncases):
        t = get_nc_years(cases[i])[-1]
        n = read_nc(cases[i]+"/data/%d_atm.nc" % t)
        endn.append(n)
        pbar.update(1)
    pbar.close()
    endn = np.array(endn)

    return endn


# Interpolate a variable onto a higher resolution grid (2 dimensions only)
def interp_2d(x_locs, y_locs, z_vals, npoints, method="linear", scaling=True):

    '''
    Depends on Scipy's RegularGridInterpolator class.

    Inputs
        x_locs  : Location of points on x-axis (1D)
        y_locs  : Location of points on y-axis (1D)
        z_vals  : Values of points (2D)
        method  : Interpolation method
        scaling : Use log-scaling if variable range is >2 OOM

    Outputs
        xxo     : Original x-locations (2D grid)
        yyo     : Original y-locations (2D grid)
        xxi     : Interpolated x-locations (2D grid)
        yyi     : Interpolated y-locations (2D grid)
        zzi     : Interpolated z_vals (2D grid)
    '''

    # check dimensions
    if (len(x_locs) < 3) or (len(y_locs) < 3):
        raise Exception("Cannot interolate grid with a resolution less than 3")
    
    # check scaling
    def _is_log(_arr):
        if not scaling:
            return False
        _arr = np.array(_arr)
        if np.any(_arr <= 0):
            return False
        return np.log10(np.amax(_arr)/np.amin(_arr)) > 2.0  # range is more than 2 order of magnitude
    
    xlog = _is_log(x_locs)
    if xlog:
        x_locs = np.log10(x_locs)
    xmin = np.amin(x_locs)
    xmax = np.amax(x_locs)

    ylog = _is_log(y_locs)
    if ylog:
        y_locs = np.log10(y_locs)
    ymin = np.amin(y_locs)
    ymax = np.amax(y_locs)

    zlog = _is_log(z_vals)
    if zlog:
        z_vals = np.log10(z_vals)

    # input samples
    xxo, yyo = np.meshgrid(x_locs, y_locs, indexing='ij')

    # generate interpolator
    interp = RegularGridInterpolator((x_locs, y_locs), z_vals, bounds_error=False, fill_value=None, method=method)

    # grid to interpolate at
    xi = np.linspace(xmin, xmax, npoints)
    yi = np.linspace(ymin, ymax, npoints)

    # do interpolation
    xxi, yyi = np.meshgrid(xi,yi,indexing='ij')
    zzi = interp((xxi,yyi))

    if xlog:
        xxi = 10.0 ** xxi
        xxo = 10.0 ** xxo
    if ylog:
        yyi = 10.0 ** yyi
        yyo = 10.0 ** yyo
    if zlog:
        zzi = 10.0 ** zzi
    return xxo,yyo,xxi,yyi,zzi

