import numpy as np
import glob, os, re
import netCDF4 as nc 
import pandas as pd
from tqdm import tqdm

# List of volatiles
volatile_species = ["H2O", "CO2", "H2", "CO", "CH4", "N2", "O2", "S", "He"]
volatile_colors  = {"H2O": "#8db4cb",
                    "CO2": "#ce5e5e",
                    "H2" : "#a0d2cb",
                    "CH4": "#eb9194",
                    "CO" : "#ff11ff",
                    "N2" : "#c29fb2",
                    "S"  : "#f1ca70",
                    "O2" : "#57ccda",
                    "He" : "#acbbbf"}

# Get paths to case outputs for a given grid parent folder
def get_cases(pgrid_dir:str):
    p = os.path.abspath(pgrid_dir)
    case_dirs = glob.glob(p + "/case_*")
    return [os.path.abspath(c) for c in case_dirs]

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
            v = cfgs[i][k]
            if is_float(v):
                values.append(v)
        if len(values) > 0:
            cvars[k] = np.array(values)
    return cvars 
    
def load_helpfiles(cases):
    helps = []
    cvars = {}
    ncases = len(cases)
    pbar = tqdm(desc="Helpfiles", total=ncases)
    for i in range(ncases):
        helps.append(read_helpfile(cases[i]))
        pbar.update(1)
    pbar.close()
    for v in volatile_species:
        v_arr = []
        for h in helps:
            v_arr.append(h.iloc[-1][v+"_mr"])
        cvars[v+"_surf"] = np.array(v_arr)

    return helps, cvars

def load_netcdfs(cases):
    ncases = len(cases)
    endt = []
    endn = []
    endp = []
    pbar = tqdm(desc="NetCDFs", total=ncases)
    for i in range(ncases):
        t = get_nc_years(cases[i])[-1]
        n = read_nc(cases[i]+"/data/%d_atm.nc" % t)
        endt.append(t)
        endn.append(n)
        endp.append(n["psurf"])
        pbar.update(1)
    pbar.close()
    endt = np.array(endt)
    endp = np.array(endp)
    endn = np.array(endn)

    return endt, endp, endn

