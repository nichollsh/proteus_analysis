import numpy as np
import glob, os, re
import netCDF4 as nc
import pandas as pd
from attrs import asdict
from scipy.interpolate import RegularGridInterpolator, griddata
from tqdm import tqdm
from proteus.config import read_config_object, Config

def latexify(s:str):
    out = ""
    for c in s:
        if str(c).isdigit():
            out += "$_%d$"%int(c)
        else:
            out += c

    return out

# Get paths to case outputs for a given grid parent folder
def get_cases(pgrid_dir:str):
    # Case folders
    p = os.path.abspath(pgrid_dir)
    case_dirs = glob.glob(p + "/case_*")
    if len(case_dirs) == 0:
        print("WARNING: Case folders not found - check your pgrid path!")
        print(pgrid_dir)
        raise

    # Sort by case index
    idxs = []
    for c in case_dirs:
        idxs.append(int(c.split("/")[-1].split("_")[-1]))
    mask = np.argsort(idxs)
    return [os.path.abspath(case_dirs[i]) for i in mask]

# Get simulation output years for which we have json data
def get_json_years(case_dir:str):
    output_files = glob.glob(os.path.join(case_dir,"data")+"/*.json")
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
    case_nums = [int(s.split("_")[-1]) for s in case_dirs]

    statuses = {}
    for i,c in enumerate(case_dirs):
        this_case = case_nums[i]
        status_file = os.path.join(os.path.abspath(c),"status")
        with open(status_file,'r') as hdl:
            statuses[this_case] = int(hdl.readlines()[0])

    return statuses

def readncdf(f, verbose=False):

    if verbose: print("Reading",f)

    ds = nc.Dataset(f)
    vars = list(ds.variables.keys())

    nlev_c = len(ds.variables ["p"][:])
    nlev_l = len(ds.variables["pl"][:])
    tsurf  = float(ds.variables["tmp_surf"][:])
    psurf  = float(ds.variables["pl"][-1])
    gases  = [str(bytearray(s).decode()).strip() for s in ds.variables["gases"][:]]
    asf    = float(ds.variables["toa_heating"][:])

    data = {
        "nlev_c":       nlev_c,
        "nlev_l":       nlev_l,
        "tmp_surf":     tsurf,
        "psurf":        psurf,
        "gases":        gases,
        "toa_heating":  asf,
    }

    if verbose: print(gases)

    for k in vars:
        if k in data.keys():
            continue
        var = ds.variables[k][:]
        try:
            data[k] = np.array(ds.variables[k][:], dtype=float)
        except:
            continue

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
def read_config(case_dir:str, extension="toml"):
    return asdict(read_config_object(case_dir+"/init_coupler.toml"))

def descend_set(config:Config, key:str, val):
    bits = tuple(key.split("."))
    depth = len(bits)-1
    if depth == 0:
        config[bits[0]] = val
    elif depth == 1:
        config[bits[0]][bits[1]] = val
    elif depth == 2:
        config[bits[0]][bits[1]][bits[3]] = val
    else:
        raise Exception("Requested key is too deep for configuration tree")

def descend_get(config:Config, key:str):
    bits = tuple(key.split("."))
    depth = len(bits)-1
    if depth == 0:
        return config[bits[0]]
    elif depth == 1:
        return config[bits[0]][bits[1]]
    elif depth == 2:
        return config[bits[0]][bits[1]][bits[2]]
    else:
        raise Exception("Requested key is too deep for configuration tree")

def read_helpfile(case_dir:str):
    try:
        df = pd.read_csv(case_dir+"/runtime_helpfile.csv",sep=r'\s+')
    except EmptyDataError as e:
        print("Empty helpfile: "+case_dir)
        df = []
    return df

# Function to access nested dictionaries
def _recursive_get(d, keys):
    if len(keys) == 1:
        return d[keys[0]]
    return _recursive_get(d[keys[0]], keys[1:])

# Get the scaled values for a particular quantity
def get_dict_values( self, keys, fmt_o='' ):
    dict_d = _recursive_get( self.data_d, keys )
    scaling = float(dict_d['scaling'])
    if len( dict_d['values'] ) == 1:
        values_a = float( dict_d['values'][0] )
    else:
        values_a = np.array( [float(value) for value in dict_d['values']] )
    scaled_values_a = scaling * values_a
    if fmt_o:
        scaled_values_a = fmt_o.ascale( scaled_values_a )
    return scaled_values_a

def load_configs(cases):
    ncases = len(cases)
    pbar = tqdm(desc="Configs", total=ncases)
    cfgs = []
    for i in range(ncases):
        cfgs.append(read_config(cases[i]))
        pbar.update(1)
    pbar.close()
    return cfgs

def access_configs(cfgs,keys):
    values = []
    for i in range(len(cfgs)):
        val = descend_get(cfgs[i],keys)
        values.append(val)
    return values

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


 # check scaling
def _is_log(_arr):
    _arr = np.array(_arr)
    if np.any(_arr <= 0):
        return False
    return np.log10(np.amax(_arr)/np.amin(_arr)) > 2.0  # range is more than 2 order of magnitude

# Interpolate a variable onto a higher resolution grid (2 dimensions only)
def interp_2d(x_locs, y_locs, z_vals, npoints, method="linear", scaling=True):

    '''
    Depends on Scipy's RegularGridInterpolator class.

    Inputs
        x_locs  : Location of points on x-axis (1D)
        y_locs  : Location of points on y-axis (1D)
        z_vals  : Values of points (1D, flattened)
        method  : Interpolation method
        scaling : Use log-scaling if variable range is >2 OOM

    Outputs
        xo     : Original x-locations (1D grid)
        yo     : Original y-locations (1D grid)
        xi     : Interpolated x-locations (2D grid)
        yi     : Interpolated y-locations (2D grid)
        zi     : Interpolated z_vals (2D grid)
    '''

    # check dimensions
    if (len(x_locs) < 3) or (len(y_locs) < 3):
        raise Exception("Cannot interpolate grid with a resolution less than 3")

    zlog = _is_log(z_vals) and scaling
    if zlog:
        z_vals = np.log10(z_vals)

    xlog = _is_log(x_locs) and scaling
    if xlog:
        x_locs = np.log10(x_locs)
    xmin = np.amin(x_locs)
    xmax = np.amax(x_locs)

    ylog = _is_log(y_locs) and scaling
    if ylog:
        y_locs = np.log10(y_locs)
    ymin = np.amin(y_locs)
    ymax = np.amax(y_locs)

    # grid to interpolate at
    xi = np.linspace(xmin, xmax, npoints)
    yi = np.linspace(ymin, ymax, npoints)

    xxi = []
    yyi = []
    for x in xi:
        for y in yi:
            xxi.append(x)
            yyi.append(y)

    # do interpolation
    zzi = griddata((x_locs, y_locs), z_vals, (xxi,yyi), method=method)

    zzi = np.reshape(zzi, (npoints, npoints)).T

    if xlog:
        xi     = 10.0 ** xi
        x_locs = 10.0 ** x_locs
    if ylog:
        yi     = 10.0 ** yi
        y_locs = 10.0 ** y_locs
    if zlog:
        zzi = 10.0 ** zzi
    return x_locs,y_locs,xi,yi,zzi

def add_cbar(fig, sm, ticks=[], tick_format="%g", label="_label", width=0.03, squeeze=0.9):
    fig.subplots_adjust(right=0.89)
    cbar_ax = fig.add_axes([squeeze, 0.15, width, 0.7])
    if len(ticks) > 1:
        cbar = fig.colorbar(sm, cax=cbar_ax, values=ticks)
    else:
        cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(label)

    if len(ticks) > 1:
        cbar.set_ticks(ticks=ticks, labels=[tick_format%t for t in ticks])

def make_legend(ax, loc='best', lw=1.0, alpha=1.0, set_color='k', title=None):

    leg = ax.legend(loc=loc)

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys(),
                        loc=loc, framealpha=alpha, title=title)

    for hdl in leg.legend_handles:
            col = hdl.set_alpha(1.0)

    if set_color is not None:
        for hdl in leg.legend_handles:
            hdl.set_color(set_color)

    for line in leg.get_lines():
        line.set_linewidth(lw)

    return leg
