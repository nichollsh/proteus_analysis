#!/usr/bin/env python3
# Analyse the L 98-59 c PROTEUS grids.
from __future__ import annotations

import math
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from proteus.atmos_clim.common import find_latest_atmosphere_time, read_ncdf_profile
from proteus.utils.plot import get_colour

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Anchor paths to this script's directory so the script runs from any cwd.
# `nogit_analysis/data` is a symlink to the PROTEUS output tree.
SCRIPT_DIR = Path(__file__).resolve().parent
grid_name = 'l9859c_grid5'
GRID_DIR = SCRIPT_DIR / 'data' / grid_name
OUT_DIR = SCRIPT_DIR / 'output' / f'{grid_name}_analysis'

SPECIES = ['CS2', 'SO2', 'H2', 'H2S', 'CO2', 'CO', 'CH4', 'H2O', 'S2']
# Species also present in the equilibrium helpfile (CS2 is VULCAN-only).
HELPFILE_SPECIES = ['SO2', 'H2', 'H2S', 'CO2', 'CO', 'CH4', 'H2O', 'S2']

GRID_KEYS = ['H_budget', 'fO2_shift_IW', 'S_budget', 'C_budget']

R_EARTH = 6.371e6  # m
M_EARTH = 5.972e24  # kg
BAR_PER_PA = 1.0e-5
P_OBS_BAR = 0.02  # observation pressure level (config atmos_clim.p_obs)

# Observed planet (NASA Exoplanet Archive, L 98-59 c)
R_OBS_MEAS = 1.329
R_OBS_MEAS_ERR = 0.029
M_OBS_MEAS = 2.00
M_OBS_MEAS_ERR = 0.13

VMR_FLOOR = 1e-30  # replace exact zeros for log operations

fo2_lbl = r'$f$O$_2$ shift [$\Delta$ IW]'

# Wong colourblind-friendly palette
WONG = {
    'black': '#000000',
    'orange': '#E69F00',
    'skyblue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermilion': '#D55E00',
    'purple': '#CC79A7',
}
# Species colours follow the PROTEUS ecosystem style (proteus.utils.plot).
# get_colour returns the preset hue per gas, or generates one from composition
# for species without a preset (e.g. CS2).
SPECIES_COLOR = {s: get_colour(s) for s in SPECIES}

plt.rcParams.update(
    {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'figure.dpi': 160,
        'savefig.dpi': 160,
    }
)


# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------
def read_grid_params(case: Path) -> dict:
    """Return the resolved grid-axis values for a case, or NaNs if unreadable."""
    cfg = case / 'init_coupler.toml'
    out = {k: math.nan for k in GRID_KEYS} | {'mass_tot': math.nan}
    if not cfg.exists():
        return out
    with open(cfg, 'rb') as f:
        data = tomllib.load(f)
    planet = data.get('planet', {})
    elements = planet.get('elements', {})
    out['mass_tot'] = planet.get('mass_tot', math.nan)
    out['H_budget'] = elements.get('H_budget', math.nan)
    out['S_budget'] = elements.get('S_budget', math.nan)
    out['C_budget'] = elements.get('C_budget', math.nan)
    out['fO2_shift_IW'] = data.get('outgas', {}).get('fO2_shift_IW', math.nan)
    return out


def read_status(case: Path) -> str:
    f = case / 'status'
    if not f.exists():
        return 'missing'
    return f.read_text().strip().splitlines()[-1]


def read_helpfile_final(case: Path) -> dict | None:
    """Final-row bulk atmosphere from runtime_helpfile.csv."""
    f = case / 'runtime_helpfile.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f, sep='\t')
    if df.empty:
        return None
    last = df.iloc[-1]
    rec = {
        'time_yr': float(last['Time']),
        'P_surf_bar': float(last['P_surf']),
        'T_surf_K': float(last['T_surf']),
        'R_int_Rearth': float(last['R_int']) / R_EARTH,
        'R_obs_Rearth': float(last['R_obs']) / R_EARTH,
        'M_planet_Mearth': float(last['M_planet']) / M_EARTH,
        'mu_atm_kg_mol': float(last['atm_kg_per_mol']),
    }
    for s in HELPFILE_SPECIES:
        rec[f'eq_{s}_vmr'] = float(last[f'{s}_vmr'])
        rec[f'eq_{s}_bar'] = float(last[f'{s}_bar'])
    return rec


def read_vulcan_profile(case: Path) -> pd.DataFrame | None:
    """VULCAN post-processed profile; pressure in Pa, ordered top->surface."""
    f = case / 'offchem' / 'vulcan.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f, sep='\t')
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith('Unnamed')]]
    if df.empty or 'p' not in df.columns:
        return None
    return df


def read_agni_profile(case: Path) -> dict | None:
    """Final AGNI atmosphere snapshot: p [Pa], t [K], z [m above the interior
    surface], ordered top (TOA) -> surface.

    This is the bulk-equilibrium (radiative-convective solver) state, not the
    post-photochemistry one -- VMR profiles should come from the VULCAN
    offchem CSV (`read_vulcan_profile`), not from this file's own `x_gas`.
    Reuses the ecosystem's own NetCDF reader (`read_ncdf_profile`) rather than
    re-parsing the file, so units/ordering/z-definition stay consistent with
    the rest of PROTEUS. Returns None if no `*_atm.nc` snapshot is found.
    """
    t_final = find_latest_atmosphere_time(str(case))
    if t_final is None:
        return None
    nc_path = case / 'data' / f'{t_final:.0f}_atm.nc'
    out = read_ncdf_profile(str(nc_path), combine_edges=False)
    if out is None:
        return None
    order = np.argsort(out['p'])  # ascending pressure = TOA -> surface
    for key, arr in out.items():
        if isinstance(arr, np.ndarray) and arr.shape == out['p'].shape:
            out[key] = arr[order]
    return out


def interp_vmr_at_pressure(p_pa: np.ndarray, vmr: np.ndarray, p_target_pa: float) -> float:
    """Log-log interpolate VMR to p_target. Returns NaN if out of range."""
    order = np.argsort(p_pa)
    p_s = p_pa[order]
    v_s = np.clip(vmr[order], VMR_FLOOR, None)
    if p_target_pa < p_s[0] or p_target_pa > p_s[-1]:
        return math.nan
    logv = np.interp(np.log10(p_target_pa), np.log10(p_s), np.log10(v_s))
    return 10.0**logv


def interp_vmr_profile(p_pa: np.ndarray, vmr: np.ndarray, p_target_pa: np.ndarray) -> np.ndarray:
    """Log-log interpolate VMR(p) onto an array of target pressures.

    Vectorized counterpart of `interp_vmr_at_pressure`; NaN for targets
    outside the source pressure range.
    """
    order = np.argsort(p_pa)
    p_s = p_pa[order]
    v_s = np.clip(vmr[order], VMR_FLOOR, None)
    logv = np.interp(np.log10(p_target_pa), np.log10(p_s), np.log10(v_s))
    out = 10.0**logv
    out[(p_target_pa < p_s[0]) | (p_target_pa > p_s[-1])] = math.nan
    return out


def summarise_offchem(prof: pd.DataFrame) -> dict:
    """Surface / observation-level / TOA / column-max VMR per species."""
    p_pa = prof['p'].to_numpy(dtype=float)
    i_surf = int(np.argmax(p_pa))  # highest pressure = surface
    i_toa = int(np.argmin(p_pa))  # lowest pressure = top
    p_surf_pa = p_pa[i_surf]
    rec = {'oc_P_surf_bar': p_surf_pa * BAR_PER_PA}
    for s in SPECIES:
        if s not in prof.columns:
            rec |= {
                f'oc_{s}_surf_vmr': math.nan,
                f'oc_{s}_obs_vmr': math.nan,
                f'oc_{s}_toa_vmr': math.nan,
                f'oc_{s}_max_vmr': math.nan,
                f'oc_{s}_surf_bar': math.nan,
            }
            continue
        v = prof[s].to_numpy(dtype=float)
        surf = v[i_surf]
        rec[f'oc_{s}_surf_vmr'] = surf
        rec[f'oc_{s}_surf_bar'] = surf * p_surf_pa * BAR_PER_PA
        rec[f'oc_{s}_toa_vmr'] = v[i_toa]
        rec[f'oc_{s}_max_vmr'] = float(np.max(v))
        rec[f'oc_{s}_obs_vmr'] = interp_vmr_at_pressure(p_pa, v, P_OBS_BAR / BAR_PER_PA)
    return rec


# ----------------------------------------------------------------------------
# Build the summary table
# ----------------------------------------------------------------------------
def build_summary() -> pd.DataFrame:
    rows = []
    for case in sorted(GRID_DIR.glob('case_*')):
        rec = {'case': case.name, 'status': read_status(case)}
        rec |= read_grid_params(case)
        hf = read_helpfile_final(case)
        rec['has_helpfile'] = hf is not None
        if hf:
            rec |= hf
        prof = read_vulcan_profile(case)
        rec['has_offchem'] = prof is not None
        if prof is not None:
            rec['offchem_nlev'] = len(prof)
            rec |= summarise_offchem(prof)
        rows.append(rec)
    if not rows:
        raise SystemExit(
            f"No 'case_*' directories found under {GRID_DIR}\n"
            f'(resolved to {GRID_DIR.resolve()}). Check grid_name and that the '
            'data symlink points at the PROTEUS output tree.'
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
# Shared alpha convention: escaped runs are drawn faint everywhere so
# they still read as data points but as lower-confidence than retained runs.
STABLE_ALPHA, UNSTABLE_ALPHA = 0.85, 0.22


def _is_unstable(status: pd.Series) -> np.ndarray:
    """True for escaped runs (matched against `status` text)."""
    return (
        status.astype(str)
        .str.contains('escaped', case=False, regex=True)
        .to_numpy(bool)
    )

def _is_failed(status: pd.Series) -> np.ndarray:
    """True for errored or died runs (matched against `status` text)."""
    return (
        status.astype(str)
        .str.contains('error|died', case=False, regex=True)
        .to_numpy(bool)
    )


def _select_retained_or_unstable(
    df: pd.DataFrame, data_mask: pd.Series
) -> tuple[pd.DataFrame, np.ndarray]:
    """Subset to `data_mask` rows, keeping escaped cases even if P_surf <= 1 bar.

    Returns the subset frame plus an aligned unstable-status boolean array, so
    callers can draw retained atmospheres at full opacity and escaped
    ones faint rather than dropping the latter from the plot entirely.
    """
    base = df[data_mask].copy()
    unstable = _is_unstable(base['status'])
    keep = (base['P_surf_bar'] > 1.0) | unstable
    return base[keep].copy(), unstable[keep.to_numpy()]


def plot_eq_vs_offchem_photosphere(df: pd.DataFrame):
    """Bulk equilibrium VMR vs post-processed photosphere-level VMR.

    x: outgassed bulk (surface) mixing ratio from the helpfile.
    y: VULCAN VMR interpolated to the photosphere (p_obs = 0.02 bar), i.e. the
    abundance an observer sees. Departure from the 1:1 line is the combined
    effect of photochemistry and vertical structure at observable altitude.
    """
    sub = df[df['has_offchem'] & df['has_helpfile'] & (df['P_surf_bar'] > 1.0)].copy()
    lo, hi = 1e-12, 1.0
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for ax, s in zip(axes.ravel(), SPECIES):
        y = np.clip(sub[f'oc_{s}_obs_vmr'].to_numpy(float), VMR_FLOOR, None)
        eq_col = f'eq_{s}_vmr'
        has_eq = eq_col in sub.columns
        if has_eq:
            x = np.clip(sub[eq_col].to_numpy(float), VMR_FLOOR, None)
            ax.scatter(x, y, s=28, c=SPECIES_COLOR[s], edgecolor='k', linewidth=0.4, zorder=3)
        else:
            for yy in y:
                ax.axhline(yy, xmin=lo, xmax=hi, color=SPECIES_COLOR[s], linewidth=1.6, zorder=3)
        ax.plot([lo, hi], [lo, hi], '--', color='0.4', linewidth=0.8, zorder=1)
        title = s if has_eq else f'{s} (kinetics, VULCAN-only)'
        ax.set(
            xscale='log',
            yscale='log',
            xlim=(lo, hi),
            ylim=(lo, hi),
            title=title,
            xlabel='equilibrium bulk VMR',
            ylabel='offchem photosphere VMR',
        )
    for ax in axes.ravel()[len(SPECIES) :]:
        ax.set_visible(False)
    fig.suptitle(
        'Equilibrium outgassing (bulk) vs VULCAN at the photosphere (p_obs = 0.02 bar)\n'
        '(1:1 line = observable abundance matches the outgassed prediction)',
        fontsize=11,
    )
    fig.savefig(OUT_DIR / 'fig_eq_vs_offchem_photosphere.png')
    plt.close(fig)


def plot_profiles(df: pd.DataFrame, ncols: int = 4):
    """VMR(p) profiles for the 7 species, cases inside the measured radius band.

    Selection: retained atmospheres (P_surf > 1 bar) whose modelled transit
    radius R_obs lies within the measured L 98-59 c 1-sigma band
    (R_OBS_MEAS +/- R_OBS_MEAS_ERR). The observation pressure level (p_obs)
    is marked so the observable part of each profile is clear.
    """
    lo, hi = R_OBS_MEAS - R_OBS_MEAS_ERR, R_OBS_MEAS + R_OBS_MEAS_ERR
    sub = df[
        df['has_offchem'] & (df['P_surf_bar'] > 1.0) & df['R_obs_Rearth'].between(lo, hi)
    ].copy()
    sub = sub.sort_values('R_obs_Rearth')
    picks = sub['case'].tolist()
    if not picks:
        return
    n = len(picks)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, cname in zip(axes, picks):
        prof = read_vulcan_profile(GRID_DIR / cname)
        p_bar = prof['p'].to_numpy(float) * BAR_PER_PA
        for s in SPECIES:
            if s not in prof.columns:
                continue
            v = np.clip(prof[s].to_numpy(float), VMR_FLOOR, None)
            ax.plot(v, p_bar, color=SPECIES_COLOR[s], label=s, linewidth=1.6)
        ax.axhline(P_OBS_BAR, color='0.4', linestyle=':', linewidth=1.0, zorder=1)
        row = df[df['case'] == cname].iloc[0]
        # Pressure decreases upward: surface (high P) at the bottom, TOA at top.
        ax.set(
            xscale='log',
            yscale='log',
            xlim=(1e-10, 2.0),
            ylim=(float(np.nanmax(p_bar)) * 1.5, float(np.nanmin(p_bar)) * 0.6),
            title=(
                f'{cname}  R={row["R_obs_Rearth"]:.3f} R$_\\oplus$\n'
                f'P$_s$={row["P_surf_bar"]:.0f} bar, '
                f'T$_s$={row["T_surf_K"]:.0f} K\n'
                f'H={row["H_budget"]:.0f} fO2={row["fO2_shift_IW"]:+.0f} '
                f'S={row["S_budget"]:.0f} C={row["C_budget"]:.0f}'
            ),
        )
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[:n]:
        ax.set_ylabel('Pressure [bar]')
        ax.set_xlabel('VMR')
    axes[0].legend(fontsize=8, loc='lower left', framealpha=0.9)
    fig.suptitle(
        'VULCAN post-processed profiles: cases matching measured radius '
        f'{R_OBS_MEAS}$\\pm${R_OBS_MEAS_ERR} R$_\\oplus$ '
        '(dotted line = observation level p_obs)',
        fontsize=11,
    )
    fig.tight_layout(h_pad=0.3)
    fig.savefig(OUT_DIR / 'fig_offchem_profiles.png', bbox_inches='tight')
    plt.close(fig)


def _radius_matched_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Retained cases (P_surf > 1 bar) whose modelled R_obs matches the
    measured L 98-59 c 1-sigma radius band, sorted by R_obs.
    """
    lo, hi = R_OBS_MEAS - R_OBS_MEAS_ERR, R_OBS_MEAS + R_OBS_MEAS_ERR
    mask = df['has_helpfile'] & (df['P_surf_bar'] > 1.0) & df['R_obs_Rearth'].between(lo, hi)
    return df[mask].sort_values('R_obs_Rearth').copy()


def plot_atm_tp_profiles(df: pd.DataFrame):
    """AGNI radiative-convective T(p) profiles for radius-matching cases,
    combined onto a single panel and coloured by fO2 shift.

    Selection: same radius-matching criterion as the VULCAN profile plot
    (retained atmosphere, P_surf > 1 bar, R_obs within the measured L 98-59 c
    1-sigma band), but drawn from the final AGNI `*_atm.nc` snapshot rather
    than the post-processed VULCAN profile, so this shows the physical
    (bulk-equilibrium) thermal structure independent of photochemistry.
    """
    sub = _radius_matched_cases(df)
    if sub.empty:
        return
    f_vals = sorted(sub['fO2_shift_IW'].dropna().unique())
    palette = [WONG['blue'], WONG['green'], WONG['vermilion'], WONG['purple'], WONG['orange']]
    f_colors = {v: palette[i % len(palette)] for i, v in enumerate(f_vals)}

    fig, ax = plt.subplots(figsize=(7, 6.5), constrained_layout=True)
    p_bar_all = []
    n_plotted = 0
    for fv in f_vals:
        label = f'fO2={fv:+.0f}'
        for cname in sub.loc[sub['fO2_shift_IW'] == fv, 'case']:
            prof = read_agni_profile(GRID_DIR / cname)
            if prof is None:
                continue
            p_bar = prof['p'] * BAR_PER_PA
            ax.plot(
                prof['t'], p_bar, color=f_colors[fv], linewidth=1.4, alpha=0.85,
                label=label, zorder=3,
            )
            label = None  # only the first line per fO2 value gets a legend entry
            p_bar_all.append(p_bar)
            n_plotted += 1
    if n_plotted == 0:
        plt.close(fig)
        return
    all_p = np.concatenate(p_bar_all)
    ax.axhline(P_OBS_BAR, color='0.4', linestyle=':', linewidth=1.0, zorder=1)
    ax.set(
        yscale='log',
        ylim=(float(np.nanmax(all_p)) * 1.5, float(np.nanmin(all_p)) * 0.6),
        xlabel='Temperature [K]',
        ylabel='Pressure [bar]',
    )
    ax.legend(fontsize=9, title=fo2_lbl, loc='best')
    fig.suptitle(
        'AGNI atmosphere T(p) profiles: cases matching measured radius '
        f'{R_OBS_MEAS}$\\pm${R_OBS_MEAS_ERR} R$_\\oplus$\n'
        '(colour = fO2 shift; dotted line = observation level p_obs)',
        fontsize=11,
    )
    fig.savefig(OUT_DIR / 'fig_atm_tp_profiles.png')
    plt.close(fig)


def write_atm_profile_csvs(df: pd.DataFrame):
    """Write per-case T, P, Z, VMR atmosphere profile CSVs for radius-matching
    cases (same selection as `plot_atm_tp_profiles`, further restricted to
    cases that have a VULCAN offchem profile).

    One file per case at OUT_DIR/profiles/<case>_atm_profile.csv, columns
    P_bar, T_K, Z_m (height above the interior surface, from the final AGNI
    snapshot) and one `<GAS>_vmr` column per SPECIES. VMR comes from the
    post-processed VULCAN profile (offchem/vulcan.csv), log-log interpolated
    onto the AGNI pressure grid -- not from the NetCDF's own bulk-equilibrium
    `x_gas`, since VULCAN carries the post-photochemistry abundance. Rows are
    ordered top (TOA) -> surface.
    """
    sub = _radius_matched_cases(df)
    sub = sub[sub['has_offchem']]
    if sub.empty:
        return
    prof_dir = OUT_DIR / 'profiles'
    prof_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for cname in sub['case']:
        case = GRID_DIR / cname
        atm = read_agni_profile(case)
        vulcan = read_vulcan_profile(case)
        if atm is None or vulcan is None:
            continue
        cols = {'P_bar': atm['p'] * BAR_PER_PA, 'T_K': atm['t'], 'Z_m': atm['z']}
        vulcan_p = vulcan['p'].to_numpy(float)
        for sp in SPECIES:
            if sp not in vulcan.columns:
                continue
            cols[f'{sp}_vmr'] = interp_vmr_profile(
                vulcan_p, vulcan[sp].to_numpy(float), atm['p']
            )
        pd.DataFrame(cols).to_csv(prof_dir / f'{cname}_atm_profile.csv', index=False)
        n_written += 1
    print(f'Wrote {n_written} atmosphere profile CSV(s) to {prof_dir}')


# Shared log10(VMR) colour-scale window so all panels/species use one colourbar.
VMR_COLOR_LO, VMR_COLOR_HI = 1e-10, 1.0
VMR_CMAP = 'gnuplot2'


def plot_grid_dependence(df: pd.DataFrame, where='obs'):
    """Offchem photosphere VMR across the redox / sulfur grid, for retained cases.

    x: fO2 shift [log10 dIW]; y: S/H mass ratio; marker colour (via a shared
    colourbar) encodes the post-processed photosphere VMR. One panel per
    species. Cases that share an (fO2, S/H) node (differing C or H budget) are
    xy-jittered so they do not fully overplot. Escaped cases are kept
    (rather than dropped by the P_surf > 1 bar retained-atmosphere cut) and
    drawn at low opacity so they read as lower-confidence.
    """
    sub, unstable = _select_retained_or_unstable(df, df['has_offchem'])
    if sub.empty:
        return
    # Jitter degenerate (fO2, S/H) nodes apart along x, keyed by C then H budget.
    c_vals = sorted(sub['C_budget'].dropna().unique())
    h_vals = sorted(sub['H_budget'].dropna().unique())

    def _jit(row):
        ci = c_vals.index(row['C_budget']) if row['C_budget'] in c_vals else 0
        hi = h_vals.index(row['H_budget']) if row['H_budget'] in h_vals else 0
        span = max(len(c_vals) * max(len(h_vals), 1), 1)
        return (ci * max(len(h_vals), 1) + hi - (span - 1) / 2) * 0.045

    # jitter
    x_jit = np.copy(sub.apply(_jit, axis=1).to_numpy(float))
    y_jit = np.copy(sub.apply(_jit, axis=1).to_numpy(float))

    # shuffle the jitter arrays
    np.random.seed(42)
    np.random.shuffle(x_jit)
    np.random.shuffle(y_jit)
    
    x = sub['fO2_shift_IW'].to_numpy(float) + x_jit
    y = sub['S_budget'].to_numpy(float) + y_jit
    norm = LogNorm(vmin=VMR_COLOR_LO, vmax=VMR_COLOR_HI)

    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for ax, s in zip(axes.ravel(), SPECIES):
        vmr = np.clip(sub[f'oc_{s}_{where}_vmr'].to_numpy(float), VMR_FLOOR, None)
        for msk, alpha in ((~unstable, STABLE_ALPHA), (unstable, UNSTABLE_ALPHA)):
            if not msk.any():
                continue
            ax.scatter(
                x[msk],
                y[msk],
                s=44,
                c=vmr[msk],
                cmap=VMR_CMAP,
                norm=norm,
                edgecolor='k',
                linewidth=0.3,
                alpha=alpha,
                zorder=3,
            )
        ax.set(
            title=s,
            xlabel=fo2_lbl,
            ylabel='S/H mass ratio',
        )
    for ax in axes.ravel()[len(SPECIES) :]:
        ax.set_visible(False)
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=VMR_CMAP),
        ax=axes.ravel().tolist(),
        label='photosphere VMR',
        shrink=0.85,
    )
    fig.suptitle(
        f'Gas VMR at {where}-layer, versus grid axes\n'
        f'(colour = VMR; faint = escaped runs)',
        fontsize=11,
    )
    fig.savefig(OUT_DIR / f'fig_grid_dependence_{where}.png')
    plt.close(fig)


# Photosphere abundance ratios of interest and the grid axes to correlate against.
RATIOS = [('CS2', 'SO2'), ('SO2', 'H2S'), ('H2S', 'CS2')]
RATIO_COLOR = {
    ('CS2', 'SO2'): WONG['vermilion'],
    ('SO2', 'H2S'): WONG['blue'],
    ('H2S', 'CS2'): WONG['green'],
}
PARAM_LABEL = {
    'H_budget': 'H budget',
    'fO2_shift_IW': fo2_lbl,
    'S_budget': 'S/H mass ratio',
    'C_budget': 'C/H mass ratio',
}


def _symmetric_ratio_ylim(ratio: np.ndarray) -> tuple[float, float]:
    """Log-symmetric y-limits about ratio=1 that fit this panel's data.

    Returns (10**-m, 10**m) where m is the largest |log10(ratio)| among the
    finite, positive values passed in (plus a small margin), so equal
    enhancement and depletion read at equal visual distance from the ratio=1
    line. Bounded to at least one decade each side; there is no upper cap, so
    each panel's view is set purely by that panel's own data range rather
    than a single global ceiling shared across all panels. The underlying
    data is never clipped, only the axis view.
    """
    finite = ratio[np.isfinite(ratio) & (ratio > 0)]
    if finite.size == 0:
        return (10.0**-1.0, 10.0**1.0)
    vmrmax = float(np.max(np.abs(np.log10(finite))))
    vmrmax = min(vmrmax, 10.0)  # cap to avoid extreme outliers dominating the view
    m = max(vmrmax + 0.3, 1.0)  # margin, >=1 decade
    return (10.0**-m, 10.0**m)


def _use_logx(x: np.ndarray, param: str) -> bool:
    """H budget is always log-scaled; other positive axes only if >1 decade."""
    pos = x[np.isfinite(x) & (x > 0)]
    return param == 'H_budget' or (pos.size > 0 and pos.max() / pos.min() > 20)


def plot_ratio_correlations(df: pd.DataFrame):
    """Photosphere abundance ratios and atmosphere MMW vs each grid parameter.

    Rows: the three ratios (CS2/SO2, SO2/H2S, H2S/CS2, all at p_obs = 0.02 bar),
    then a final row of bulk atmospheric mean molecular weight (from the
    helpfile). Columns: the four grid axes. Ratio rows are log-scaled with
    y-limits computed per panel (not shared across the row, and with no global
    cap) from whichever cases have a finite value for that panel's grid
    parameter, kept symmetric about ratio=1 (dashed reference line); the
    plotted ratio values themselves are never clipped. Cases whose modelled
    transit radius falls in the measured L 98-59 c 1-sigma band are drawn as
    stars; all others as circles.
    """
    sub, unstable = _select_retained_or_unstable(df, df['has_offchem'])
    if sub.empty:
        return
    lo_r, hi_r = R_OBS_MEAS - R_OBS_MEAS_ERR, R_OBS_MEAS + R_OBS_MEAS_ERR
    match = (sub['has_helpfile'] & sub['R_obs_Rearth'].between(lo_r, hi_r)).to_numpy(bool)

    def _scatter(ax, x, y, color):
        # Split by radius match (circle vs star) and by run stability (alpha).
        shapes = ((~match, 'o', 0.3, 3), (match, '*', 0.6, 4))
        for msk, marker, mk_lw, mk_z in shapes:
            for stab_msk, alpha in ((~unstable, STABLE_ALPHA), (unstable, UNSTABLE_ALPHA)):
                m = msk & stab_msk & np.isfinite(x) & np.isfinite(y)
                if not m.any():
                    continue
                ax.scatter(
                    x[m],
                    y[m],
                    marker=marker,
                    s=(150 if marker == '*' else 34),
                    c=color,
                    edgecolor='k',
                    linewidth=mk_lw,
                    zorder=mk_z,
                    alpha=alpha,
                )

    nrows, ncols = len(RATIOS) + 1, len(GRID_KEYS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), constrained_layout=True
    )
    for i, (num, den) in enumerate(RATIOS):
        n = np.clip(sub[f'oc_{num}_obs_vmr'].to_numpy(float), VMR_FLOOR, None)
        d = np.clip(sub[f'oc_{den}_obs_vmr'].to_numpy(float), VMR_FLOOR, None)
        ratio = n / d
        color = RATIO_COLOR[(num, den)]
        for j, param in enumerate(GRID_KEYS):
            ax = axes[i, j]
            x = sub[param].to_numpy(float)
            # Per-panel symmetric ylim: only cases with a finite value for
            # *this* grid parameter set the view, since that can differ by column.
            ylim = _symmetric_ratio_ylim(ratio[np.isfinite(x)])
            ax.axhline(1.0, color='0.5', linestyle='--', linewidth=0.8, zorder=1)
            _scatter(ax, x, ratio, color)
            ax.set(yscale='log', ylim=ylim)
            if _use_logx(x, param):
                ax.set_xscale('log')
            # if j == 0:
            ax.set_ylabel(f'{num}/{den} VMR')

    # Final row: bulk atmospheric mean molecular weight (helpfile), g/mol.
    mmw = sub['mu_atm_kg_mol'].to_numpy(float) * 1.0e3
    i = len(RATIOS)
    for j, param in enumerate(GRID_KEYS):
        ax = axes[i, j]
        x = sub[param].to_numpy(float)
        _scatter(ax, x, mmw, WONG['black'])
        ax.set_ylim(0, 32)
        if _use_logx(x, param):
            ax.set_xscale('log')
        # if j == 0:
        ax.set_ylabel('atm MMW [g/mol]')
        ax.set_xlabel(PARAM_LABEL[param])

    fig.suptitle(
        'Photosphere abundance ratios and atmosphere MMW (p_obs = 0.02 bar) '
        'vs grid parameters\n(stars = radius-matched; faint = escaped '
        'runs; dashed line = ratio 1)',
        fontsize=11,
    )
    fig.savefig(OUT_DIR / 'fig_ratio_correlations.png')
    plt.close(fig)


def plot_cs2(df: pd.DataFrame):
    """CS2 is VULCAN-only: photosphere-level VMR across the grid.

    Escaped cases are kept (rather than dropped by the retained-
    atmosphere cut) and drawn at low opacity so they read as lower-confidence;
    they are excluded from the legend to avoid duplicate entries.
    """
    sub, unstable = _select_retained_or_unstable(df, df['has_offchem'])
    if sub.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    c_vals = sorted(sub['C_budget'].dropna().unique())
    c_colors = {v: c for v, c in zip(c_vals, [WONG['blue'], WONG['vermilion']])}
    for cv in c_vals:
        gm = (sub['C_budget'] == cv).to_numpy(bool)
        y = np.clip(sub['oc_CS2_obs_vmr'].to_numpy(float), VMR_FLOOR, None)
        x = sub['fO2_shift_IW'].to_numpy(float)
        for stab_msk, alpha, label in (
            (~unstable, STABLE_ALPHA, f'C/H={cv:.0f}'),
            (unstable, UNSTABLE_ALPHA, None),
        ):
            m = gm & stab_msk
            if not m.any():
                continue
            a1.scatter(
                x[m],
                y[m],
                s=34,
                c=c_colors[cv],
                edgecolor='k',
                linewidth=0.3,
                alpha=alpha,
                label=label,
            )
    a1.set(
        yscale='log',
        xlabel=fo2_lbl,
        ylabel='CS2 photosphere VMR',
        title='CS2 photosphere abundance vs redox',
        xticks=[-3, -2, -1],
    )
    a1.legend(fontsize=9)
    # CS2 photosphere vs S_budget, coloured by fO2
    f_vals = sorted(sub['fO2_shift_IW'].dropna().unique())
    f_colors = {v: c for v, c in zip(f_vals, [WONG['blue'], WONG['green'], WONG['vermilion']])}
    for fv in f_vals:
        gm = (sub['fO2_shift_IW'] == fv).to_numpy(bool)
        y = np.clip(sub['oc_CS2_obs_vmr'].to_numpy(float), VMR_FLOOR, None)
        x = sub['S_budget'].to_numpy(float)
        for stab_msk, alpha, label in (
            (~unstable, STABLE_ALPHA, f'fO2={fv:+.0f}'),
            (unstable, UNSTABLE_ALPHA, None),
        ):
            m = gm & stab_msk
            if not m.any():
                continue
            a2.scatter(
                x[m],
                y[m],
                s=34,
                c=f_colors[fv],
                edgecolor='k',
                linewidth=0.3,
                alpha=alpha,
                label=label,
            )
    a2.set(
        yscale='log',
        xlabel='S/H mass ratio',
        ylabel='CS2 photosphere VMR',
        title='CS2 photosphere abundance vs sulfur budget',
    )
    a2.legend(fontsize=9)
    fig.suptitle(
        'CS2 at the photosphere (p_obs = 0.02 bar; produced only in VULCAN)\n'
        '(faint = escaped runs)',
        fontsize=11,
    )
    fig.savefig(OUT_DIR / 'fig_cs2.png')
    plt.close(fig)


# Magma-ocean threshold: T_surf above the mantle solidus implies a molten
# (at least partially) surface rather than a fully solidified crust.
SOLIDUS_T_K = 1400.0


def plot_radius(df: pd.DataFrame):
    """Modelled transit radius vs measured L 98-59 c radius.

    Escaped cases are kept (rather than dropped by the retained-
    atmosphere cut) and drawn at low opacity so they read as lower-confidence.
    Cases with T_surf above the solidus (SOLIDUS_T_K, i.e. a magma-ocean
    surface) are drawn as squares; all other cases as circles.
    """
    sub, unstable = _select_retained_or_unstable(df, df['has_helpfile'])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    x = np.arange(len(sub))
    r = sub['R_obs_Rearth'].to_numpy(float)
    s = np.log10(sub['H_budget'].to_numpy(float)) * 5 + 20.0  # marker size keyed to H budget
    magma = sub['T_surf_K'].to_numpy(float) > SOLIDUS_T_K
    shape_groups = (
        (~magma, 'o', 'likely solidified'),
        (magma, 's', f'likely has magma ocean (T$_s$>{SOLIDUS_T_K:.0f} K)'),
    )
    for shape_msk, marker, base_label in shape_groups:
        for stab_msk, alpha, use_label in (
            (~unstable, STABLE_ALPHA, True),
            (unstable, UNSTABLE_ALPHA, False),
        ):
            m = shape_msk & stab_msk
            if not m.any():
                continue
            ax.scatter(
                x[m],
                r[m],
                s=s[m],
                marker=marker,
                c=WONG['blue'],
                edgecolor='k',
                linewidth=0.3,
                zorder=3,
                alpha=alpha,
                label=(base_label if use_label else None),
            )
    ax.axhspan(
        R_OBS_MEAS - R_OBS_MEAS_ERR,
        R_OBS_MEAS + R_OBS_MEAS_ERR,
        color=WONG['orange'],
        alpha=0.25,
        zorder=1,
        label=f'measured {R_OBS_MEAS}$\\pm${R_OBS_MEAS_ERR} R$_\\oplus$',
    )
    ax.axhline(R_OBS_MEAS, color=WONG['orange'], linewidth=1.0, zorder=2)
    ax.set(
        ylabel='planet radius [R$_\\oplus$]',
        title='Modelled transit radius vs measured L 98-59 c radius\n'
        f'(faint = escaped runs; squares = magma ocean, T$_s$>{SOLIDUS_T_K:.0f} K)',
    )
    ax.set_xticks(x)
    labels = [
        f'{c.replace("case_0000", "")}, IW={f:+.0f}, H={h:.1e}, S/H={sh:.0f}'
        for c, f, h, sh in zip(sub['case'], sub['fO2_shift_IW'], sub['H_budget'], sub['S_budget'])
    ]
    ax.set_xticklabels(labels, fontsize=7, rotation=90, ha='right', rotation_mode='anchor')
    ax.legend(fontsize=9)
    fig.savefig(OUT_DIR / 'fig_radius.png')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def print_report(df: pd.DataFrame):
    n = len(df)
    n_off = int(df['has_offchem'].sum())
    n_retained = int((df['has_offchem'] & (df['P_surf_bar'] > 1.0)).sum())
    print('=' * 78)
    print(f'L 98-59 c grid  ({GRID_DIR})')
    print('=' * 78)
    print(f'cases total ............ {n}')
    print(f'with offchem/vulcan.csv  {n_off}')
    print(f'  of which retained P_s>1 bar ... {n_retained}')
    print(f'status breakdown:\n{df["status"].value_counts().to_string()}')
    print('-' * 78)

    ret = df[df['has_offchem'] & (df['P_surf_bar'] > 1.0)].copy()
    if not ret.empty:
        print('Retained-atmosphere cases: photosphere VMR ranges (VULCAN, p_obs = 0.02 bar)')
        for s in SPECIES:
            col = f'oc_{s}_obs_vmr'
            v = ret[col].replace(0, np.nan).dropna()
            if v.empty:
                print(f'  {s:4s}: all zero / absent')
                continue
            print(f'  {s:4s}: min={v.min():.2e}  median={v.median():.2e}  max={v.max():.2e}')
        print('-' * 78)
        # equilibrium (bulk) vs offchem photosphere, median over retained cases
        print('Median VMR: equilibrium bulk (eq) vs post-processed photosphere (oc)')
        for s in HELPFILE_SPECIES:
            e = ret[f'eq_{s}_vmr'].replace(0, np.nan).dropna()
            o = ret[f'oc_{s}_obs_vmr'].replace(0, np.nan).dropna()
            em = e.median() if not e.empty else float('nan')
            om = o.median() if not o.empty else float('nan')
            print(f'  {s:4s}: eq={em:.2e}   oc={om:.2e}')
        print('-' * 78)
        print('CS2 (VULCAN-only) photosphere VMR by case:')
        for _, row in ret.sort_values('oc_CS2_obs_vmr', ascending=False).iterrows():
            print(
                f'  {row["case"]}  photosphere={row["oc_CS2_obs_vmr"]:.2e}  '
                f'(surface={row["oc_CS2_surf_vmr"]:.2e}) '
                f'(fO2={row["fO2_shift_IW"]:+.0f} S={row["S_budget"]:.0f} '
                f'C={row["C_budget"]:.0f} H={row["H_budget"]:.0f})'
            )
        print('-' * 78)
        r = ret['R_obs_Rearth'].dropna()
        print(
            f'Modelled transit radius: min={r.min():.3f} median={r.median():.3f} '
            f'max={r.max():.3f} R_earth'
        )
        print(
            f'Measured L 98-59 c: R={R_OBS_MEAS}+/-{R_OBS_MEAS_ERR}, '
            f'M={M_OBS_MEAS}+/-{M_OBS_MEAS_ERR} M_earth (grid mass_tot=2.0)'
        )
    print('=' * 78)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_summary()
    df.to_csv(OUT_DIR / 'summary_grid.csv', index=False)
    print_report(df)
    plot_eq_vs_offchem_photosphere(df)
    plot_profiles(df)
    plot_atm_tp_profiles(df)
    write_atm_profile_csvs(df)
    plot_grid_dependence(df, where='obs')
    plot_grid_dependence(df, where='surf')
    plot_ratio_correlations(df)
    plot_cs2(df)
    plot_radius(df)
    print(f'\nWrote summary_grid.csv and figures to:\n  {OUT_DIR}')


if __name__ == '__main__':
    main()
