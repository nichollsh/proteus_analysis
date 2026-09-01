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
from proteus.utils.helper import resolve_fwl_data_dir
from proteus.utils.plot import get_colour

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Anchor paths to this script's directory so the script runs from any cwd.
# `nogit_analysis/data` is a symlink to the PROTEUS output tree.
SCRIPT_DIR = Path(__file__).resolve().parent
grid_name = "l9859d_grid4"
GRID_DIR = SCRIPT_DIR / "data" / grid_name
OUT_DIR = SCRIPT_DIR / "output" / f"{grid_name}_analysis"

SPECIES = ["CS2", "SO2", "H2", "H2S", "CO2", "CO", "CH4", "H2O", "S2"]
# Species also present in the equilibrium helpfile (CS2 is VULCAN-only).
HELPFILE_SPECIES = ["SO2", "H2", "H2S", "CO2", "CO", "CH4", "H2O", "S2"]

# Constants
R_EARTH = 6.371e6  # m
M_EARTH = 5.972e24  # kg
BAR_PER_PA = 1.0e-5
P_OBS_BAR = 0.02  # observation pressure level (config atmos_clim.p_obs)

# Observed planet (NASA Exoplanet Archive, L 98-59 c)
# GRID_KEYS = ['H_budget', 'fO2_shift_IW', 'S_budget', 'C_budget']
# R_OBS_MEAS = 1.329
# R_OBS_MEAS_ERR = 0.029
# M_OBS_MEAS = 2.00
# M_OBS_MEAS_ERR = 0.13

# Observed planet (L 98-59 d)
GRID_KEYS = ["H_budget", "fO2_shift_IW", "core_frac", "C_budget"]
R_OBS_MEAS = 1.627
R_OBS_MEAS_ERR = 0.041
M_OBS_MEAS = 1.64
M_OBS_MEAS_ERR = 0.07
# Observed L 98-59 d temperature, as supplied by the user; asymmetric
# 1-sigma error bar (+T_OBS_MEAS_ERR_HI, -T_OBS_MEAS_ERR_LO).
T_OBS_MEAS = 571.0  # K
T_OBS_MEAS_ERR_HI = 133.0  # K
T_OBS_MEAS_ERR_LO = 143.0  # K

VMR_FLOOR = 1e-30  # replace exact zeros for log operations

# H2/He + metal-admixture mean-molecular-weight reference model, used by
# `plot_density_vs_mmw`. Molar masses in g/mol.
M_H2_GMOL = 2.016
M_HE_GMOL = 4.003
METAL_M_GMOL = {"CO2": 44.01, "H2O": 18.015, "CO": 28.01, "CH4": 16.04, "O2": 32.00}
# Jupiter atmospheric He mass fraction from the Galileo probe mass
# spectrometer (von Zahn et al. 1998, Science 272, 846): He/H2 mole ratio
# 0.157 -> He mass fraction Y_He = 0.234. The non-metal remainder of every
# admixture below is split H2/He in this ratio.
Y_HE_JUPITER = 0.234

fo2_lbl = r"$f$O$_2$ shift [$\Delta$ IW]"

# Wong colourblind-friendly palette
WONG = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
}
# Species colours follow the PROTEUS ecosystem style (proteus.utils.plot).
# get_colour returns the preset hue per gas, or generates one from composition
# for species without a preset (e.g. CS2).
SPECIES_COLOR = {s: get_colour(s) for s in SPECIES}

# fO2 shift is coloured with the 'cool' colormap everywhere it gets a
# discrete per-value colour (as opposed to a continuous colourbar).
FO2_CMAP = "autumn"


def fo2_colors(f_vals: list[float]) -> dict[float, tuple]:
    """Map sorted, unique fO2 shift values to evenly spaced `FO2_CMAP` colours."""
    cmap = plt.get_cmap(FO2_CMAP)
    n = len(f_vals)
    if n == 0:
        return {}
    if n == 1:
        return {f_vals[0]: cmap(0.5)}
    return {v: cmap(i / (n - 1)) for i, v in enumerate(f_vals)}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.dpi": 160,
        "savefig.dpi": 160,
    }
)


# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------
def read_grid_params(case: Path) -> dict:
    """Return the resolved grid-axis values for a case, or NaNs if unreadable."""
    cfg = case / "init_coupler.toml"
    out = {k: math.nan for k in GRID_KEYS} | {"mass_tot": math.nan}
    if not cfg.exists():
        return out
    with open(cfg, "rb") as f:
        data = tomllib.load(f)
    planet = data.get("planet", {})
    elements = planet.get("elements", {})
    out["mass_tot"] = planet.get("mass_tot", math.nan)
    out["H_budget"] = elements.get("H_budget", math.nan)
    out["S_budget"] = elements.get("S_budget", math.nan)
    out["C_budget"] = elements.get("C_budget", math.nan)
    out["fO2_shift_IW"] = data.get("outgas", {}).get("fO2_shift_IW", math.nan)
    return out


def read_status(case: Path) -> str:
    f = case / "status"
    if not f.exists():
        return "missing"
    return f.read_text().strip().splitlines()[-1]


def read_helpfile_final(case: Path) -> dict | None:
    """Final-row bulk atmosphere from runtime_helpfile.csv."""
    f = case / "runtime_helpfile.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, sep="\t")
    if df.empty:
        return None
    last = df.iloc[-1]
    rec = {
        "time_yr": float(last["Time"]),
        "P_surf_bar": float(last["P_surf"]),
        "T_surf_K": float(last["T_surf"]),
        "R_int_Rearth": float(last["R_int"]) / R_EARTH,
        "R_obs_Rearth": float(last["R_obs"]) / R_EARTH,
        "M_planet_Mearth": float(last["M_planet"]) / M_EARTH,
        "mu_atm_kg_mol": float(last["atm_kg_per_mol"]),
    }
    for s in HELPFILE_SPECIES:
        rec[f"eq_{s}_vmr"] = float(last[f"{s}_vmr"])
        rec[f"eq_{s}_bar"] = float(last[f"{s}_bar"])
    return rec


def read_helpfile_timeseries(case: Path, cols: list[str]) -> pd.DataFrame | None:
    """Full time series of the requested `runtime_helpfile.csv` columns."""
    f = case / "runtime_helpfile.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, sep="\t")
    if df.empty:
        return None
    return df[["Time"] + cols]


def read_vulcan_profile(case: Path) -> pd.DataFrame | None:
    """VULCAN post-processed profile; pressure in Pa, ordered top->surface."""
    f = case / "offchem" / "vulcan.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    if df.empty or "p" not in df.columns:
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
    nc_path = case / "data" / f"{t_final:.0f}_atm.nc"
    out = read_ncdf_profile(str(nc_path), combine_edges=False)
    if out is None:
        return None
    order = np.argsort(out["p"])  # ascending pressure = TOA -> surface
    for key, arr in out.items():
        if isinstance(arr, np.ndarray) and arr.shape == out["p"].shape:
            out[key] = arr[order]
    return out


def interp_vmr_at_pressure(
    p_pa: np.ndarray, vmr: np.ndarray, p_target_pa: float
) -> float:
    """Log-log interpolate VMR to p_target. Returns NaN if out of range."""
    order = np.argsort(p_pa)
    p_s = p_pa[order]
    v_s = np.clip(vmr[order], VMR_FLOOR, None)
    if p_target_pa < p_s[0] or p_target_pa > p_s[-1]:
        return math.nan
    logv = np.interp(np.log10(p_target_pa), np.log10(p_s), np.log10(v_s))
    return 10.0**logv


def interp_vmr_profile(
    p_pa: np.ndarray, vmr: np.ndarray, p_target_pa: np.ndarray
) -> np.ndarray:
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


def mmw_h_he_metal(
    f_metal: np.ndarray | float, m_metal_gmol: float
) -> np.ndarray | float:
    """Mean molecular weight [g/mol] of a Jupiter-ratio H2/He gas diluted by
    a metal species at mass fraction `f_metal` (in [0, 1)).

    1/mu = sum_i(w_i / M_i); the (1 - f_metal) non-metal remainder is split
    H2/He in the Jupiter mass ratio (Y_HE_JUPITER). This is a composition-only
    mixing-rule estimate -- it says nothing about the resulting bulk density,
    which also depends on pressure, temperature and self-gravity.
    """
    w_rest = 1.0 - f_metal
    w_h2 = w_rest * (1.0 - Y_HE_JUPITER)
    w_he = w_rest * Y_HE_JUPITER
    inv_mu = f_metal / m_metal_gmol + w_h2 / M_H2_GMOL + w_he / M_HE_GMOL
    return 1.0 / inv_mu


def summarise_offchem(prof: pd.DataFrame) -> dict:
    """Surface / observation-level / TOA / column-max VMR per species."""
    p_pa = prof["p"].to_numpy(dtype=float)
    i_surf = int(np.argmax(p_pa))  # highest pressure = surface
    i_toa = int(np.argmin(p_pa))  # lowest pressure = top
    p_surf_pa = p_pa[i_surf]
    rec = {"oc_P_surf_bar": p_surf_pa * BAR_PER_PA}
    for s in SPECIES:
        if s not in prof.columns:
            rec |= {
                f"oc_{s}_surf_vmr": math.nan,
                f"oc_{s}_obs_vmr": math.nan,
                f"oc_{s}_toa_vmr": math.nan,
                f"oc_{s}_max_vmr": math.nan,
                f"oc_{s}_surf_bar": math.nan,
            }
            continue
        v = prof[s].to_numpy(dtype=float)
        surf = v[i_surf]
        rec[f"oc_{s}_surf_vmr"] = surf
        rec[f"oc_{s}_surf_bar"] = surf * p_surf_pa * BAR_PER_PA
        rec[f"oc_{s}_toa_vmr"] = v[i_toa]
        rec[f"oc_{s}_max_vmr"] = float(np.max(v))
        rec[f"oc_{s}_obs_vmr"] = interp_vmr_at_pressure(p_pa, v, P_OBS_BAR / BAR_PER_PA)
    return rec


# ----------------------------------------------------------------------------
# Build the summary table
# ----------------------------------------------------------------------------
def build_summary() -> pd.DataFrame:
    rows = []
    print("Loading cases...")
    for case in sorted(GRID_DIR.glob("case_*")):
        rec = {"case": case.name, "status": read_status(case)}
        rec |= read_grid_params(case)
        hf = read_helpfile_final(case)
        rec["has_helpfile"] = hf is not None
        if hf:
            rec |= hf
        prof = read_vulcan_profile(case)
        rec["has_offchem"] = prof is not None
        if prof is not None:
            rec["offchem_nlev"] = len(prof)
            rec |= summarise_offchem(prof)
        rows.append(rec)
    if not rows:
        raise SystemExit(
            f"No 'case_*' directories found under {GRID_DIR}\n"
            f"(resolved to {GRID_DIR.resolve()}). Check grid_name and that the "
            "data symlink points at the PROTEUS output tree."
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
# Shared alpha convention: escaped runs are drawn faint everywhere so
# they still read as data points but as lower-confidence than retained runs.
STABLE_ALPHA, UNSTABLE_ALPHA = 0.85, 0.22
# Cases whose modelled radius misses the measured L 98-59 d band are still
# shown (not dropped), but dimmed to this alpha so a radius match reads as
# higher-confidence without hiding the rest of the grid.
RADIUS_MATCH_ALPHA, RADIUS_MISS_ALPHA = 0.85, 0.20


def _is_unstable(status: pd.Series) -> np.ndarray:
    """True for escaped runs (matched against `status` text)."""
    return (
        status.astype(str)
        .str.contains("escaped", case=False, regex=True)
        .to_numpy(bool)
    )


def _is_failed(status: pd.Series) -> np.ndarray:
    """True for errored or died runs (matched against `status` text)."""
    return (
        status.astype(str)
        .str.contains("error|died", case=False, regex=True)
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
    unstable = _is_unstable(base["status"])
    keep = (base["P_surf_bar"] > 1.0) | unstable
    return base[keep].copy(), unstable[keep.to_numpy()]


def plot_eq_vs_offchem_photosphere(df: pd.DataFrame):
    """Bulk equilibrium VMR vs post-processed photosphere-level VMR.

    x: outgassed bulk (surface) mixing ratio from the helpfile.
    y: VULCAN VMR interpolated to the photosphere (p_obs = 0.02 bar), i.e. the
    abundance an observer sees. Departure from the 1:1 line is the combined
    effect of photochemistry and vertical structure at observable altitude.
    """
    sub = df[df["has_offchem"] & df["has_helpfile"] & (df["P_surf_bar"] > 1.0)].copy()
    lo, hi = 1e-12, 1.0
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for ax, s in zip(axes.ravel(), SPECIES):
        y = np.clip(sub[f"oc_{s}_obs_vmr"].to_numpy(float), VMR_FLOOR, None)
        eq_col = f"eq_{s}_vmr"
        has_eq = eq_col in sub.columns
        if has_eq:
            x = np.clip(sub[eq_col].to_numpy(float), VMR_FLOOR, None)
            ax.scatter(
                x, y, s=28, c=SPECIES_COLOR[s], edgecolor="k", linewidth=0.4, zorder=3
            )
        else:
            for yy in y:
                ax.axhline(
                    yy,
                    xmin=lo,
                    xmax=hi,
                    color=SPECIES_COLOR[s],
                    linewidth=1.6,
                    zorder=3,
                )
        ax.plot([lo, hi], [lo, hi], "--", color="0.4", linewidth=0.8, zorder=1)
        title = s if has_eq else f"{s} (kinetics, VULCAN-only)"
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(lo, hi),
            ylim=(lo, hi),
            title=title,
            xlabel="equilibrium bulk VMR",
            ylabel="offchem photosphere VMR",
        )
    for ax in axes.ravel()[len(SPECIES) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Equilibrium outgassing (bulk) vs VULCAN at the photosphere (p_obs = 0.02 bar)\n"
        "(1:1 line = observable abundance matches the outgassed prediction)",
        fontsize=11,
    )
    fig.savefig(OUT_DIR / "fig_eq_vs_offchem_photosphere.png")
    plt.close(fig)


def plot_profiles(df: pd.DataFrame, ncols: int = 4):
    """VMR(p) profiles for the 7 species, cases inside the measured radius band.

    Selection: retained atmospheres (P_surf > 1 bar) whose modelled transit
    radius R_obs lies within the measured L 98-59 d 1-sigma band
    (R_OBS_MEAS +/- R_OBS_MEAS_ERR). Unlike `plot_atm_tp_profiles`, misses
    are dropped rather than dimmed -- one subplot per case, so including
    every retained case (dozens) would make the figure unusably large. The
    observation pressure level (p_obs) is marked so the observable part of
    each profile is clear.
    """
    lo, hi = R_OBS_MEAS - R_OBS_MEAS_ERR, R_OBS_MEAS + R_OBS_MEAS_ERR
    sub = df[
        df["has_offchem"]
        & (df["P_surf_bar"] > 1.0)
        & df["R_obs_Rearth"].between(lo, hi)
    ].copy()
    sub = sub.sort_values("R_obs_Rearth")
    picks = sub["case"].tolist()
    if not picks:
        return
    n = len(picks)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, cname in zip(axes, picks):
        prof = read_vulcan_profile(GRID_DIR / cname)
        p_bar = prof["p"].to_numpy(float) * BAR_PER_PA
        for s in SPECIES:
            if s not in prof.columns:
                continue
            v = np.clip(prof[s].to_numpy(float), VMR_FLOOR, None)
            ax.plot(v, p_bar, color=SPECIES_COLOR[s], label=s, linewidth=1.6)
        ax.axhline(P_OBS_BAR, color="0.4", linestyle=":", linewidth=1.0, zorder=1)
        row = df[df["case"] == cname].iloc[0]
        # Pressure decreases upward: surface (high P) at the bottom, TOA at top.
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(1e-10, 2.0),
            ylim=(float(np.nanmax(p_bar)) * 1.5, float(np.nanmin(p_bar)) * 0.6),
            title=(
                f"{cname}  R={row['R_obs_Rearth']:.3f} R$_\\oplus$\n"
                f"P$_s$={row['P_surf_bar']:.0f} bar, "
                f"T$_s$={row['T_surf_K']:.0f} K\n"
                f"H={row['H_budget']:.0f} fO2={row['fO2_shift_IW']:+.0f} "
                f"S={row['S_budget']:.0f} C={row['C_budget']:.0f}"
            ),
        )
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[:n]:
        ax.set_ylabel("Pressure [bar]")
        ax.set_xlabel("VMR")
    axes[0].legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.suptitle(
        "VULCAN post-processed profiles: cases matching measured radius "
        f"{R_OBS_MEAS}$\\pm${R_OBS_MEAS_ERR} R$_\\oplus$ "
        "(dotted line = observation level p_obs)",
        fontsize=11,
    )
    fig.tight_layout(h_pad=0.3)
    fig.savefig(OUT_DIR / "fig_offchem_profiles.png", bbox_inches="tight")
    plt.close(fig)


def _radius_matched_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Retained cases (P_surf > 1 bar) whose modelled R_obs matches the
    measured L 98-59 c 1-sigma radius band, sorted by R_obs.
    """
    lo, hi = R_OBS_MEAS - R_OBS_MEAS_ERR, R_OBS_MEAS + R_OBS_MEAS_ERR
    mask = (
        df["has_helpfile"]
        & (df["P_surf_bar"] > 1.0)
        & df["R_obs_Rearth"].between(lo, hi)
    )
    return df[mask].sort_values("R_obs_Rearth").copy()


def _radius_match_mask(sub: pd.DataFrame) -> np.ndarray:
    """True where `sub['R_obs_Rearth']` falls within the measured L 98-59 d
    1-sigma radius band (R_OBS_MEAS +/- R_OBS_MEAS_ERR).
    """
    lo, hi = R_OBS_MEAS - R_OBS_MEAS_ERR, R_OBS_MEAS + R_OBS_MEAS_ERR
    return sub["R_obs_Rearth"].between(lo, hi).to_numpy(bool)


def _draw_temperature_reference(ax):
    """Overlay the observed L 98-59 d temperature (T_OBS_MEAS, solid black)
    with its asymmetric 1-sigma band shaded, annotated directly. The x-view
    is widened (never narrowed) so the band stays visible even when the
    plotted temperatures don't themselves reach that range.
    """
    lo = T_OBS_MEAS - T_OBS_MEAS_ERR_LO
    hi = T_OBS_MEAS + T_OBS_MEAS_ERR_HI
    xmin, xmax = ax.get_xlim()
    xmin = min(xmin, lo * 0.95)
    xmax = max(xmax, hi * 1.05)
    ax.set_xlim(xmin, xmax)
    ax.axvspan(lo, hi, color=WONG["black"], alpha=0.12, zorder=1)
    ax.axvline(T_OBS_MEAS, color=WONG["black"], linestyle="-", linewidth=1.4, zorder=2)
    ax.text(
        T_OBS_MEAS,
        0.02,
        f"observed {T_OBS_MEAS:.0f}"
        f"$^{{+{T_OBS_MEAS_ERR_HI:.0f}}}_{{-{T_OBS_MEAS_ERR_LO:.0f}}}$ K",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=7,
        color=WONG["black"],
        zorder=5,
    )


def plot_atm_tp_profiles(df: pd.DataFrame):
    """AGNI radiative-convective T(p) profiles for all retained-atmosphere
    cases, combined onto a single panel and coloured by fO2 shift.

    Selection: retained atmosphere (P_surf > 1 bar) -- not restricted to the
    measured radius band. Cases whose modelled R_obs falls within the
    measured L 98-59 d 1-sigma band (R_OBS_MEAS +/- R_OBS_MEAS_ERR) are
    drawn at full opacity; the rest are still shown, dimmed, so a radius
    miss stays visible as context (with no matches for grid4, dropping
    misses left this figure empty). Drawn from the final AGNI `*_atm.nc`
    snapshot rather than the post-processed VULCAN profile, so this shows
    the physical (bulk-equilibrium) thermal structure independent of
    photochemistry. The observed L 98-59 d temperature (T_OBS_MEAS, with its
    asymmetric error band) is overlaid for comparison.
    """
    sub = df[df["has_helpfile"] & (df["P_surf_bar"] > 1.0)].copy()
    sub = sub.sort_values("R_obs_Rearth")
    if sub.empty:
        return
    matched = _radius_match_mask(sub)
    f_vals = sorted(sub["fO2_shift_IW"].dropna().unique())
    f_colors = fo2_colors(f_vals)

    fig, ax = plt.subplots(figsize=(7, 6.5), constrained_layout=True)
    p_bar_all = []
    n_plotted = 0
    for cname, fv, is_matched in zip(sub["case"], sub["fO2_shift_IW"], matched):
        prof = read_agni_profile(GRID_DIR / cname)
        if prof is None:
            continue
        p_bar = prof["p"] * BAR_PER_PA
        ax.plot(
            prof["t"],
            p_bar,
            color=f_colors.get(fv, WONG["black"]),
            linewidth=1.4,
            alpha=(RADIUS_MATCH_ALPHA if is_matched else RADIUS_MISS_ALPHA),
            zorder=(3 if is_matched else 2),
        )
        p_bar_all.append(p_bar)
        n_plotted += 1
    if n_plotted == 0:
        plt.close(fig)
        return
    all_p = np.concatenate(p_bar_all)
    ax.axhline(P_OBS_BAR, color="0.4", linestyle=":", linewidth=1.0, zorder=1)
    ax.set(
        yscale="log",
        ylim=(float(np.nanmax(all_p)) * 1.5, float(np.nanmin(all_p)) * 0.6),
        xlabel="Temperature [K]",
        ylabel="Pressure [bar]",
    )
    _draw_temperature_reference(ax)
    for fv in f_vals:
        ax.plot([], [], color=f_colors[fv], linewidth=1.4, label=f"fO2={fv:+.0f}")
    ax.legend(fontsize=9, title=fo2_lbl, loc="best")
    fig.suptitle(
        "AGNI atmosphere T(p) profiles: all retained-atmosphere cases\n"
        "(colour = fO2 shift; faint = misses measured radius;\n"
        "dotted = p_obs; solid black = observed temperature)",
        fontsize=10,
    )
    fig.savefig(OUT_DIR / "fig_atm_tp_profiles.png")
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
    sub = sub[sub["has_offchem"]]
    if sub.empty:
        return
    prof_dir = OUT_DIR / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for cname in sub["case"]:
        case = GRID_DIR / cname
        atm = read_agni_profile(case)
        vulcan = read_vulcan_profile(case)
        if atm is None or vulcan is None:
            continue
        cols = {"P_bar": atm["p"] * BAR_PER_PA, "T_K": atm["t"], "Z_m": atm["z"]}
        vulcan_p = vulcan["p"].to_numpy(float)
        for sp in SPECIES:
            if sp not in vulcan.columns:
                continue
            cols[f"{sp}_vmr"] = interp_vmr_profile(
                vulcan_p, vulcan[sp].to_numpy(float), atm["p"]
            )
        pd.DataFrame(cols).to_csv(prof_dir / f"{cname}_atm_profile.csv", index=False)
        n_written += 1
    print(f"Wrote {n_written} atmosphere profile CSV(s) to {prof_dir}")


# Shared log10(VMR) colour-scale window so all panels/species use one colourbar.
VMR_COLOR_LO, VMR_COLOR_HI = 1e-10, 1.0
VMR_CMAP = "gnuplot2"


def plot_grid_dependence(df: pd.DataFrame, where="obs"):
    """Offchem photosphere VMR across the redox / sulfur grid, for retained cases.

    x: fO2 shift [log10 dIW]; y: S/H mass ratio; marker colour (via a shared
    colourbar) encodes the post-processed photosphere VMR. One panel per
    species. Cases that share an (fO2, S/H) node (differing C or H budget) are
    xy-jittered so they do not fully overplot. Escaped cases are kept
    (rather than dropped by the P_surf > 1 bar retained-atmosphere cut) and
    drawn at low opacity so they read as lower-confidence.
    """
    sub, unstable = _select_retained_or_unstable(df, df["has_offchem"])
    if sub.empty:
        return
    # Jitter degenerate (fO2, S/H) nodes apart along x, keyed by C then H budget.
    c_vals = sorted(sub["C_budget"].dropna().unique())
    h_vals = sorted(sub["H_budget"].dropna().unique())

    def _jit(row):
        ci = c_vals.index(row["C_budget"]) if row["C_budget"] in c_vals else 0
        hi = h_vals.index(row["H_budget"]) if row["H_budget"] in h_vals else 0
        span = max(len(c_vals) * max(len(h_vals), 1), 1)
        return (ci * max(len(h_vals), 1) + hi - (span - 1) / 2) * 0.045

    # jitter
    x_jit = np.copy(sub.apply(_jit, axis=1).to_numpy(float))
    y_jit = np.copy(sub.apply(_jit, axis=1).to_numpy(float))

    # shuffle the jitter arrays
    np.random.seed(42)
    np.random.shuffle(x_jit)
    np.random.shuffle(y_jit)

    x = sub["fO2_shift_IW"].to_numpy(float) + x_jit
    y = sub["S_budget"].to_numpy(float) + y_jit
    norm = LogNorm(vmin=VMR_COLOR_LO, vmax=VMR_COLOR_HI)

    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for ax, s in zip(axes.ravel(), SPECIES):
        vmr = np.clip(sub[f"oc_{s}_{where}_vmr"].to_numpy(float), VMR_FLOOR, None)
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
                edgecolor="k",
                linewidth=0.3,
                alpha=alpha,
                zorder=3,
            )
        ax.set(
            title=s,
            xlabel=fo2_lbl,
            ylabel="S/H mass ratio",
        )
    for ax in axes.ravel()[len(SPECIES) :]:
        ax.set_visible(False)
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=VMR_CMAP),
        ax=axes.ravel().tolist(),
        label="photosphere VMR",
        shrink=0.85,
    )
    fig.suptitle(
        f"Gas VMR at {where}-layer, versus grid axes\n(colour = VMR; faint = escaped runs)",
        fontsize=11,
    )
    fig.savefig(OUT_DIR / f"fig_grid_dependence_{where}.png")
    plt.close(fig)


# Photosphere abundance ratios of interest and the grid axes to correlate against.
RATIOS = [("CS2", "SO2"), ("SO2", "H2S"), ("H2S", "CS2")]
RATIO_COLOR = {
    ("CS2", "SO2"): WONG["vermilion"],
    ("SO2", "H2S"): WONG["blue"],
    ("H2S", "CS2"): WONG["green"],
}
PARAM_LABEL = {
    "H_budget": "H budget",
    "fO2_shift_IW": fo2_lbl,
    "S_budget": "S/H mass ratio",
    "C_budget": "C/H mass ratio",
    "core_frac": "CMF",
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
    return param == "H_budget" or (pos.size > 0 and pos.max() / pos.min() > 20)


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
    sub, unstable = _select_retained_or_unstable(df, df["has_offchem"])
    if sub.empty:
        return
    lo_r, hi_r = R_OBS_MEAS - R_OBS_MEAS_ERR, R_OBS_MEAS + R_OBS_MEAS_ERR
    match = (sub["has_helpfile"] & sub["R_obs_Rearth"].between(lo_r, hi_r)).to_numpy(
        bool
    )

    def _scatter(ax, x, y, color):
        # Split by radius match (circle vs star) and by run stability (alpha).
        shapes = ((~match, "o", 0.3, 2), (match, "*", 0.6, 6))
        for msk, marker, mk_lw, mk_z in shapes:
            for stab_msk, alpha in (
                (~unstable, STABLE_ALPHA),
                (unstable, UNSTABLE_ALPHA),
            ):
                m = msk & stab_msk & np.isfinite(x) & np.isfinite(y)
                if not m.any():
                    continue
                ax.scatter(
                    x[m],
                    y[m],
                    marker=marker,
                    s=(150 if marker == "*" else 34),
                    c=color,
                    edgecolor="k",
                    linewidth=mk_lw,
                    zorder=mk_z,
                    alpha=alpha,
                )

    nrows, ncols = len(RATIOS) + 1, len(GRID_KEYS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), constrained_layout=True
    )
    for i, (num, den) in enumerate(RATIOS):
        n = np.clip(sub[f"oc_{num}_obs_vmr"].to_numpy(float), VMR_FLOOR, None)
        d = np.clip(sub[f"oc_{den}_obs_vmr"].to_numpy(float), VMR_FLOOR, None)
        ratio = n / d
        color = RATIO_COLOR[(num, den)]
        for j, param in enumerate(GRID_KEYS):
            ax = axes[i, j]
            x = sub[param].to_numpy(float)
            # Per-panel symmetric ylim: only cases with a finite value for
            # *this* grid parameter set the view, since that can differ by column.
            ylim = _symmetric_ratio_ylim(ratio[np.isfinite(x)])
            ax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8, zorder=1)
            _scatter(ax, x, ratio, color)
            ax.set(yscale="log", ylim=ylim)
            if _use_logx(x, param):
                ax.set_xscale("log")
            # if j == 0:
            ax.set_ylabel(f"{num}/{den} VMR")

    # Final row: bulk atmospheric mean molecular weight (helpfile), g/mol.
    mmw = sub["mu_atm_kg_mol"].to_numpy(float) * 1.0e3
    i = len(RATIOS)
    for j, param in enumerate(GRID_KEYS):
        ax = axes[i, j]
        x = sub[param].to_numpy(float)
        _scatter(ax, x, mmw, WONG["black"])
        ax.set_ylim(0, 32)
        if _use_logx(x, param):
            ax.set_xscale("log")
        # if j == 0:
        ax.set_ylabel("atm MMW [g/mol]")
        ax.set_xlabel(PARAM_LABEL[param])

    fig.suptitle(
        "Photosphere abundance ratios and atmosphere MMW (p_obs = 0.02 bar) "
        "vs grid parameters\n(stars = radius-matched; faint = escaped "
        "runs; dashed line = ratio 1)",
        fontsize=11,
    )
    fig.savefig(OUT_DIR / "fig_ratio_correlations.png")
    plt.close(fig)


def plot_cs2(df: pd.DataFrame):
    """CS2 is VULCAN-only: photosphere-level VMR across the grid.

    Escaped cases are kept (rather than dropped by the retained-
    atmosphere cut) and drawn at low opacity so they read as lower-confidence;
    they are excluded from the legend to avoid duplicate entries.
    """
    sub, unstable = _select_retained_or_unstable(df, df["has_offchem"])
    if sub.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    c_vals = sorted(sub["C_budget"].dropna().unique())
    c_colors = {
        v: c
        for v, c in zip(
            c_vals,
            [(1.0, 1.0 - v / len(c_vals), v / len(c_vals)) for v in range(len(c_vals))],
        )
    }
    for cv in c_vals:
        gm = (sub["C_budget"] == cv).to_numpy(bool)
        y = np.clip(sub["oc_CS2_obs_vmr"].to_numpy(float), VMR_FLOOR, None)
        x = sub["fO2_shift_IW"].to_numpy(float)
        for stab_msk, alpha, label in (
            (~unstable, STABLE_ALPHA, f"C/H={cv:.0f}"),
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
                edgecolor="k",
                linewidth=0.3,
                alpha=alpha,
                label=label,
            )
    a1.set(
        yscale="log",
        xlabel=fo2_lbl,
        ylabel="CS2 photosphere VMR",
        title="CS2 photosphere abundance vs redox",
    )
    a1.legend(fontsize=9)
    # CS2 photosphere vs C_budget, coloured by fO2
    c_vals = sorted(sub["C_budget"].dropna().unique())
    c_colors = {
        v: c
        for v, c in zip(
            c_vals,
            [(1.0, 1.0 - v / len(c_vals), v / len(c_vals)) for v in range(len(c_vals))],
        )
    }
    for cv in c_vals:
        gm = (sub["C_budget"] == cv).to_numpy(bool)
        y = np.clip(sub["oc_CS2_obs_vmr"].to_numpy(float), VMR_FLOOR, None)
        x = sub["C_budget"].to_numpy(float)
        for stab_msk, alpha, label in (
            (~unstable, STABLE_ALPHA, f"C/H={cv:.0f}"),
            (unstable, UNSTABLE_ALPHA, None),
        ):
            m = gm & stab_msk
            if not m.any():
                continue
            a2.scatter(
                x[m],
                y[m],
                s=34,
                c=c_colors[cv],
                edgecolor="k",
                linewidth=0.3,
                alpha=alpha,
                label=label,
            )
    a2.set(
        yscale="log",
        xlabel="C/H mass ratio",
        ylabel="CS2 photosphere VMR",
        title="CS2 photosphere abundance vs C budget",
    )
    a2.legend(fontsize=9)
    fig.suptitle(
        "CS2 at the photosphere (p_obs = 0.02 bar; produced only in VULCAN)\n"
        "(faint = escaped runs)",
        fontsize=11,
    )
    fig.savefig(OUT_DIR / "fig_cs2.png")
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
    sub, unstable = _select_retained_or_unstable(df, df["has_helpfile"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(sub))
    r = sub["R_obs_Rearth"].to_numpy(float)
    s = (
        np.log10(sub["H_budget"].to_numpy(float)) * 5 + 20.0
    )  # marker size keyed to H budget
    magma = sub["T_surf_K"].to_numpy(float) > SOLIDUS_T_K
    shape_groups = (
        (~magma, "o", "likely solidified"),
        (magma, "s", f"likely has magma ocean (T$_s$>{SOLIDUS_T_K:.0f} K)"),
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
                c=WONG["blue"],
                edgecolor="k",
                linewidth=0.3,
                zorder=3,
                alpha=alpha,
                label=(base_label if use_label else None),
            )
    ax.axhspan(
        R_OBS_MEAS - R_OBS_MEAS_ERR,
        R_OBS_MEAS + R_OBS_MEAS_ERR,
        color=WONG["orange"],
        alpha=0.25,
        zorder=1,
        label=f"measured {R_OBS_MEAS}$\\pm${R_OBS_MEAS_ERR} R$_\\oplus$",
    )
    ax.axhline(R_OBS_MEAS, color=WONG["orange"], linewidth=1.0, zorder=2)
    ax.set(
        ylabel="planet radius [R$_\\oplus$]",
        title="Modelled vs measured radius\n"
        f"(faint = escaped runs; squares = magma ocean, T$_s$>{SOLIDUS_T_K:.0f} K)",
    )
    ax.set_xticks(x)
    labels = [
        f"{c.replace('case_0000', '')}, IW={f:+.0f}, H={h:.1e}, C/H={ch:.0f}"
        for c, f, h, ch in zip(
            sub["case"], sub["fO2_shift_IW"], sub["H_budget"], sub["C_budget"]
        )
    ]
    ax.set_xticklabels(
        labels, fontsize=5, rotation=90, ha="right", rotation_mode="anchor"
    )
    ax.legend(fontsize=9)
    fig.savefig(OUT_DIR / "fig_radius.png")
    plt.close(fig)


# Metal admixture drawn as a dashed line alongside each species' solid pure
# (f_metal=1) reference line in `plot_density_vs_mmw`. 10% dilution means
# 90% metal + 10% Jupiter-ratio H2/He by mass, i.e. f_metal = 1 - 0.1.
DILUTION_FRAC = 0.9


def _read_zeng_mass_radius(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a Zeng et al. (2019) mass-radius curve from $FWL_DATA/mass_radius/
    Zeng2019/<filename>: two whitespace-separated columns (mass, radius),
    both in Earth units, ascending in mass.
    """
    path = resolve_fwl_data_dir() / "mass_radius" / "Zeng2019" / filename
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def _earthlike_density_gcc(mass_mearth: float) -> float:
    """Bulk density [g/cm^3] of an Earth-composition rocky planet (Zeng
    et al. 2019 'Earth-like rocky' curve: Earth's core mass fraction, no
    volatile envelope) at `mass_mearth`, read off `massradiusEarthlikeRocky.
    txt` and converted via M/((4/3) pi R^3). Density is mass-dependent for a
    fixed composition (self-compression), so this differs from Earth's own
    actual mean density once evaluated away from 1 M_earth.
    """
    m, r = _read_zeng_mass_radius("massradiusEarthlikeRocky.txt")
    r_mearth = np.interp(mass_mearth, m, r)
    return (
        mass_mearth
        * M_EARTH
        / ((4.0 / 3.0) * math.pi * (r_mearth * R_EARTH) ** 3)
        * 1.0e-3
    )


# Density reference points for `plot_density_vs_mmw`, g/cm^3.
RHO_WATER_GCC = 1.0  # liquid water at STP
# Earth-composition (Zeng et al. 2019) density evaluated at the *observed*
# L 98-59 d mass (M_OBS_MEAS), not Earth's own actual mean density -- a
# fixed rocky composition compresses to higher density at higher mass, so
# the comparison needs to be made at this planet's mass, not Earth's.
RHO_EARTH_GCC = _earthlike_density_gcc(M_OBS_MEAS)
# Observed L 98-59 d bulk density from the measured mass and radius
# (M_OBS_MEAS, R_OBS_MEAS), kg/m^3 -> g/cm^3. Does not propagate the
# M_OBS_MEAS_ERR / R_OBS_MEAS_ERR uncertainties -- point estimate only.
RHO_OBS_GCC = (
    M_OBS_MEAS
    * M_EARTH
    / ((4.0 / 3.0) * math.pi * (R_OBS_MEAS * R_EARTH) ** 3)
    * 1.0e-3
)


def _draw_mmw_reference(ax):
    """Overlay MMW reference lines on a density-vs-MMW axis: a dash-dot line
    for pure Jupiter-ratio H2/He, a solid line per metal species at its pure
    (f_metal=1) end-member, and a dashed line (same colour) at 10% dilution
    (f_metal=DILUTION_FRAC, i.e. 90% metal + 10% Jupiter-ratio H2/He by
    mass). Each line is annotated directly (composition + MMW) rather than
    via the legend; labels are staggered onto rows ranked by MMW so nearby
    values (the diluted end-members bunch within a few g/mol of each other,
    since MMW is mole- not mass-weighted) don't overlap. Composition-only
    (see `mmw_h_he_metal`); does not predict where a point should fall in y.
    """
    xaxis_t = ax.get_xaxis_transform()  # x in data coords, y in axes coords
    row_step = 0.09
    cluster_gap_gmol = 3.0  # values closer than this (in MMW) drop to the next row

    def _staggered_rows(mu_by_species: dict[str, float], y0: float):
        """species -> y (axes coords), dropping to the next row only when the
        previous (sorted-by-MMW) label is within `cluster_gap_gmol` of it."""
        prev_mu, row = None, 0
        for species in sorted(mu_by_species, key=mu_by_species.get):
            mu = mu_by_species[species]
            row = (
                row + 1
                if prev_mu is not None and (mu - prev_mu) < cluster_gap_gmol
                else 0
            )
            prev_mu = mu
            yield species, y0 - row * row_step

    mu_h_he = mmw_h_he_metal(0.0, 1.0)
    ax.axvline(mu_h_he, color=get_colour("H2"), linestyle="-.", linewidth=1.1, zorder=1)
    ax.text(
        mu_h_he,
        0.02,
        f"H$_2$/He {mu_h_he:.1f}",
        transform=xaxis_t,
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=6.5,
        color=get_colour("H2"),
        zorder=5,
    )

    mu_pure = {s: mmw_h_he_metal(1.0, m) for s, m in METAL_M_GMOL.items()}
    mu_dil = {s: mmw_h_he_metal(DILUTION_FRAC, m) for s, m in METAL_M_GMOL.items()}
    for species, m_metal in METAL_M_GMOL.items():
        color = get_colour(species)
        ax.axvline(
            mu_pure[species],
            color=color,
            linestyle="-",
            linewidth=1.2,
            alpha=0.7,
            zorder=1,
        )
        ax.axvline(
            mu_dil[species],
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            zorder=1,
        )
    for species, y in _staggered_rows(mu_pure, 0.98):
        ax.text(
            mu_pure[species],
            y,
            f"{species} ({mu_pure[species]:.0f} g/mol)",
            transform=xaxis_t,
            rotation=90,
            va="top",
            ha="right",
            fontsize=6.5,
            color=get_colour(species),
            zorder=5,
        )
    for species, y in _staggered_rows(mu_dil, 0.98):
        ax.text(
            mu_dil[species],
            y,
            f"{species}@10% ({mu_dil[species]:.0f} g/mol)",
            transform=xaxis_t,
            rotation=90,
            va="top",
            ha="right",
            fontsize=6.5,
            color=get_colour(species),
            alpha=0.85,
            zorder=5,
        )


def _draw_density_reference(ax):
    """Overlay horizontal density reference lines: liquid water (1 g/cm^3),
    an Earth-composition rocky planet at the L 98-59 d mass (RHO_EARTH_GCC,
    Zeng et al. 2019 mass-radius curve evaluated at M_OBS_MEAS -- not
    Earth's own actual density, since self-compression makes a fixed rocky
    composition denser at higher mass), and the observed L 98-59 d bulk
    density (RHO_OBS_GCC, from the measured mass and radius, solid black),
    each annotated directly rather than via the legend. The y-view is
    widened (never narrowed) so all three references stay visible even when
    the plotted densities don't themselves reach that low/high.
    """
    ymin, ymax = ax.get_ylim()
    ymin = min(ymin, RHO_WATER_GCC * 0.9, RHO_OBS_GCC * 0.9)
    ymax = max(ymax, RHO_EARTH_GCC * 1.05, RHO_OBS_GCC * 1.05)
    ax.set_ylim(ymin, ymax)
    yaxis_t = ax.get_yaxis_transform()  # x in axes coords, y in data coords
    ax.axhline(
        RHO_WATER_GCC, color=WONG["skyblue"], linestyle=":", linewidth=1.2, zorder=1
    )
    ax.text(
        0.99,
        RHO_WATER_GCC,
        f"water {RHO_WATER_GCC:.1f} g/cm$^3$",
        transform=yaxis_t,
        va="bottom",
        ha="right",
        fontsize=6.5,
        color=WONG["skyblue"],
        zorder=5,
    )
    ax.axhline(
        RHO_EARTH_GCC, color=WONG["vermilion"], linestyle=":", linewidth=1.2, zorder=1
    )
    ax.text(
        0.99,
        RHO_EARTH_GCC,
        f"Earth comp. @ M$_{{obs}}$ {RHO_EARTH_GCC:.2f} g/cm$^3$",
        transform=yaxis_t,
        va="bottom",
        ha="right",
        fontsize=6.5,
        color=WONG["vermilion"],
        zorder=5,
    )
    ax.axhline(RHO_OBS_GCC, color=WONG["black"], linestyle="-", linewidth=1.4, zorder=2)
    ax.text(
        0.99,
        RHO_OBS_GCC,
        f"observed {RHO_OBS_GCC:.2f} g/cm$^3$",
        transform=yaxis_t,
        va="bottom",
        ha="right",
        fontsize=6.5,
        color=WONG["black"],
        zorder=5,
    )


def plot_density_vs_mmw(df: pd.DataFrame):
    """Bulk density vs atmosphere mean molecular weight, at t=1 yr and t=end.

    x: atmosphere MMW (helpfile `atm_kg_per_mol`, converted to g/mol).
    y: bulk density (log scale) from the modelled transit radius and planet
    mass (helpfile `rho_obs`, converted kg/m^3 -> g/cm^3). Left panel uses
    each case's helpfile row nearest Time=1 yr; right panel uses each case's
    final row (t=end, which differs case to case for runs that stopped
    early, e.g. via escape). Horizontal dotted lines mark liquid water, an
    Earth-composition rocky planet at the L 98-59 d mass (Zeng et al. 2019),
    and the observed density (solid black), for scale. Vertical lines mark
    the MMW of a
    Jupiter-ratio (Galileo probe) H2/He gas admixed with CO2, H2O, CO, CH4
    or O2 -- solid at the pure (100%) metal end-member, dashed at 10%
    dilution by H2/He (see `mmw_h_he_metal`) -- a composition-only guide,
    not a density prediction, since density also depends on pressure,
    temperature and self-gravity that this reference ignores. Escaped cases
    are kept, drawn at low opacity.
    """
    sub, unstable = _select_retained_or_unstable(df, df["has_helpfile"])
    if sub.empty:
        return

    f_vals = sorted(sub["fO2_shift_IW"].dropna().unique())
    f_colors = fo2_colors(f_vals)

    tend = read_helpfile_timeseries(GRID_DIR / sub["case"].iloc[0], ["Time"])[
        "Time"
    ].values.flatten()[-1]
    tend = float(tend) / 1.0e9  # Gyr

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.5), constrained_layout=True, sharey=True
    )
    for ax, target_yr, title in (
        (ax1, 1.0, "t = 1 yr"),
        (ax2, None, f"t = {tend:.1f} Gyr"),
    ):
        n_plotted = 0
        for cname, fv, is_unstable in zip(sub["case"], sub["fO2_shift_IW"], unstable):
            ts = read_helpfile_timeseries(
                GRID_DIR / cname, ["rho_obs", "atm_kg_per_mol"]
            )
            if ts is None:
                continue
            row = (
                ts.iloc[(ts["Time"] - target_yr).abs().idxmin()]
                if target_yr is not None
                else ts.iloc[-1]
            )
            mu = row["atm_kg_per_mol"] * 1.0e3
            rho = row["rho_obs"] * 1.0e-3  # kg/m^3 -> g/cm^3
            if not (np.isfinite(mu) and np.isfinite(rho)):
                continue
            ax.scatter(
                mu,
                rho,
                s=40,
                color=f_colors.get(fv, WONG["black"]),
                edgecolor="k",
                linewidth=0.3,
                alpha=(UNSTABLE_ALPHA if is_unstable else STABLE_ALPHA),
                zorder=3,
            )
            n_plotted += 1
        ax.set(
            xlabel="Atmosphere MMW [g/mol]",
            xlim=(1.0, 50.0),
            ylabel="Bulk density [g/cm$^3$]",
            yscale="log",
            title=title,
        )
        if n_plotted == 0:
            continue
        _draw_density_reference(ax)
        _draw_mmw_reference(ax)
    if f_vals:
        for fv, c in f_colors.items():
            ax1.scatter(
                [], [], color=c, edgecolor="k", linewidth=0.3, label=f"{fv:+.0f}"
            )
        ax1.legend(fontsize=8, loc="lower left", title=fo2_lbl)
    fig.suptitle(
        "Bulk density vs atmosphere MMW",
        fontsize=11,
    )
    ax1.invert_yaxis()
    fig.savefig(OUT_DIR / "fig_density_vs_mmw.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def print_report(df: pd.DataFrame):
    n = len(df)
    n_off = int(df["has_offchem"].sum())
    n_retained = int((df["has_offchem"] & (df["P_surf_bar"] > 1.0)).sum())
    print("=" * 78)
    print(f"PROTEUS grid analysis  ({GRID_DIR})")
    print("=" * 78)
    print(f"cases total ............ {n}")
    print(f"with offchem/vulcan.csv  {n_off}")
    print(f"  of which retained P_s>1 bar ... {n_retained}")
    print(f"status breakdown:\n{df['status'].value_counts().to_string()}")
    print("-" * 78)

    ret = df[df["has_offchem"] & (df["P_surf_bar"] > 1.0)].copy()
    if not ret.empty:
        print(
            "Retained-atmosphere cases: photosphere VMR ranges (VULCAN, p_obs = 0.02 bar)"
        )
        for s in SPECIES:
            col = f"oc_{s}_obs_vmr"
            v = ret[col].replace(0, np.nan).dropna()
            if v.empty:
                print(f"  {s:4s}: all zero / absent")
                continue
            print(
                f"  {s:4s}: min={v.min():.2e}  median={v.median():.2e}  max={v.max():.2e}"
            )
        print("-" * 78)
        # equilibrium (bulk) vs offchem photosphere, median over retained cases
        print("Median VMR: equilibrium bulk (eq) vs post-processed photosphere (oc)")
        for s in HELPFILE_SPECIES:
            e = ret[f"eq_{s}_vmr"].replace(0, np.nan).dropna()
            o = ret[f"oc_{s}_obs_vmr"].replace(0, np.nan).dropna()
            em = e.median() if not e.empty else float("nan")
            om = o.median() if not o.empty else float("nan")
            print(f"  {s:4s}: eq={em:.2e}   oc={om:.2e}")
        print("-" * 78)
        print("CS2 (VULCAN-only) photosphere VMR by case:")
        for _, row in ret.sort_values("oc_CS2_obs_vmr", ascending=False).iterrows():
            print(
                f"  {row['case']}  photosphere={row['oc_CS2_obs_vmr']:.2e}  "
                f"(surface={row['oc_CS2_surf_vmr']:.2e}) "
                f"(fO2={row['fO2_shift_IW']:+.0f} S={row['S_budget']:.0f} "
                f"C={row['C_budget']:.0f} H={row['H_budget']:.0f})"
            )
        print("-" * 78)
        r = ret["R_obs_Rearth"].dropna()
        print(
            f"Modelled transit radius: min={r.min():.3f} median={r.median():.3f} "
            f"max={r.max():.3f} R_earth"
        )
        print(
            f"Measured radius: R={R_OBS_MEAS}+/-{R_OBS_MEAS_ERR} R_earth, "
            f"Measured mass:   M={M_OBS_MEAS}+/-{M_OBS_MEAS_ERR} M_earth"
        )
    print("=" * 78)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_summary()
    df.to_csv(OUT_DIR / "summary_grid.csv", index=False)
    print_report(df)

    plot_radius(df)
    plot_density_vs_mmw(df)

    plot_profiles(df)
    plot_atm_tp_profiles(df)
    write_atm_profile_csvs(df)

    plot_eq_vs_offchem_photosphere(df)
    plot_grid_dependence(df, where="obs")
    plot_grid_dependence(df, where="surf")
    plot_ratio_correlations(df)
    plot_cs2(df)

    print(f"\nWrote summary_grid.csv and figures to:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
