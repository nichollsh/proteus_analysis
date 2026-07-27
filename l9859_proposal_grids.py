#!/usr/bin/env python3
# Analyse the L 98-59 c PROTEUS grids.
from __future__ import annotations

import math
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.plot import get_colour

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Anchor paths to this script's directory so the script runs from any cwd.
# `nogit_analysis/data` is a symlink to the PROTEUS output tree.
SCRIPT_DIR = Path(__file__).resolve().parent
grid_name = "l9859c_grid4"
GRID_DIR = SCRIPT_DIR / "data" / grid_name
OUT_DIR = SCRIPT_DIR / "output" / f"{grid_name}_analysis"

SPECIES = ["CS2", "SO2", "H2", "H2S", "CO2", "CH4", "H2O", "S2"]
# Species also present in the equilibrium helpfile (CS2 is VULCAN-only).
HELPFILE_SPECIES = ["SO2", "H2", "H2S", "CO2", "CH4", "H2O", "S2"]

GRID_KEYS = ["H_budget", "fO2_shift_IW", "S_budget", "C_budget"]

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

# Wong colourblind-friendly palette
WONG = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermilion": "#D55E00", "purple": "#CC79A7",
}
# Species colours follow the PROTEUS ecosystem style (proteus.utils.plot).
# get_colour returns the preset hue per gas, or generates one from composition
# for species without a preset (e.g. CS2).
SPECIES_COLOR = {s: get_colour(s) for s in SPECIES}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 160, "savefig.dpi": 160,
})


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


def interp_vmr_at_pressure(p_pa: np.ndarray, vmr: np.ndarray, p_target_pa: float) -> float:
    """Log-log interpolate VMR to p_target. Returns NaN if out of range."""
    order = np.argsort(p_pa)
    p_s = p_pa[order]
    v_s = np.clip(vmr[order], VMR_FLOOR, None)
    if p_target_pa < p_s[0] or p_target_pa > p_s[-1]:
        return math.nan
    logv = np.interp(np.log10(p_target_pa), np.log10(p_s), np.log10(v_s))
    return 10.0**logv


def summarise_offchem(prof: pd.DataFrame) -> dict:
    """Surface / observation-level / TOA / column-max VMR per species."""
    p_pa = prof["p"].to_numpy(dtype=float)
    i_surf = int(np.argmax(p_pa))   # highest pressure = surface
    i_toa = int(np.argmin(p_pa))    # lowest pressure = top
    p_surf_pa = p_pa[i_surf]
    rec = {"oc_P_surf_bar": p_surf_pa * BAR_PER_PA}
    for s in SPECIES:
        if s not in prof.columns:
            rec |= {f"oc_{s}_surf_vmr": math.nan, f"oc_{s}_obs_vmr": math.nan,
                    f"oc_{s}_toa_vmr": math.nan, f"oc_{s}_max_vmr": math.nan,
                    f"oc_{s}_surf_bar": math.nan}
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
def plot_eq_vs_offchem_photosphere(df: pd.DataFrame):
    """Bulk equilibrium VMR vs post-processed photosphere-level VMR.

    x: outgassed bulk (surface) mixing ratio from the helpfile.
    y: VULCAN VMR interpolated to the photosphere (p_obs = 0.02 bar), i.e. the
    abundance an observer sees. Departure from the 1:1 line is the combined
    effect of photochemistry and vertical structure at observable altitude.
    """
    sub = df[df["has_offchem"] & df["has_helpfile"] & (df["P_surf_bar"] > 1.0)].copy()
    lo, hi = 1e-12, 1.0
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    for ax, s in zip(axes.ravel(), SPECIES):
        eq_col = f"eq_{s}_vmr"
        has_eq = eq_col in sub.columns
        if has_eq:
            x = np.clip(sub[eq_col].to_numpy(float), VMR_FLOOR, None)
        else:
            # VULCAN-only species (e.g. CS2): no outgassed counterpart. Pin at the
            # left-edge floor so the panel shows purely photochemical production.
            x = np.full(len(sub), lo, dtype=float)
        y = np.clip(sub[f"oc_{s}_obs_vmr"].to_numpy(float), VMR_FLOOR, None)
        ax.scatter(x, y, s=28, c=SPECIES_COLOR[s], edgecolor="k", linewidth=0.4, zorder=3)
        ax.plot([lo, hi], [lo, hi], "--", color="0.4", linewidth=0.8, zorder=1)
        title = s if has_eq else f"{s} (no eq; VULCAN-only)"
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
               title=title, xlabel="equilibrium bulk VMR", ylabel="offchem photosphere VMR")
    for ax in axes.ravel()[len(SPECIES):]:
        ax.set_visible(False)
    fig.suptitle("Equilibrium outgassing (bulk) vs VULCAN at the photosphere (p_obs = 0.02 bar)\n"
                 "(1:1 line = observable abundance matches the outgassed prediction)", fontsize=11)
    fig.savefig(OUT_DIR / "fig_eq_vs_offchem_photosphere.png")
    plt.close(fig)


def plot_profiles(df: pd.DataFrame, ncols: int = 4):
    """VMR(p) profiles for the 7 species, cases inside the measured radius band.

    Selection: retained atmospheres (P_surf > 1 bar) whose modelled transit
    radius R_obs lies within the measured L 98-59 c 1-sigma band
    (R_OBS_MEAS +/- R_OBS_MEAS_ERR). The observation pressure level (p_obs)
    is marked so the observable part of each profile is clear.
    """
    lo, hi = R_OBS_MEAS - R_OBS_MEAS_ERR, R_OBS_MEAS + R_OBS_MEAS_ERR
    sub = df[df["has_offchem"] & (df["P_surf_bar"] > 1.0)
             & df["R_obs_Rearth"].between(lo, hi)].copy()
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
        ax.set(xscale="log", yscale="log", xlim=(1e-10, 2.0),
               ylim=(float(np.nanmax(p_bar)) * 1.5, float(np.nanmin(p_bar)) * 0.6),
               title=(f"{cname}  R={row['R_obs_Rearth']:.3f} R$_\\oplus$\n"
                      f"P$_s$={row['P_surf_bar']:.0f} bar, "
                      f"T$_s$={row['T_surf_K']:.0f} K\n"
                      f"H={row['H_budget']:.0f} fO2={row['fO2_shift_IW']:+.0f} "
                      f"S={row['S_budget']:.0f} C={row['C_budget']:.0f}"))
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[:n]:
        ax.set_ylabel("Pressure [bar]")
        ax.set_xlabel("VMR")
    axes[0].legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.suptitle(
        "VULCAN post-processed profiles: cases matching measured radius "
        f"{R_OBS_MEAS}$\\pm${R_OBS_MEAS_ERR} R$_\\oplus$ "
        "(dotted line = observation level p_obs)", fontsize=11)
    fig.tight_layout(h_pad=0.3)
    fig.savefig(OUT_DIR / "fig_offchem_profiles.png", bbox_inches="tight")
    plt.close(fig)


# Marker-size mapping for VMR (log-scaled). VMR spans many decades, so marker
# area is linear in log10(VMR) over a fixed decade window shared across panels.
VMR_SIZE_LO, VMR_SIZE_HI = -12.0, 0.0  # log10(VMR) window mapped to [SMIN, SMAX]
SMIN, SMAX = 6.0, 340.0                # marker area (pt^2)


def _vmr_to_size(vmr: np.ndarray) -> np.ndarray:
    """Map VMR -> scatter marker area, linear in log10(VMR), clamped to window."""
    logv = np.log10(np.clip(vmr, VMR_FLOOR, None))
    frac = np.clip((logv - VMR_SIZE_LO) / (VMR_SIZE_HI - VMR_SIZE_LO), 0.0, 1.0)
    return SMIN + (SMAX - SMIN) * frac


def plot_grid_dependence(df: pd.DataFrame):
    """Offchem photosphere VMR across the redox / sulfur grid, for retained cases.

    x: fO2 shift [log10 dIW]; y: S/H mass ratio; marker area scales with the
    post-processed photosphere VMR (larger = more abundant). One panel per
    species; colour follows the PROTEUS species palette. Cases that share an
    (fO2, S/H) node (differing C or H budget) are x-jittered so they do not
    fully overplot.
    """
    sub = df[df["has_offchem"] & (df["P_surf_bar"] > 1.0)].copy()
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
    x_jit = sub.apply(_jit, axis=1).to_numpy(float)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for ax, s in zip(axes.ravel(), SPECIES[:6]):
        vmr = sub[f"oc_{s}_obs_vmr"].to_numpy(float)
        sizes = _vmr_to_size(vmr)
        ax.scatter(sub["fO2_shift_IW"].to_numpy(float) + x_jit,
                   sub["S_budget"].to_numpy(float),
                   s=sizes, c=SPECIES_COLOR[s], edgecolor="k", linewidth=0.3,
                   alpha=0.85, zorder=3)
        ax.set(title=s, xlabel="fO2 shift [log10 dIW]", ylabel="S/H mass ratio",
               xticks=[-3, -2, -1])
    # Size legend: representative VMR decades, shown as grey reference markers.
    ref_vmr = [1e-10, 1e-6, 1e-2]
    handles = [plt.scatter([], [], s=_vmr_to_size(np.array([v]))[0],
                           c="0.6", edgecolor="k", linewidth=0.3,
                           label=f"VMR=1e{int(round(np.log10(v)))}") for v in ref_vmr]
    axes.ravel()[0].legend(handles=handles, fontsize=8, loc="best",
                           title="marker size", labelspacing=1.1)
    fig.suptitle("Post-processed photosphere VMR (p_obs = 0.02 bar) across the "
                 "fO2 / sulfur grid\n(marker size = VMR)", fontsize=11)
    fig.savefig(OUT_DIR / "fig_grid_dependence.png")
    plt.close(fig)


# Photosphere abundance ratios of interest and the grid axes to correlate against.
RATIOS = [("CS2", "SO2"), ("SO2", "H2S"), ("H2S", "CS2")]
RATIO_COLOR = {("CS2", "SO2"): WONG["vermilion"],
               ("SO2", "H2S"): WONG["blue"],
               ("H2S", "CS2"): WONG["green"]}
PARAM_LABEL = {"H_budget": "H budget", "fO2_shift_IW": "fO2 shift [log10 dIW]",
               "S_budget": "S/H mass ratio", "C_budget": "C/H mass ratio"}
RATIO_CLIP = (1e-8, 1e8)  # display bounds; ratios span the VMR-floor dynamic range


def plot_ratio_correlations(df: pd.DataFrame):
    """Photosphere abundance ratios vs each grid parameter, for retained cases.

    Rows: the three ratios (CS2/SO2, SO2/H2S, H2S/CS2, all at p_obs = 0.02 bar).
    Columns: the four grid axes. y is log-scaled; ratios are clipped to
    RATIO_CLIP for display because the VMR floor (1e-30) makes raw ratios span
    ~60 decades when a species is absent. Cases whose modelled transit radius
    falls in the measured L 98-59 c 1-sigma band are drawn as stars; all other
    retained cases as circles.
    """
    sub = df[df["has_offchem"] & (df["P_surf_bar"] > 1.0)].copy()
    if sub.empty:
        return
    lo_r, hi_r = R_OBS_MEAS - R_OBS_MEAS_ERR, R_OBS_MEAS + R_OBS_MEAS_ERR
    match = (sub["has_helpfile"] & sub["R_obs_Rearth"].between(lo_r, hi_r)).to_numpy(bool)

    nrows, ncols = len(RATIOS), len(GRID_KEYS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows),
                             constrained_layout=True)
    for i, (num, den) in enumerate(RATIOS):
        n = np.clip(sub[f"oc_{num}_obs_vmr"].to_numpy(float), VMR_FLOOR, None)
        d = np.clip(sub[f"oc_{den}_obs_vmr"].to_numpy(float), VMR_FLOOR, None)
        ratio = np.clip(n / d, *RATIO_CLIP)
        color = RATIO_COLOR[(num, den)]
        for j, param in enumerate(GRID_KEYS):
            ax = axes[i, j]
            x = sub[param].to_numpy(float)
            # H budget is always log-scaled; other positive axes only if they
            # span more than a decade.
            pos = x[np.isfinite(x) & (x > 0)]
            use_logx = param == "H_budget" or (pos.size > 0 and pos.max() / pos.min() > 20)
            for msk, marker, mk_lw, mk_z in ((~match, "o", 0.3, 3), (match, "*", 0.6, 4)):
                if not msk.any():
                    continue
                ax.scatter(x[msk], ratio[msk], marker=marker,
                           s=(150 if marker == "*" else 34), c=color,
                           edgecolor="k", linewidth=mk_lw, zorder=mk_z,
                           alpha=0.85)
            ax.set(yscale="log", ylim=RATIO_CLIP)
            if use_logx:
                ax.set_xscale("log")
            if j == 0:
                ax.set_ylabel(f"{num}/{den} VMR")
            if i == nrows - 1:
                ax.set_xlabel(PARAM_LABEL[param])
    fig.suptitle("Photosphere abundance ratios (p_obs = 0.02 bar) vs grid parameters\n"
                 "(stars = cases matching the measured L 98-59 c radius)", fontsize=11)
    fig.savefig(OUT_DIR / "fig_ratio_correlations.png")
    plt.close(fig)


def plot_cs2(df: pd.DataFrame):
    """CS2 is VULCAN-only: photosphere-level VMR across the grid."""
    sub = df[df["has_offchem"] & (df["P_surf_bar"] > 1.0)].copy()
    if sub.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    c_vals = sorted(sub["C_budget"].dropna().unique())
    c_colors = {v: c for v, c in zip(c_vals, [WONG["blue"], WONG["vermilion"]])}
    for cv in c_vals:
        g = sub[sub["C_budget"] == cv]
        a1.scatter(g["fO2_shift_IW"].to_numpy(float),
                   np.clip(g["oc_CS2_obs_vmr"].to_numpy(float), VMR_FLOOR, None),
                   s=34, c=c_colors[cv], edgecolor="k", linewidth=0.3,
                   label=f"C/H={cv:.0f}")
    a1.set(yscale="log", xlabel="fO2 shift [log10 dIW]", ylabel="CS2 photosphere VMR",
           title="CS2 photosphere abundance vs redox", xticks=[-3, -2, -1])
    a1.legend(fontsize=9)
    # CS2 photosphere vs S_budget, coloured by fO2
    f_vals = sorted(sub["fO2_shift_IW"].dropna().unique())
    f_colors = {v: c for v, c in zip(f_vals, [WONG["blue"], WONG["green"], WONG["vermilion"]])}
    for fv in f_vals:
        g = sub[sub["fO2_shift_IW"] == fv]
        a2.scatter(g["S_budget"].to_numpy(float),
                   np.clip(g["oc_CS2_obs_vmr"].to_numpy(float), VMR_FLOOR, None),
                   s=34, c=f_colors[fv], edgecolor="k", linewidth=0.3,
                   label=f"fO2={fv:+.0f}")
    a2.set(yscale="log", xlabel="S/H mass ratio", ylabel="CS2 photosphere VMR",
           title="CS2 photosphere abundance vs sulfur budget")
    a2.legend(fontsize=9)
    fig.suptitle("CS2 at the photosphere (p_obs = 0.02 bar; produced only in VULCAN)",
                 fontsize=11)
    fig.savefig(OUT_DIR / "fig_cs2.png")
    plt.close(fig)


def plot_radius(df: pd.DataFrame):
    """Modelled transit radius vs measured L 98-59 c radius."""
    sub = df[df["has_helpfile"] & (df["P_surf_bar"] > 1.0)].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    x = np.arange(len(sub))
    r = sub["R_obs_Rearth"].to_numpy(float)
    ax.scatter(x, r, s=34, c=WONG["blue"], edgecolor="k", linewidth=0.3, zorder=3,
               label="modelled R (p_obs=0.02 bar)")
    ax.axhspan(R_OBS_MEAS - R_OBS_MEAS_ERR, R_OBS_MEAS + R_OBS_MEAS_ERR,
               color=WONG["orange"], alpha=0.25, zorder=1,
               label=f"measured {R_OBS_MEAS}$\\pm${R_OBS_MEAS_ERR} R$_\\oplus$")
    ax.axhline(R_OBS_MEAS, color=WONG["orange"], linewidth=1.0, zorder=2)
    ax.set(ylabel="planet radius [R$_\\oplus$]",
           title="Modelled transit radius vs measured L 98-59 c radius")
    ax.set_xticks(x)
    labels = [f"{c.replace('case_0000', '')}, IW={f:+.0f}, S/H={s:.0f}"
              for c, f, s in zip(sub["case"], sub["fO2_shift_IW"], sub["S_budget"])]
    ax.set_xticklabels(labels, fontsize=7, rotation=90, ha="right", rotation_mode="anchor")
    ax.legend(fontsize=9)
    fig.savefig(OUT_DIR / "fig_radius.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def print_report(df: pd.DataFrame):
    n = len(df)
    n_off = int(df["has_offchem"].sum())
    n_retained = int((df["has_offchem"] & (df["P_surf_bar"] > 1.0)).sum())
    print("=" * 78)
    print(f"L 98-59 c grid  ({GRID_DIR})")
    print("=" * 78)
    print(f"cases total ............ {n}")
    print(f"with offchem/vulcan.csv  {n_off}")
    print(f"  of which retained P_s>1 bar ... {n_retained}")
    print(f"status breakdown:\n{df['status'].value_counts().to_string()}")
    print("-" * 78)

    ret = df[df["has_offchem"] & (df["P_surf_bar"] > 1.0)].copy()
    if not ret.empty:
        print("Retained-atmosphere cases: photosphere VMR ranges "
              "(VULCAN, p_obs = 0.02 bar)")
        for s in SPECIES:
            col = f"oc_{s}_obs_vmr"
            v = ret[col].replace(0, np.nan).dropna()
            if v.empty:
                print(f"  {s:4s}: all zero / absent")
                continue
            print(f"  {s:4s}: min={v.min():.2e}  median={v.median():.2e}  max={v.max():.2e}")
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
            print(f"  {row['case']}  photosphere={row['oc_CS2_obs_vmr']:.2e}  "
                  f"(surface={row['oc_CS2_surf_vmr']:.2e}) "
                  f"(fO2={row['fO2_shift_IW']:+.0f} S={row['S_budget']:.0f} "
                  f"C={row['C_budget']:.0f} H={row['H_budget']:.0f})")
        print("-" * 78)
        r = ret["R_obs_Rearth"].dropna()
        print(f"Modelled transit radius: min={r.min():.3f} median={r.median():.3f} "
              f"max={r.max():.3f} R_earth")
        print(f"Measured L 98-59 c: R={R_OBS_MEAS}+/-{R_OBS_MEAS_ERR}, "
              f"M={M_OBS_MEAS}+/-{M_OBS_MEAS_ERR} M_earth (grid mass_tot=2.0)")
    print("=" * 78)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_summary()
    df.to_csv(OUT_DIR / "summary_grid.csv", index=False)
    print_report(df)
    plot_eq_vs_offchem_photosphere(df)
    plot_profiles(df)
    plot_grid_dependence(df)
    plot_ratio_correlations(df)
    plot_cs2(df)
    plot_radius(df)
    print(f"\nWrote summary_grid.csv and figures to:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
