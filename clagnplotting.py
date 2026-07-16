# AGN Changing-Look Pipeline —with objects with all data already in CSV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# AGN Changing-Look Pipeline —with objects with all data already in CSV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astroquery.sdss import SDSS
from sparcl.client import SparclClient
from alerce.core import Alerce
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d, median_filter
from pathlib import Path
import time
import warnings
import io
import contextlib
import os, sys
import re


class _SuppressPartitionWarning:
    def write(self, msg):
        if "partition" not in msg and "MaskedArray" not in msg:
            sys.__stderr__.write(msg)
    def flush(self):
        sys.__stderr__.flush()

sys.stderr = _SuppressPartitionWarning()
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV          = "In_LSST_ZTF_SDSS_DESI.csv"   # RA | DEC | ztf_oid | lsst_oid | redshift
OUTPUT_PDF         = "agn_cl_results.pdf"    # All figures saved here
OUTPUT_CANDIDATE_PDF = "agn_cl_candidates.pdf"  # Only CLAGN/candidate figures
OUTPUT_CANDIDATE_CSV = "agn_cl_candidates.csv"  # Only CLAGN/candidate summary rows
SDSS_RADIUS_ARCSEC = 2.0

BAND_COLORS = {
    "u": "#56B4E9", "g": "#009E73", "r": "#D55E00",
    "i": "#E69F00", "z": "#CC79A7", "y": "#0072B2",
}
BAND_ORDER = ["u", "g", "r", "i", "z", "y"]

# ============================================================
# PYQSOFIT / QSOFITMORE INTEGRATION  (full replacement for the previous
# per-line custom Gaussian fitting)
# ============================================================
#
# qsofitmore (a maintained wrapper/fork of PyQSOFit -- Guo, Shen & Wang 2018;
# https://github.com/rudolffu/qsofitmore, built on legolason/PyQSOFit) fits
# the continuum (power law + optional host galaxy), an FeII pseudo-continuum
# template, and physically-tied emission-line complexes SIMULTANEOUSLY,
# rather than fitting one Gaussian per line in an isolated window. This
# directly addresses issues found while developing the custom fitter above:
#   - FeII emission blending under Hb/MgII/CIII] was not modelled at all
#     before; it was silently absorbed into each line's local "baseline".
#   - CIII]/[Si III]/Al III blending was previously forced into a single
#     narrow-line Gaussian with ad hoc sigma bounds. qsofitmore's default
#     line list fits CIII] as one broad complex with NO separate narrow
#     component -- consistent with what was found empirically here (CIII]'s
#     width is often not meaningfully narrower than the genuinely broad
#     lines), so that failure mode simply can't happen anymore.
#
# Install (not on PyPI -- installed directly from GitHub):
#   pip install lmfit uncertainties astropy
#   pip install git+https://github.com/rudolffu/qsofitmore.git
# Optional, only needed if PYQSOFIT_DEREDDEN is set True below:
#   pip install dustmaps
#   python -c "from dustmaps.config import config; config['data_dir']='./dustmaps_data'; import dustmaps.sfd; dustmaps.sfd.fetch()"

from qsofitmore import QSOFitNew
from astropy.table import Table as _AstropyTable

PYQSOFIT_WORKDIR        = "pyqsofit_runs"    # per-object scratch/output dir for qsofitmore's own fits tables + QA plots
PYQSOFIT_LINE_LIST_PATH = "qsopar_log.fits"  # generated once by build_qsopar_line_list() if missing
PYQSOFIT_DEREDDEN       = False   # True requires a dust map download -- see install note above
PYQSOFIT_INCLUDE_IRON   = True
PYQSOFIT_IRON_TEMPLATE  = "V09"   # Verner+2009 template, covers 2000-10000 A (wider than the BG92-VW01 default)
PYQSOFIT_DECOMP_HOST    = False   # host-galaxy PCA decomposition; usually negligible for luminous CL-AGN -- off by default for speed
PYQSOFIT_MC             = False   # True enables Monte-Carlo flux uncertainties via flux randomization (much slower: refits n_trails times per spectrum)
PYQSOFIT_N_TRAILS       = 20

# The standard PyQSOFit/qsofitmore default line-parameter list: tied
# broad+narrow Gaussian complexes for Ha/[NII]/[SII], Hb/[OIII], Hg, [OII],
# [NeV], MgII, CIII], and CIV. Embedded directly here (rather than requiring
# a clone of the qsofitmore repo just to get this file) so the pipeline is
# self-contained.
_QSOPAR_LINE_LIST_COLNAMES = ["lambda", "compname", "minwav", "maxwav", "linename", "ngauss",
                              "inisig", "minsig", "maxsig", "voff", "vindex", "windex", "findex", "fvalue"]
_QSOPAR_LINE_LIST_ROWS = [
    (6564.61, "Ha",   6400.0, 6800.0, "Ha_br",     3.0, 0.005, 0.003000, 0.0100, 0.0050,  0.0, 0.0, 0.0, 0.050),
    (6564.61, "Ha",   6400.0, 6800.0, "Ha_na",     1.0, 0.001, 0.000500, 0.0017, 0.0100,  1.0, 1.0, 0.0, 0.002),
    (6549.85, "Ha",   6400.0, 6800.0, "NII6549",   1.0, 0.001, 0.000230, 0.0017, 0.0050,  1.0, 1.0, 1.0, 0.001),
    (6585.28, "Ha",   6400.0, 6800.0, "NII6585",   1.0, 0.001, 0.000230, 0.0017, 0.0050,  1.0, 1.0, 1.0, 0.003),
    (6718.29, "Ha",   6400.0, 6800.0, "SII6718",   1.0, 0.001, 0.000230, 0.0017, 0.0050,  1.0, 1.0, 2.0, 0.001),
    (6732.67, "Ha",   6400.0, 6800.0, "SII6732",   1.0, 0.001, 0.000230, 0.0017, 0.0050,  1.0, 1.0, 2.0, 0.001),
    (4862.68, "Hb",   4640.0, 5100.0, "Hb_br",     3.0, 0.005, 0.003000, 0.0100, 0.0030,  0.0, 0.0, 0.0, 0.010),
    (4862.68, "Hb",   4640.0, 5100.0, "Hb_na",     1.0, 0.001, 0.000230, 0.0017, 0.0100,  1.0, 1.0, 0.0, 0.002),
    (4960.30, "Hb",   4640.0, 5100.0, "OIII4959",  1.0, 0.001, 0.000230, 0.0017, 0.0100,  1.0, 1.0, 0.0, 0.002),
    (5008.24, "Hb",   4640.0, 5100.0, "OIII5007",  1.0, 0.001, 0.000230, 0.0017, 0.0100,  1.0, 1.0, 0.0, 0.004),
    (4955.30, "Hb",   4640.0, 5100.0, "OIII4959w", 1.0, 0.001, 0.000230, 0.0017, 0.0100,  2.0, 2.0, 0.0, 0.001),
    (4995.24, "Hb",   4640.0, 5100.0, "OIII5007w", 1.0, 0.001, 0.000230, 0.0017, 0.0100,  2.0, 2.0, 0.0, 0.002),
    (4341.68, "Hg",   4250.0, 4440.0, "Hg_br",     1.0, 0.005, 0.004000, 0.0250, 0.0017,  0.0, 0.0, 0.0, 0.050),
    (4341.68, "Hg",   4250.0, 4440.0, "Hg_na",     1.0, 0.001, 0.000230, 0.0017, 0.0050,  1.0, 1.0, 0.0, 0.001),
    (3728.48, "OII",  3650.0, 3800.0, "OII3728",   1.0, 0.001, 0.000333, 0.0017, 0.0100,  1.0, 1.0, 0.0, 0.001),
    (3426.84, "NeV",  3380.0, 3480.0, "NeV3426",   1.0, 0.001, 0.000333, 0.0017, 0.0050,  0.0, 0.0, 0.0, 0.001),
    (2798.75, "MgII", 2700.0, 2900.0, "MgII_br",   2.0, 0.005, 0.004000, 0.0150, 0.0017,  0.0, 0.0, 0.0, 0.050),
    (2798.75, "MgII", 2700.0, 2900.0, "MgII_na",   1.0, 0.001, 0.000230, 0.0017, 0.0100,  0.0, 0.0, 0.0, 0.002),
    (1908.73, "CIII", 1700.0, 1970.0, "CIII_br",   2.0, 0.005, 0.004000, 0.0150, 0.0150, 99.0, 0.0, 0.0, 0.010),
    # Added narrow CIII] component (NOT in qsofitmore's stock line list).
    # vindex=windex=0 -- there is no established narrow-line group to tie it
    # to the way Hb_na/Ha_na are tied to [OIII] or [NII]/[SII], so this is an
    # untied free Gaussian, same class of caveat as MgII_na: it can end up
    # absorbing residual flux from the broad component rather than measuring
    # a genuinely independent narrow-line region. See QSOFIT_NARROW_PRIORITY
    # below for how this is used (and flagged) in classification.
    (1908.73, "CIII", 1700.0, 1970.0, "CIII_na",   1.0, 0.001, 0.000230, 0.0017, 0.0100,  0.0, 0.0, 0.0, 0.002),
    (1549.06, "CIV",  1500.0, 1700.0, "CIV_br",    3.0, 0.005, 0.004000, 0.0150, 0.0150,  0.0, 0.0, 0.0, 0.050),
]


def build_qsopar_line_list(path):
    """Write the qsofitmore line-parameter-list FITS table if it doesn't
    already exist at `path`. Safe to call before every fit; it's a no-op
    once the file has been written."""
    if os.path.exists(path):
        return
    df_lines = pd.DataFrame(_QSOPAR_LINE_LIST_ROWS, columns=_QSOPAR_LINE_LIST_COLNAMES)
    tbl = _AstropyTable.from_pandas(df_lines)
    tbl.write(path, format="fits", overwrite=True)
    print(f"  ✓ wrote qsofitmore line-parameter list to {path}")


def run_qsofitmore_epoch(wave_obs, flux_obs, err_obs, z, ra, dec, name, workdir):
    """
    Fit one epoch's spectrum with qsofitmore: continuum + FeII pseudo-
    continuum + tied emission-line complexes, all fit simultaneously.

    Expects OBSERVED-frame wavelength/flux/err in the native SDSS/DESI
    convention (flux and err in units of 1e-17 erg/s/cm^2/Angstrom --
    qsofitmore's required input unit, and also what SDSS/DESI spectra are
    natively stored in, so no unit conversion is needed here). qsofitmore
    does its own rest-frame conversion internally given `z`.

    Returns (result_row, qa_plot_jpg_path):
      - result_row: a plain dict of every column in q.result_table (line
        fluxes/areas, FWHM, sigma, EW, continuum/FeII parameters, etc.), or
        None if there wasn't enough data or the fit failed.
      - qa_plot_jpg_path: path to qsofitmore's own saved QA plot (continuum +
        FeII + line decomposition), or None if unavailable.
    """
    if wave_obs is None or flux_obs is None:
        return None, None
    if not (np.isfinite(z) and z > 0):
        return None, None

    # Same absorption-contamination cleaning used for our own Figure 1/2
    # plots: a cosmic-ray/bad-pixel trough would bias qsofitmore's
    # continuum/line fit exactly the way it biased the old custom fits.
    # (This also now patches genuine NaN flux pixels -- see the function's
    # docstring -- but err_obs can independently contain NaN/non-positive
    # values, e.g. DESI pixels with ivar<=0, so a final joint finite-value
    # filter below is still needed: lmfit cannot handle any NaN in its input.)
    wave_arr = np.asarray(wave_obs, dtype=float)
    flux_clean = clean_absorption_contamination(np.asarray(flux_obs, dtype=float))

    if err_obs is None:
        # No real per-pixel error available from the source (shouldn't happen
        # for SDSS/DESI now that both fetchers populate ivar-based sigma, but
        # kept as a safety net for any future spectrum source that lacks one).
        # A robust MAD-based scatter estimate is used as a uniform error
        # rather than silently dropping the whole epoch.
        robust_sigma = 1.4826 * np.nanmedian(np.abs(flux_clean - np.nanmedian(flux_clean)))
        err_arr = np.full_like(flux_clean, robust_sigma if np.isfinite(robust_sigma) and robust_sigma > 0 else 1.0)
        print(f"    ⚠ {name}: no per-pixel error array available -- using a uniform "
              f"robust-scatter estimate ({err_arr[0]:.3g}) instead of real errors")
    else:
        err_arr = np.asarray(err_obs, dtype=float)

    good = np.isfinite(wave_arr) & np.isfinite(flux_clean) & np.isfinite(err_arr) & (err_arr > 0)
    if good.sum() < 50:  # need enough pixels left for a meaningful fit
        print(f"    ⚠ qsofitmore skipped for {name}: only {good.sum()} finite pixels after cleaning")
        return None, None
    wave_arr, flux_clean, err_arr = wave_arr[good], flux_clean[good], err_arr[good]

    os.makedirs(workdir, exist_ok=True)
    build_qsopar_line_list(PYQSOFIT_LINE_LIST_PATH)

    try:
        q = QSOFitNew(
            lam=wave_arr,
            flux=flux_clean,
            err=err_arr,
            z=z,
            ra=ra if np.isfinite(ra) else 0.0,
            dec=dec if np.isfinite(dec) else 0.0,
            name=name, is_sdss=False, path=workdir,
        )
        q.Fit(
            name=name,
            deredden=PYQSOFIT_DEREDDEN,
            decomposition_host=PYQSOFIT_DECOMP_HOST,
            include_iron=PYQSOFIT_INCLUDE_IRON, iron_temp_name=PYQSOFIT_IRON_TEMPLATE,
            poly=False, broken_pl=True, BC=False,
            MC=PYQSOFIT_MC, n_trails=PYQSOFIT_N_TRAILS,
            linefit=True, tie_lambda=True, tie_width=True,
            tie_flux_1=True, tie_flux_2=True,
            save_result=True, plot_fig=True, save_fig=True,
            plot_line_name=True, plot_legend=True,
            line_list_path=PYQSOFIT_LINE_LIST_PATH,
            save_fits_path=workdir, save_fits_name=name,
        )
    except Exception as e:
        print(f"    ⚠ qsofitmore fit failed for {name}: {e}")
        return None, None

    row = {c: q.result_table[c][0] for c in q.result_table.colnames}
    jpg_path = os.path.join(workdir, f"plot_fit_{name}.jpg")
    return row, (jpg_path if os.path.exists(jpg_path) else None)


# Priority order for picking which broad/narrow complex to use as the
# classification anchor, mirroring the old LINE_FIT_CONFIG priority but keyed
# to qsofitmore's result_table column-name prefixes ("<prefix>_area").
QSOFIT_BROAD_PRIORITY = [
    ("Hβ",    "Hb_whole_br"),
    ("Hα",    "Ha_whole_br"),
    # C III] is intentionally NOT used as a broad-line anchor. At high redshift
    # prefer Mg II or C IV for broad-line variability, while allowing C III]
    # only as the fallback narrow-line anchor below.
    ("Mg II", "MgII_whole_br"),
    ("C IV",  "CIV_whole_br"),
]

QSOFIT_NARROW_PRIORITY = [
    ("[O III] 5007", "OIII5007"),
    ("[O III] 4959", "OIII4959"),
    ("Hα narrow",    "Ha_na"),
    # [Ne V] 3426 deliberately excluded from classification per explicit
    # request, despite being a physically clean AGN-only indicator (see
    # earlier discussion) -- it is NOT used as a narrow-line anchor here.
    # The NeV3426 component remains in the line list above purely so it
    # still shows up in qsofitmore's own QA plots for visual inspection;
    # it's just never picked up by get_common_flux_qsofit.
    # Deliberately NO Mg II narrow entry. In the embedded line list above,
    # MgII_na has vindex=0/windex=0 -- i.e. it is NOT tied to any independent
    # narrow-line-region group the way Hb_na/Ha_na are tied to [OIII] or
    # [NII]/[SII] (vindex=windex=1 there). MgII has no well-established
    # forbidden narrow counterpart the way Balmer lines do; "MgII_na" is
    # really just a free second Gaussian soaking up whatever the single
    # broad component doesn't fit well at the line core, not an independent
    # physical measurement. Using it as a narrow-line anchor risks pairing a
    # broad line against a piece of itself (MgII_br vs. MgII_na, both from
    # the SAME variable transition) rather than a genuine broad-region vs.
    # narrow-region contrast.
    #
    # C III] narrow IS included below, at the user's request, despite
    # carrying the same untied caveat (vindex=windex=0 -- see the CIII_na row
    # added to the line list above): there is no established narrow-line
    # group to tie it to either. It is ranked last and used only as a fallback
    # narrow-line anchor at redshifts where [OIII] and Hα have shifted out of
    # the optical window entirely.
    ("C III] narrow", "CIII_na"),
]

# Broad/narrow label pairs that must never be paired against each other
# because they come from decomposing the SAME transition with an untied
# (unconstrained) narrow component. C III] is no longer a broad-line anchor
# and Mg II narrow is excluded, so there are no current conflicts.
CONFLICTING_BROAD_NARROW_PAIRS = set()


def get_common_flux_qsofit(sdss_row, desi_row, priority, verbose=False, exclude_labels=None):
    """Return (label, sdss_flux, sdss_flux_err, desi_flux, desi_flux_err) for
    the first line/complex in `priority` whose '<key>_area' column is finite
    and positive in BOTH epochs' qsofitmore result rows.

    If verbose=True, prints the specific reason each candidate was skipped
    (column missing entirely -- e.g. that line wasn't in the fitted
    wavelength range for that epoch -- vs. present but non-positive/NaN in
    one or both epochs vs. present in only one epoch), so a "why didn't this
    pair up even though it looks like both lines are there" question can be
    answered directly from the printed output instead of guessed at.

    exclude_labels: an optional set of labels to skip outright if a future
    broad/narrow candidate pair is judged physically self-conflicting.
    """
    exclude_labels = exclude_labels or set()
    if sdss_row is None or desi_row is None:
        if verbose:
            print("    [flux-pairing] one or both epochs have no qsofitmore result at all")
        return None, np.nan, np.nan, np.nan, np.nan
    for label, key in priority:
        if label in exclude_labels:
            if verbose:
                print(f"    [flux-pairing] {label}: skipped -- would conflict with the "
                      f"already-chosen narrow/broad pairing from the same transition "
                      f"(falling through to the next candidate instead)")
            continue
        area_key = f"{key}_area"
        err_key  = f"{key}_area_err"
        fa = sdss_row.get(area_key)
        fb = desi_row.get(area_key)
        if fa is None or fb is None:
            if verbose:
                missing = []
                if fa is None: missing.append("SDSS")
                if fb is None: missing.append("DESI")
                print(f"    [flux-pairing] {label} ({area_key}): column missing in "
                      f"{' and '.join(missing)} -- line not in that epoch's fitted "
                      f"wavelength range (check qsofitmore's own QA plot: is this "
                      f"line actually inside the plotted rest-frame window for that "
                      f"epoch, or just outside the axis?)")
            continue
        if not (np.isfinite(fa) and np.isfinite(fb) and fa > 0 and fb > 0):
            if verbose:
                print(f"    [flux-pairing] {label} ({area_key}): present but not usable "
                      f"-- SDSS area={fa}, DESI area={fb} (need finite AND >0 in both; "
                      f"a fit that converged to ~0 or negative amplitude, e.g. the line "
                      f"wasn't actually detected above the continuum/FeII model, will "
                      f"show up here as a non-positive area even though the QA plot "
                      f"draws a curve for it)")
            continue
        ea = sdss_row.get(err_key, np.nan)
        eb = desi_row.get(err_key, np.nan)
        if verbose:
            print(f"    [flux-pairing] {label} ({area_key}): ACCEPTED -- SDSS={fa:.3g}, DESI={fb:.3g}")
        return label, fa, ea, fb, eb
    return None, np.nan, np.nan, np.nan, np.nan


# ============================================================
# AGN TYPE CLASSIFICATION  (Osterbrock 1977 / Zeltyn+2024)
# ============================================================

def classify_agn_type(broad_flux, narrow_flux):
    if not (np.isfinite(broad_flux) and np.isfinite(narrow_flux)) or narrow_flux == 0:
        return "unknown", np.nan
    ratio = broad_flux / narrow_flux
    if ratio > 5:
        agn_type = "1.0"
    elif ratio > 2:
        agn_type = "1.2"
    elif ratio > 0.33:
        agn_type = "1.5"
    elif ratio > 0:
        agn_type = "1.8"
    else:
        agn_type = "2.0"
    return agn_type, ratio


def classify_cl_event(sdss_type, desi_type, broad_pct_change=np.nan,
                      broad_threshold_pct=50):
    type_order = {"1.0": 0, "1.2": 1, "1.5": 2, "1.8": 3, "2.0": 4, "unknown": 5}
    s = type_order.get(sdss_type, 5)
    d = type_order.get(desi_type, 5)
    if s != 5 and d != 5:
        if s == d:
            return f"no type change (both Type {sdss_type})"
        direction = "turn-off (dimming)" if d > s else "turn-on (brightening)"
        return f"{direction}: Type {sdss_type} → Type {desi_type}"

    if np.isfinite(broad_pct_change):
        if abs(broad_pct_change) >= broad_threshold_pct:
            direction = "broad-line fading" if broad_pct_change < 0 else "broad-line brightening"
            return f"candidate CLAGN ({direction}; broad flux change {broad_pct_change:+.1f}%)"
        return f"broad-line variable but below CL threshold ({broad_pct_change:+.1f}%)"

    return "undetermined: no usable broad+narrow line-ratio pair"


def is_clagn_candidate(verdict):
    """Return True for objects that should go into the candidate-only outputs."""
    if not isinstance(verdict, str):
        return False
    verdict_l = verdict.lower()
    return (
        verdict_l.startswith("candidate clagn")
        or verdict_l.startswith("turn-on")
        or verdict_l.startswith("turn-off")
    )


# ============================================================
# FLUX / MAG HELPERS
# ============================================================

def flux_nJy_to_mag(flux_nJy):
    return 31.4 - 2.5 * np.log10(flux_nJy)


def normalise_mjd(df, col="mjd"):
    df = df.copy()
    if col in df.columns:
        mask = df[col] > 2_400_000
        df.loc[mask, col] = df.loc[mask, col] - 2_400_000.5
    return df


def mjd_to_date_str(mjd):
    if mjd is None or not np.isfinite(mjd):
        return "unknown date"
    return Time(mjd, format="mjd", scale="utc").iso[:10]


def sdss_mag_to_nJy(mag):
    return 10 ** ((8.9 - mag) / 2.5) * 1e9


def sdss_mag_err_to_nJy_err(mag_err, mag):
    flux_nJy = sdss_mag_to_nJy(mag)
    return mag_err * flux_nJy * np.log(10) / 2.5


def sdss_df_to_detections(df_raw):
    rows = []
    band_err_map = {"u": "Err_u", "g": "Err_g", "r": "Err_r",
                    "i": "Err_i", "z": "Err_z"}
    for _, row in df_raw.iterrows():
        for band, err_col in band_err_map.items():
            mag     = row.get(band, np.nan)
            mag_err = row.get(err_col, np.nan)
            if not np.isfinite(mag) or mag <= 0:
                continue
            rows.append({
                "oid":          row.get("objid", np.nan),
                "mjd":          row.get("mjd",   np.nan),
                "band":         band,
                "flux_nJy":     sdss_mag_to_nJy(mag),
                "flux_err_nJy": sdss_mag_err_to_nJy_err(mag_err, mag) if np.isfinite(mag_err) else np.nan,
                "mag":          mag,
                "mag_err":      mag_err,
            })
    return pd.DataFrame(rows)


def lsst_add_mag(df):
    df = df.copy()
    flux_col = ("scienceFlux"    if "scienceFlux"    in df.columns else
                "flux_nJy"       if "flux_nJy"       in df.columns else None)
    err_col  = ("scienceFluxErr" if "scienceFluxErr" in df.columns else
                "flux_err_nJy"   if "flux_err_nJy"   in df.columns else None)
    if flux_col is None:
        return df
    valid = df[flux_col] > 0
    df["mag"] = np.nan
    df.loc[valid, "mag"] = flux_nJy_to_mag(df.loc[valid, flux_col])
    if err_col and err_col in df.columns:
        df["mag_err"] = np.nan
        df.loc[valid, "mag_err"] = (
            2.5 / np.log(10) * df.loc[valid, err_col] / df.loc[valid, flux_col]
        )
    return df

def get_mag_ranges(ztf_df, lsst_lc, df_sdss):
    """Return per-band magnitude ranges as a dict, and print them."""
    ranges = {}
    print("  📊 Magnitude ranges:")

    for survey, df, bands, band_col_candidates in [
        ("ZTF",  ztf_df,  ["g", "r", "i"],  ["band"]),
        ("LSST", lsst_lc, BAND_ORDER,        ["band_name", "band"]),
        ("SDSS", df_sdss, ["u","g","r","i","z"], ["band"]),
    ]:
        if df is None or df.empty or "mag" not in df.columns:
            continue
        band_col = next((c for c in band_col_candidates if c in df.columns), None)
        if band_col is None:
            continue
        for band in bands:
            mags = df.loc[df[band_col] == band, "mag"].dropna()
            mags = mags[np.isfinite(mags)]
            if mags.empty:
                continue
            key   = f"{survey}_{band}"
            delta = mags.max() - mags.min()
            ranges[key] = {"min": mags.min(), "max": mags.max(),
                           "median": mags.median(), "N": len(mags),
                           "delta": delta}
            print(f"    {survey:4s} {band}: {mags.min():.2f} – {mags.max():.2f}  "
                  f"(median {mags.median():.2f}, N={len(mags)}, Δmag={delta:.2f})")

    if not ranges:
        print("    (no photometry available)")
    return ranges
# ============================================================
# ZTF PHOTOMETRY — fetched live via ALeRCE
# ============================================================

def fetch_ztf_photometry(ztf_id, alerce_client):
    """Fetch ZTF detections + forced photometry for a ZTF object ID."""
    try:
        df_det = alerce_client.query_detections(ztf_id, survey="ztf", format="pandas")
    except Exception as e:
        print(f"    ZTF detections fetch failed for {ztf_id}: {e}")
        df_det = pd.DataFrame()

    try:
        df_forced = alerce_client.query_forced_photometry(ztf_id, survey="ztf", format="pandas")
    except Exception as e:
        print(f"    ZTF forced phot fetch failed for {ztf_id}: {e}")
        df_forced = pd.DataFrame()

    frames = []
    for df, source in [(df_det, "det"), (df_forced, "forced")]:
        if df is None or df.empty:
            continue
        df = df.copy()
        df = normalise_mjd(df)

        # Standardise band column
        for cand in ("fid", "band", "filter", "filterid"):
            if cand in df.columns:
                df["band"] = df[cand].astype(str)
                break

        # Map ZTF fid integers (1=g, 2=r, 3=i) to letters if needed
        fid_map = {"1": "g", "2": "r", "3": "i"}
        df["band"] = df["band"].map(lambda x: fid_map.get(str(x), str(x)))

        # Standardise magnitude columns
        for mag_c in ("mag", "magpsf", "magap"):
            if mag_c in df.columns:
                df["mag"] = df[mag_c]
                break
        for err_c in ("e_mag", "sigmapsf", "sigmagap", "mag_err"):
            if err_c in df.columns:
                df["mag_err"] = df[err_c]
                break
        if "mag_err" not in df.columns:
            df["mag_err"] = np.nan

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Keep only finite, plausible magnitudes
    if "mag" in out.columns:
        out = out[out["mag"].notna() & np.isfinite(out["mag"]) &
                  (out["mag"] > 10) & (out["mag"] < 25)].copy()

    return out


# ============================================================
# LSST PHOTOMETRY
# ============================================================

def fetch_lsst(lsst_oid, alerce_client):
    try:
        df_dets   = alerce_client.query_detections(lsst_oid, survey="lsst", format="pandas")
        df_forced = alerce_client.query_forced_photometry(lsst_oid, survey="lsst", format="pandas")
    except Exception:
        return None, np.nan, np.nan

    if df_dets is None or df_dets.empty:
        return None, np.nan, np.nan

    mjd_now = Time.now().mjd
    df_dets = normalise_mjd(df_dets)
    df_dets = df_dets[df_dets["mjd"] <= mjd_now]

    if df_forced is not None and not df_forced.empty:
        df_forced = normalise_mjd(df_forced)
        df_forced = df_forced[df_forced["mjd"] <= mjd_now]
    else:
        df_forced = pd.DataFrame()

    lsst_all = pd.concat([df_dets, df_forced], ignore_index=True)
    lsst_all = lsst_add_mag(lsst_all)

    ra_obj  = df_dets["ra"].median()
    dec_obj = df_dets["dec"].median()
    return lsst_all, ra_obj, dec_obj


# ============================================================
# SDSS FETCHERS
# ============================================================

def fetch_sdss_photometry(ra, dec, radius=2.0, retries=2):
    pos = SkyCoord(ra, dec, unit="deg", frame="icrs")
    for attempt in range(retries):
        try:
            xid = SDSS.query_region(
                pos, radius=radius * u.arcsec,
                photoobj_fields=["objid", "ra", "dec", "mjd",
                                 "u", "g", "r", "i", "z",
                                 "Err_u", "Err_g", "Err_r", "Err_i", "Err_z"],
            )
            if xid is None or len(xid) == 0:
                return pd.DataFrame()
            return xid.to_pandas()
        except Exception:
            time.sleep(5)
    return pd.DataFrame()


def fetch_sdss_spectrum(ra, dec, radius=2.0):
    pos = SkyCoord(ra, dec, unit="deg")
    try:
        xid = SDSS.query_region(pos, radius=radius * u.arcsec, spectro=True)
    except Exception:
        return None, None, None, None

    if xid is None or len(xid) == 0:
        return None, None, None, None

    for i in range(len(xid)):
        try:
            sp = SDSS.get_spectra(matches=xid[i:i+1])
            if not sp:
                continue
            hdulist = sp[0]
            data    = hdulist[1].data
            wave    = 10 ** data["loglam"]
            flux    = data["flux"]
            # SDSS spectra carry inverse variance (not sigma) in the same
            # binary table, right alongside flux/loglam -- this was never
            # being read before, so every SDSS epoch got a `None` error
            # array and was silently skipped by anything downstream that
            # requires real errors (including qsofitmore). ivar<=0 pixels
            # (masked/bad) become NaN here; clean_absorption_contamination
            # and the finite-value filter in run_qsofitmore_epoch handle
            # those NaNs the same way DESI's bad pixels are handled.
            sigma = None
            if "ivar" in data.columns.names:
                ivar = np.asarray(data["ivar"], dtype=float)
                sigma = np.full_like(ivar, np.nan, dtype=float)
                good_ivar = ivar > 0
                sigma[good_ivar] = 1.0 / np.sqrt(ivar[good_ivar])
            mjd_obs = None
            try:
                mjd_obs = float(hdulist[0].header.get("MJD", None))
            except (TypeError, ValueError):
                pass
            return wave, flux, sigma, mjd_obs
        except Exception:
            continue

    return None, None, None, None


# ============================================================
# DESI FETCHER
# ============================================================

def desi_get_spectrum(sparcl_client, ra, dec, radius_arcsec=2.0):
    radius_deg = radius_arcsec / 3600.0
    try:
        res = sparcl_client.find(
            outfields=["sparcl_id", "targetid", "datasetgroup", "ra", "dec"],
            constraints={
                "ra":  [ra  - radius_deg, ra  + radius_deg],
                "dec": [dec - radius_deg, dec + radius_deg],
            },
            limit=10,
        )
        if len(res.ids) == 0:
            return None, None, None, None

        records = res.records
        seps = [
            np.sqrt(
                ((r["ra"]  - ra)  * np.cos(np.radians(dec))) ** 2 +
                 (r["dec"] - dec) ** 2
            )
            for r in records
        ]
        best   = records[int(np.argmin(seps))]
        bestid = res.ids[int(np.argmin(seps))]
        targetid = best.get("targetid", "unknown")
        print(f"    Closest SPARCL record: TARGETID={targetid}, "
              f"dataset={best.get('datasetgroup', '?')}, "
              f"sep={min(seps) * 3600:.2f}\"")

        spec = sparcl_client.retrieve(
            [bestid],
            include=["flux", "ivar", "wavelength", "wavemin", "wavemax", "wave_sigma"],
        )
        rec   = spec.records[0]
        flux  = np.array(rec["flux"])
        ivar  = np.array(rec["ivar"])
        sigma = np.full_like(ivar, np.nan, dtype=float)
        good_ivar = ivar > 0
        sigma[good_ivar] = 1.0 / np.sqrt(ivar[good_ivar])

        if "wavelength" in rec and rec["wavelength"] is not None:
            wave = np.array(rec["wavelength"])
        else:
            wave = np.arange(rec["wavemin"], rec["wavemax"],
                             rec["wave_sigma"])[: len(flux)]

        return wave, flux, sigma, targetid

    except Exception as e:
        print(f"    SPARCL retrieval failed: {e}")
        return None, None, None, None


# ============================================================
# SHARED Y-AXIS HELPER
# ============================================================

def shared_flux_ylim(flux_a, flux_b):
    a = flux_a[np.isfinite(flux_a)]
    b = flux_b[np.isfinite(flux_b)]
    if a.size == 0 or b.size == 0:
        return None, None
    ymin = min(a.min(), b.min())
    ymax = max(a.max(), b.max())
    pad  = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    return ymin - pad, ymax + pad


# ============================================================
# SPECTRAL SMOOTHING
# ============================================================

def smooth_spectrum(flux, sigma=1.0):
    """Gaussian-weighted smoothing for display. Pure numpy/scipy so it handles
    big-endian arrays returned by FITS/SPARCL without crashing.

    Replaces the previous boxcar (moving-average) smoothing: a boxcar weighs
    every point in the window equally, which smears/widens real line shapes
    more than necessary to knock down pixel-to-pixel noise. A Gaussian kernel
    weights nearby points much more heavily than distant ones, so it damps
    noise with a gentler effect on genuine line profiles. `sigma` is the
    Gaussian kernel width in pixels (not the emission-line sigma).
    """
    f = np.array(flux, dtype=np.float64)  # force native endian + copy
    # Forward-fill NaNs
    nans = np.isnan(f)
    idx  = np.where(~nans, np.arange(len(f)), 0)
    np.maximum.accumulate(idx, out=idx)
    f = f[idx]
    # Back-fill any leading NaNs that ffill couldn't reach
    still_nan = np.isnan(f)
    if still_nan.any():
        f[still_nan] = np.nanmedian(f)
    # mode="nearest" avoids the artificial edge droop that zero-padded
    # boxcar convolution ("same" mode) introduced at the spectrum edges.
    return gaussian_filter1d(f, sigma=sigma, mode="nearest")


def clean_absorption_contamination(flux, median_window=21, sigma_thresh=5.0):
    """
    Detect narrow absorption-like contamination (intervening absorption systems,
    bad pixels/columns, cosmic-ray-induced troughs) sitting on top of an emission
    line or continuum, and linearly interpolate across just those points. Also
    interpolates over any genuine NaN pixels (e.g. DESI pixels with ivar<=0),
    since those would otherwise pass straight through untouched -- previously
    they weren't included in the sigma-clipping check at all, so raw NaNs
    could reach downstream fitters (which choke on NaN input) unmodified.

    A wide running MEDIAN (not mean) is used as the local continuum estimate,
    since a median is naturally robust to narrow dips/spikes as long as they are
    much narrower than `median_window` -- unlike a boxcar mean, it won't get
    dragged down by the contamination itself. Points that fall `sigma_thresh`
    robust-sigmas (MAD-based) below that local median are flagged and replaced
    via linear interpolation from the surrounding clean points. Only downward
    excursions are treated as contamination -- real narrow emission bumps (e.g.
    a genuine narrow line riding on a broader one) are upward and are left alone.

    Returns a cleaned copy of `flux`; does not modify in place.
    """
    flux = np.asarray(flux, dtype=float)
    orig_nan = ~np.isfinite(flux)
    if len(flux) < median_window * 2:
        # Too short to run the median-filter continuum estimate -- but still
        # patch outright NaNs so a short/edge-case spectrum doesn't crash the
        # downstream fitter.
        if orig_nan.any() and (~orig_nan).sum() >= 2:
            idx = np.arange(len(flux))
            out = flux.copy()
            out[orig_nan] = np.interp(idx[orig_nan], idx[~orig_nan], flux[~orig_nan])
            return out
        return flux.copy()

    # Pre-fill NaNs (forward/back-fill, same approach as smooth_spectrum) purely
    # so the median-filter baseline and MAD scale aren't corrupted by NaN --
    # scipy's median_filter does not skip NaN values, so even one NaN inside a
    # window can poison that window's output.
    work = flux.copy()
    if orig_nan.any():
        idx_all = np.arange(len(work))
        good_idx = np.where(~orig_nan, idx_all, 0)
        np.maximum.accumulate(good_idx, out=good_idx)
        work = work[good_idx]
        still_bad = ~np.isfinite(work)
        if still_bad.any():
            work[still_bad] = np.nanmedian(flux[~orig_nan]) if (~orig_nan).any() else 0.0

    baseline = median_filter(work, size=median_window, mode="nearest")
    resid    = work - baseline
    finite   = np.isfinite(resid)
    if finite.sum() < median_window:
        contaminated = orig_nan.copy()
    else:
        mad   = np.nanmedian(np.abs(resid[finite] - np.nanmedian(resid[finite])))
        scale = 1.4826 * mad if mad > 0 else np.nanstd(resid[finite])
        dip_flagged = finite & (resid < -sigma_thresh * scale) if np.isfinite(scale) and scale != 0 else np.zeros_like(finite)
        contaminated = orig_nan | dip_flagged

    if not contaminated.any():
        return flux.copy()

    good = ~contaminated & np.isfinite(flux)
    if good.sum() < 2:
        return flux.copy()  # not enough clean points to interpolate from

    idx = np.arange(len(flux))
    cleaned = flux.copy()
    cleaned[contaminated] = np.interp(idx[contaminated], idx[good], flux[good])
    return cleaned


# ============================================================
# EMISSION-LINE MEASUREMENT  (trapz, pre-filter only)
# ============================================================

EMISSION_LINES = {
    "Hα":         {"wave": 6562.8, "half_win": 30,  "cont_off": 100},
    "Hβ":         {"wave": 4861.3, "half_win": 25,  "cont_off": 80},
    "[OIII]5007": {"wave": 5006.8, "half_win": 20,  "cont_off": 80},
    "[OIII]4959": {"wave": 4958.9, "half_win": 20,  "cont_off": 80},
    "MgII":       {"wave": 2798.0, "half_win": 40,  "cont_off": 120},
    "CIV":        {"wave": 1549.0, "half_win": 40,  "cont_off": 120},
    "[CIII]1909": {"wave": 1909.0, "half_win": 30,  "cont_off": 100},
}


def _local_continuum(wave, flux, line_wave, half_win, cont_off):
    lo_mask = (wave >= line_wave - cont_off - half_win) & (wave < line_wave - cont_off)
    hi_mask = (wave >  line_wave + cont_off)             & (wave <= line_wave + cont_off + half_win)
    mask    = (lo_mask | hi_mask) & np.isfinite(flux)
    if mask.sum() < 4:
        return np.full_like(wave, np.nanmedian(flux[np.isfinite(flux)]))
    coeffs = np.polyfit(wave[mask], flux[mask], 1)
    return np.polyval(coeffs, wave)


def measure_emission_lines(wave, flux, noise, label=""):
    wave  = np.asarray(wave, dtype=float)
    flux  = np.asarray(flux, dtype=float)
    if noise is not None:
        noise = np.asarray(noise, dtype=float)

    line_results = {}
    for name, cfg in EMISSION_LINES.items():
        lw       = cfg["wave"]
        hw       = cfg["half_win"]
        cont_off = cfg["cont_off"]

        if lw < wave.min() + cont_off or lw > wave.max() - cont_off:
            continue

        cont     = _local_continuum(wave, flux, lw, hw, cont_off)
        flux_sub = flux - cont

        win_mask = (wave >= lw - hw) & (wave <= lw + hw) & np.isfinite(flux_sub)
        if win_mask.sum() < 3:
            continue

        w_win  = wave[win_mask]
        f_win  = flux_sub[win_mask]
        dw     = np.gradient(w_win)
        f_line = float(np.trapezoid(f_win, w_win))

        if noise is not None and np.any(np.isfinite(noise[win_mask])):
            n_win      = noise[win_mask]
            mean_noise = np.nanmean(n_win[np.isfinite(n_win)])
            f_err      = mean_noise * np.sqrt(win_mask.sum()) * np.mean(dw)
        else:
            lo_mask2 = (wave >= lw - cont_off - hw) & (wave < lw - cont_off) & np.isfinite(flux_sub)
            hi_mask2 = (wave >  lw + cont_off)       & (wave <= lw + cont_off + hw) & np.isfinite(flux_sub)
            cont_residuals = flux_sub[lo_mask2 | hi_mask2]
            rms   = float(np.std(cont_residuals)) if len(cont_residuals) > 3 else np.nan
            f_err = rms * np.sqrt(win_mask.sum()) * np.mean(dw) if np.isfinite(rms) else np.nan

        snr      = abs(f_line / f_err) if (np.isfinite(f_err) and f_err > 0) else np.nan
        detected = bool(snr > 3) if np.isfinite(snr) else False

        cont_at_line = np.nanmedian(cont[win_mask])
        ew = float(np.trapezoid(f_win / cont_at_line, w_win)) if cont_at_line != 0 else np.nan

        line_results[name] = {
            "flux": f_line, "flux_err": f_err,
            "snr": snr, "ew": ew, "detected": detected,
        }

    return line_results


def compute_narrow_over_broad_ratio(sdss_lines, desi_lines,
                                    narrow_priority=("[OIII]5007", "[OIII]4959", "[CIII]1909"),
                                    broad_priority=("Hβ", "MgII", "CIV")):
    narrow_name = broad_name = None
    for n_name in narrow_priority:
        if not (n_name in sdss_lines and np.isfinite(sdss_lines[n_name]["flux"]) and
                n_name in desi_lines and np.isfinite(desi_lines[n_name]["flux"])):
            continue
        for b_name in broad_priority:
            if (b_name in sdss_lines and np.isfinite(sdss_lines[b_name]["flux"]) and
                b_name in desi_lines and np.isfinite(desi_lines[b_name]["flux"])):
                narrow_name, broad_name = n_name, b_name
                break
        if narrow_name is not None:
            break

    if narrow_name is None or broad_name is None:
        return np.nan, np.nan, None, None

    sdss_b = sdss_lines[broad_name]["flux"]
    desi_b = desi_lines[broad_name]["flux"]
    if sdss_b == 0 or desi_b == 0:
        return np.nan, np.nan, narrow_name, broad_name

    sdss_ratio = sdss_lines[narrow_name]["flux"] / sdss_b
    desi_ratio = desi_lines[narrow_name]["flux"] / desi_b
    return sdss_ratio, desi_ratio, narrow_name, broad_name


def _annotate_lines_on_axis(ax, line_results, wave, flux_sub_range,
                             color="limegreen", fontsize=7):
    ymin, ymax = ax.get_ylim()
    label_y    = ymin + 0.85 * (ymax - ymin)
    for name, res in line_results.items():
        if not res["detected"]:
            continue
        lw = EMISSION_LINES[name]["wave"]
        ax.axvline(lw, color=color, lw=0.8, linestyle="--", alpha=0.7)
        ax.text(lw + 5, label_y, name, color=color, fontsize=fontsize,
                rotation=90, va="top")


# ============================================================
# PLOTTING
# ============================================================

ZOOM_WINDOW = 30

FIT_LINES = {
    "Hβ":      4861.0,
    "[O III]": 5007.0,
    "Hα":      6563.0,
    "Mg II":   2798.0,
    "C III":  1909.0,
    "C IV":    1549.0,
}
FIT_COLORS = {
    "Hβ":      "purple",
    "[O III]": "green",
    "Hα":      "red",
    "Mg II":   "darkorange",
    "C III":  "steelblue",
    "C IV":    "brown",
}


def _to_rest(wave, flux, err, z):
    wave_r = wave / (1 + z)
    flux_r = flux * (1 + z)
    err_r  = err  * (1 + z) if err is not None else None
    return wave_r, flux_r, err_r


def _plot_zoom(ax, wave, flux, err, line_wave, color, title, smooth_sigma=1.0):
    mask = (wave > line_wave - ZOOM_WINDOW) & (wave < line_wave + ZOOM_WINDOW)
    if not np.any(mask):
        ax.set_title(f"{title}\n(no data)")
        return
    w = wave[mask]
    f = flux[mask]

    # Raw spectrum (faint)
    ax.plot(w, f, color=color, lw=0.6, alpha=0.35, zorder=1)

    # Error shading on raw
    if err is not None:
        e = err[mask]
        ax.fill_between(w, f - e, f + e, color=color, alpha=0.15, zorder=0)

    # Smoothed spectrum (bold, for visual comparison between epochs)
    f_smooth = smooth_spectrum(f, sigma=smooth_sigma)
    ax.plot(w, f_smooth, color=color, lw=1.8, alpha=0.95, zorder=2)

    ax.axvline(line_wave, color="k", ls="--", lw=0.8, zorder=3)
    ax.set_xlabel("Rest-frame wavelength [Å]")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_all(ztf_df, lsst_lc, sdss_spec, desi_spec, df_sdss,
             ztf_id, lsst_oid, sdss_date, ra, dec,
             sdss_lines=None, desi_lines=None, z=np.nan,
             pdf_writer=None, candidate_pdf_writer=None):
    """
    Figure 1 : ZTF + LSST light curve + full rest-frame spectra
    Figure 2 : 2×3 zoom grid on selected lines
    Figure 3 : qsofitmore per-epoch fits (continuum+FeII+lines) + CL verdict

    All figures are saved to pdf_writer (a PdfPages object) if provided.
    If candidate_pdf_writer is provided, CLAGN/candidate objects are also
    saved there after the final verdict is known.

    Returns a dict with the qsofitmore-based classification.
    """
    sdss_wave, sdss_flux, sdss_sigma, sdss_mjd = sdss_spec if sdss_spec[0] is not None else (None, None, None, None)
    desi_wave, desi_flux, desi_sigma, targetid  = desi_spec if desi_spec[0] is not None else (None, None, None, None)
    object_figs = []

    _empty = np.array([])
    if desi_wave is not None:
        if np.isfinite(z) and z > 0:
            dw_r, df_r, de_r = _to_rest(desi_wave, desi_flux, desi_sigma, z)
        else:
            dw_r, df_r, de_r = desi_wave, desi_flux, desi_sigma
        # Interpolate over narrow absorption-like contamination (intervening
        # absorbers, bad pixels/columns) before this spectrum is used anywhere
        # -- full-spectrum plot, zoom grid, or the emission-line Gaussian fits.
        df_r = clean_absorption_contamination(df_r)
    else:
        dw_r = df_r = de_r = None

    if sdss_wave is not None:
        if np.isfinite(z) and z > 0:
            sw_r, sf_r, se_r = _to_rest(sdss_wave, sdss_flux, sdss_sigma, z)
        else:
            sw_r, sf_r, se_r = sdss_wave, sdss_flux, sdss_sigma
        sf_r = clean_absorption_contamination(sf_r)
    else:
        sw_r = sf_r = se_r = None

    suptitle = f"LSST {lsst_oid} / ZTF {ztf_id}  (z = {z:.4f})"

    # ── Figure 1: light curves + full spectra ─────────────────────────
    fig1 = plt.figure(figsize=(16, 9))
    axs1 = fig1.subplot_mosaic([["lc", "lc"], ["desi", "sdss"]])
    ax_lc, ax_desi, ax_sdss = axs1["lc"], axs1["desi"], axs1["sdss"]

    # ZTF photometry
    if not ztf_df.empty and "band" in ztf_df.columns and "mag" in ztf_df.columns:
        for band in ["g", "r", "i"]:
            color = BAND_COLORS.get(band, "black")
            sub   = ztf_df[ztf_df["band"] == band]
            det   = sub[sub["mag"].notna() & np.isfinite(sub["mag"])]
            if not det.empty:
                yerr = det["mag_err"].values if "mag_err" in det.columns else None
                ax_lc.errorbar(det["mjd"], det["mag"],
                               yerr=np.clip(pd.Series(yerr).fillna(0.3), 0, 0.3) if yerr is not None else None,
                               fmt="o", color=color, label=f"ZTF {band}",
                               markersize=4, alpha=0.8)

    # LSST photometry
    band_col = ("band_name" if "band_name" in lsst_lc.columns else
                "band"      if "band"      in lsst_lc.columns else None)
    if band_col and "mag" in lsst_lc.columns:
        for band in BAND_ORDER:
            sub = lsst_lc[lsst_lc[band_col] == band].dropna(subset=["mag"])
            if sub.empty:
                continue
            yerr = sub["mag_err"].values if "mag_err" in sub.columns else None
            ax_lc.errorbar(sub["mjd"].values, sub["mag"].values, yerr=yerr,
                           fmt="s", color=BAND_COLORS.get(band, "black"),
                           label=f"LSST {band}", markersize=5,
                           markerfacecolor="none", markeredgewidth=1.2, alpha=0.9)

    ax_lc.invert_yaxis()
    ax_lc.set_title(f"ZTF (●) + LSST (■) — ZTF {ztf_id} / LSST OID {lsst_oid}", fontsize=11)
    ax_lc.set_xlabel("MJD")
    ax_lc.set_ylabel("AB Magnitude")
    ax_lc.legend(fontsize=7, ncol=5, loc="upper right")
    ax_lc.grid(alpha=0.3)

    if dw_r is not None:
        ax_desi.plot(dw_r, df_r, lw=0.8, color="purple")
        if de_r is not None:
            ax_desi.fill_between(dw_r, df_r - de_r, df_r + de_r, alpha=0.3, color="lavender")
        if sdss_lines:
            _annotate_lines_on_axis(ax_desi, sdss_lines, dw_r, None)
    else:
        ax_desi.text(0.5, 0.5, "No DESI spectrum", transform=ax_desi.transAxes,
                     ha="center", va="center", fontsize=12, color="gray")
    ax_desi.set_title(f"DESI (rest frame)  z={z:.4f}")
    ax_desi.set_xlabel("Rest-frame wavelength [Å]")
    ax_desi.set_ylabel(r"Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
    ax_desi.grid(alpha=0.3)

    if sw_r is not None:
        ax_sdss.plot(sw_r, sf_r, lw=0.8, color="green")
        if se_r is not None:
            ax_sdss.fill_between(sw_r, sf_r - se_r, sf_r + se_r, alpha=0.3, color="lightgreen")
        if desi_lines:
            _annotate_lines_on_axis(ax_sdss, desi_lines, sw_r, None, color="limegreen")
    else:
        ax_sdss.text(0.5, 0.5, "No SDSS spectrum", transform=ax_sdss.transAxes,
                     ha="center", va="center", fontsize=12, color="gray")
    ax_sdss.set_title(f"SDSS (rest frame)  Obs: {mjd_to_date_str(sdss_mjd)}")
    ax_sdss.set_xlabel("Rest-frame wavelength [Å]")
    ax_sdss.set_ylabel(r"Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
    ax_sdss.grid(alpha=0.3)

    if dw_r is not None and sw_r is not None:
        ymin, ymax = shared_flux_ylim(df_r, sf_r)
        if ymin is not None:
            ax_desi.set_ylim(ymin, ymax)
            ax_sdss.set_ylim(ymin, ymax)

    fig1.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    if pdf_writer is not None:
        pdf_writer.savefig(fig1)
    object_figs.append(fig1)

    # ── Figure 2: zoom grid ────────────────────────────────────────────
    ALL_ZOOM_LINES = [
        ("Hβ",      4861.0),
        ("[O III]", 5007.0),
        ("Hα",      6563.0),
        ("Mg II",   2798.0),
        ("C III",  1909.0),
        ("C IV",    1549.0),
    ]

    def has_data(wave, lw, window=30):
        mask = (wave > lw - window) & (wave < lw + window)
        return np.any(mask) and np.any(np.isfinite(wave[mask]))

    both_lines   = [(n, w) for n, w in ALL_ZOOM_LINES
                    if dw_r is not None and sw_r is not None
                    and has_data(dw_r, w) and has_data(sw_r, w)]
    either_lines = [(n, w) for n, w in ALL_ZOOM_LINES
                    if (n, w) not in both_lines and
                       ((dw_r is not None and has_data(dw_r, w)) or
                        (sw_r is not None and has_data(sw_r, w)))]
    zoom_lines   = (both_lines + either_lines)[:3]
    if not zoom_lines:
        zoom_lines = [("Mg II", 2798.0), ("C III", 1909.0), ("C IV", 1549.0)]

    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 7), sharey="row")
    _dw = dw_r if dw_r is not None else np.array([])
    _df = df_r if df_r is not None else np.array([])
    _sw = sw_r if sw_r is not None else np.array([])
    _sf = sf_r if sf_r is not None else np.array([])
    for j, (name, lwave) in enumerate(zoom_lines):
        _plot_zoom(axs2[0, j], _dw, _df, de_r, lwave, "purple", f"DESI {name}")
        _plot_zoom(axs2[1, j], _sw, _sf, se_r, lwave, "green",  f"SDSS {name}")
    axs2[0, 0].set_ylabel("Flux")
    axs2[1, 0].set_ylabel("Flux")
    fig2.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    if pdf_writer is not None:
        pdf_writer.savefig(fig2)
    object_figs.append(fig2)

    # ── Figure 3: qsofitmore per-epoch fits + classification ───────────
    # Full replacement for the previous per-line custom Gaussian fitting.
    # qsofitmore fits continuum + FeII pseudo-continuum + tied emission-line
    # complexes SIMULTANEOUSLY per epoch (see the PYQSOFIT_* config block
    # near the top of this file for why that matters). Fed the OBSERVED-frame
    # spectra directly (qsofitmore does its own rest-frame conversion given z).
    safe_id  = re.sub(r"[^\w.-]", "_", f"{lsst_oid}_{ztf_id}") or "obj"
    workdir  = os.path.join(PYQSOFIT_WORKDIR, safe_id)

    desi_row, desi_jpg = run_qsofitmore_epoch(
        desi_wave, desi_flux, desi_sigma, z, ra, dec, name=f"{safe_id}_DESI", workdir=workdir)
    sdss_row, sdss_jpg = run_qsofitmore_epoch(
        sdss_wave, sdss_flux, sdss_sigma, z, ra, dec, name=f"{safe_id}_SDSS", workdir=workdir)

    broad_name = narrow_name = None
    sdss_broad = desi_broad = sdss_narrow = desi_narrow = np.nan
    if sdss_row is not None and desi_row is not None:
        # Narrow is picked first, since it doesn't depend on which broad line
        # ends up chosen. Broad is picked second from the preferred broad-line
        # anchors (Balmer first when available, then Mg II, then C IV).
        print("  [qsofitmore] narrow-line candidates:")
        narrow_name, sdss_narrow, _sdss_narrow_err, desi_narrow, _desi_narrow_err = \
            get_common_flux_qsofit(sdss_row, desi_row, QSOFIT_NARROW_PRIORITY, verbose=True)

        excluded_broad = {b for (b, n) in CONFLICTING_BROAD_NARROW_PAIRS if n == narrow_name}
        print("  [qsofitmore] broad-line candidates:" +
              (f"  (excluding {sorted(excluded_broad)} -- conflicts with narrow choice {narrow_name!r})"
               if excluded_broad else ""))
        broad_name,  sdss_broad,  _sdss_broad_err,  desi_broad,  _desi_broad_err  = \
            get_common_flux_qsofit(sdss_row, desi_row, QSOFIT_BROAD_PRIORITY, verbose=True,
                                    exclude_labels=excluded_broad)

    sdss_type, sdss_gauss_ratio = (
        classify_agn_type(sdss_broad, sdss_narrow)
        if np.isfinite(sdss_broad) and np.isfinite(sdss_narrow) and sdss_narrow != 0
        else ("unknown", np.nan)
    )
    desi_type, desi_gauss_ratio = (
        classify_agn_type(desi_broad, desi_narrow)
        if np.isfinite(desi_broad) and np.isfinite(desi_narrow) and desi_narrow != 0
        else ("unknown", np.nan)
    )

    line_pair_label = f"{broad_name or 'n/a'}/{narrow_name or 'n/a'}"

    broad_pct_change = np.nan
    if broad_name and np.isfinite(sdss_broad) and np.isfinite(desi_broad) and sdss_broad != 0:
        broad_pct_change = (desi_broad - sdss_broad) / sdss_broad * 100

    cl_verdict      = classify_cl_event(sdss_type, desi_type, broad_pct_change)

    print(f"  [qsofitmore] Line pair used for BOTH epochs: {line_pair_label}")
    print(f"  [qsofitmore] SDSS type: {sdss_type}" +
          (f"  (ratio={sdss_gauss_ratio:.3f})" if np.isfinite(sdss_gauss_ratio) else ""))
    print(f"  [qsofitmore] DESI type: {desi_type}" +
          (f"  (ratio={desi_gauss_ratio:.3f})" if np.isfinite(desi_gauss_ratio) else ""))
    print(f"  [qsofitmore] CL verdict: {cl_verdict}")
    if np.isfinite(broad_pct_change):
        print(f"  [qsofitmore] Broad {broad_name} flux change (SDSS→DESI): {broad_pct_change:+.1f}%")

    # Rather than re-deriving a fit-quality plot ourselves, embed qsofitmore's
    # own QA plots (continuum + FeII + full line decomposition) side by side,
    # with the classification verdict annotated on top.
    fig3, (ax_d, ax_s) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, jpg_path, label in [(ax_d, desi_jpg, "DESI"), (ax_s, sdss_jpg, "SDSS")]:
        if jpg_path and os.path.exists(jpg_path):
            ax.imshow(plt.imread(jpg_path))
        else:
            ax.text(0.5, 0.5, f"No {label} qsofitmore fit", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
        ax.set_title(f"{label} qsofitmore fit")
        ax.axis("off")

    for ax, b_flux, n_flux in [(ax_d, desi_broad, desi_narrow), (ax_s, sdss_broad, sdss_narrow)]:
        if np.isfinite(b_flux) and np.isfinite(n_flux) and n_flux != 0:
            ratio = b_flux / n_flux
            agn_type, _ = classify_agn_type(b_flux, n_flux)
            ann = f"{broad_name}/{narrow_name} = {ratio:.3f}\nType {agn_type}"
        elif np.isfinite(b_flux):
            ann = f"{broad_name} detected\n(no narrow-line anchor)"
        elif np.isfinite(n_flux):
            ann = f"{narrow_name} only\n(no broad-line detection)"
        else:
            ann = "No common lines fit"
        ax.text(0.02, 0.02, ann, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    fig3.suptitle(suptitle + f"\n{cl_verdict}  (lines: {line_pair_label})", fontsize=11)
    plt.tight_layout()
    if pdf_writer is not None:
        pdf_writer.savefig(fig3)
    object_figs.append(fig3)

    if candidate_pdf_writer is not None and is_clagn_candidate(cl_verdict):
        for fig in object_figs:
            candidate_pdf_writer.savefig(fig)

    plt.show()
    for fig in object_figs:
        plt.close(fig)

    return {
        "broad_line":       broad_name,
        "narrow_line":      narrow_name,
        "sdss_type":        sdss_type,
        "desi_type":        desi_type,
        "sdss_gauss_ratio": sdss_gauss_ratio,
        "desi_gauss_ratio": desi_gauss_ratio,
        "cl_verdict":       cl_verdict,
        "broad_pct_change": broad_pct_change,
    }


# ============================================================
# SPARCL CLIENT FACTORY
# ============================================================

def make_sparcl_client(retries=3, backoff_sec=5):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return SparclClient(connect_timeout=10, read_timeout=60,
                                announcement=False)
        except Exception as e:
            last_err = e
            print(f"  SparclClient init failed (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(backoff_sec)
    raise RuntimeError(
        "Could not initialize SparclClient after several attempts."
    ) from last_err


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ── Load input table ──────────────────────────────────────────────
    df_input = pd.read_csv(INPUT_CSV, dtype={"lsst_oid": "string", "ztf_oid": "string"}, skipinitialspace=True)
    if not {"RA", "DEC"}.issubset(df_input.columns):
        raise ValueError(f"{INPUT_CSV} must have at least RA and DEC columns")

    alerce = Alerce()
    sparcl = make_sparcl_client()

    all_objects = []

    # ── Open PDF writer for all figures ───────────────────────────────
    with PdfPages(OUTPUT_PDF) as pdf, PdfPages(OUTPUT_CANDIDATE_PDF) as candidate_pdf:

        # Add PDF metadata
        pdf_info = pdf.infodict()
        pdf_info["Title"]   = "AGN Changing-Look Pipeline Results"
        pdf_info["Subject"] = f"CL-AGN analysis from {INPUT_CSV}"

        candidate_pdf_info = candidate_pdf.infodict()
        candidate_pdf_info["Title"]   = "AGN Changing-Look Candidate Results"
        candidate_pdf_info["Subject"] = f"CL-AGN candidates from {INPUT_CSV}"

        for row_i, row in df_input.iterrows():
            ra_csv  = float(row["RA"])
            dec_csv = float(row["DEC"])

            ztf_id   = str(row["ztf_oid"]).strip() if pd.notna(row.get("ztf_oid")) else None
            lsst_raw = row.get("lsst_oid")
            lsst_oid = str(lsst_raw).strip() if pd.notna(lsst_raw) and str(lsst_raw).strip() else None
            z        = float(row["redshift"])        if pd.notna(row.get("redshift")) else np.nan

            print(f"\n{'='*60}")
            print(f"Row {row_i+1} | RA={ra_csv:.5f}, Dec={dec_csv:.5f}"
                  f"  ZTF={ztf_id or '—'}  LSST={lsst_oid or '—'}"
                  f"  z={z:.4f}" if np.isfinite(z) else
                  f"\n{'='*60}\nRow {row_i+1} | RA={ra_csv:.5f}, Dec={dec_csv:.5f}"
                  f"  ZTF={ztf_id or '—'}  LSST={lsst_oid or '—'}  z=unknown")

            # ── Position: prefer LSST median, fall back to CSV coords ─
            ra_obj, dec_obj = ra_csv, dec_csv
            obj_entry = {"ra": ra_csv, "dec": dec_csv,
                         "lsst_oid": lsst_oid, "ztf_id": ztf_id, "z": z}

            # ── LSST photometry ───────────────────────────────────────
            lsst_lc = pd.DataFrame()
            if lsst_oid is not None:
                lsst_lc, ra_lsst, dec_lsst = fetch_lsst(lsst_oid, alerce)
                if lsst_lc is None or lsst_lc.empty:
                    lsst_lc = pd.DataFrame()
                    print(f"  ⚠ No LSST data for OID {lsst_oid}")
                else:
                    ra_obj, dec_obj = ra_lsst, dec_lsst
                    print(f"  ✓ LSST: {len(lsst_lc)} detections  (pos refined to {ra_obj:.5f}, {dec_obj:.5f})")
            else:
                print("  — LSST OID not provided")

            # ── ZTF photometry ────────────────────────────────────────
            ztf_df = pd.DataFrame()
            if ztf_id is not None:
                ztf_df = fetch_ztf_photometry(ztf_id, alerce)
                if ztf_df.empty:
                    print(f"  ⚠ No ZTF photometry for {ztf_id}")
                else:
                    print(f"  ✓ ZTF: {len(ztf_df)} detections")
            else:
                print("  — ZTF OID not provided")

            # ── SDSS photometry ───────────────────────────────────────
            df_sdss = pd.DataFrame()
            df_sdss_raw = fetch_sdss_photometry(ra_obj, dec_obj, radius=SDSS_RADIUS_ARCSEC)
            if df_sdss_raw is None or df_sdss_raw.empty:
                print("  ⚠ No SDSS photometry found")
            else:
                df_sdss = sdss_df_to_detections(df_sdss_raw)
                if df_sdss.empty:
                    print("  ⚠ SDSS photometry empty after conversion")
                else:
                    print(f"  ✓ SDSS photometry: {len(df_sdss)} band-epochs")
            # ── Mag ranges ─────────────────────────────────────────────
            obj_entry["mag_ranges"] = get_mag_ranges(ztf_df, lsst_lc, df_sdss)

            # ── SDSS spectrum ─────────────────────────────────────────
            sdss_spec = fetch_sdss_spectrum(ra_obj, dec_obj, radius=SDSS_RADIUS_ARCSEC)
            if sdss_spec[0] is None:
                print("  ⚠ No SDSS spectrum found")
            else:
                print(f"  ✓ SDSS spectrum: obs {mjd_to_date_str(sdss_spec[3])}")

            # ── DESI spectrum ─────────────────────────────────────────
            desi_spec = desi_get_spectrum(sparcl, ra_obj, dec_obj)
            if desi_spec[0] is None:
                print("  ⚠ No DESI spectrum found")
            else:
                print(f"  ✓ DESI spectrum: TARGETID={desi_spec[3]}")

            # ── Determine what we can plot ────────────────────────────
            has_lc      = not lsst_lc.empty or not ztf_df.empty
            has_spectra = sdss_spec[0] is not None and desi_spec[0] is not None

            if not has_lc and not has_spectra:
                print("  ✗ No data at all for this object — skipping plots")
                all_objects.append({"ra": ra_csv, "dec": dec_csv,
                                     "lsst_oid": lsst_oid, "ztf_id": ztf_id, "z": z,
                                     "cl_verdict": "no data"})
                continue

            print(f"  Redshift: z = {z:.4f}" if np.isfinite(z) else "  Redshift: not available")
            print("✅ Plotting")

            fit_summary = plot_all(
                ztf_df, lsst_lc, sdss_spec, desi_spec, df_sdss,
                ztf_id or "—", lsst_oid or "—",
                sdss_date=mjd_to_date_str(sdss_spec[3]) if sdss_spec[0] is not None else "n/a",
                ra=ra_obj, dec=dec_obj,
                z=z,
                pdf_writer=pdf,   # <-- pass the open PdfPages object
                candidate_pdf_writer=candidate_pdf,
            )
            if fit_summary:
                obj_entry.update(fit_summary)
            all_objects.append(obj_entry)

    candidate_objects = [
        obj for obj in all_objects
        if is_clagn_candidate(obj.get("cl_verdict", ""))
    ]

    candidate_rows = []
    for obj in candidate_objects:
        mag_ranges = obj.get("mag_ranges", {})
        mag_summary = ""
        if mag_ranges:
            mag_summary = " | ".join(
                f"{k}: min={v['min']:.3f}, max={v['max']:.3f}, "
                f"median={v['median']:.3f}, N={v['N']}, delta={v['delta']:.3f}"
                for k, v in mag_ranges.items()
            )
        candidate_rows.append({
            "ra": obj.get("ra", np.nan),
            "dec": obj.get("dec", np.nan),
            "lsst_oid": obj.get("lsst_oid"),
            "ztf_id": obj.get("ztf_id"),
            "redshift": obj.get("z", np.nan),
            "broad_line": obj.get("broad_line"),
            "narrow_line": obj.get("narrow_line"),
            "sdss_type": obj.get("sdss_type"),
            "desi_type": obj.get("desi_type"),
            "sdss_gauss_ratio": obj.get("sdss_gauss_ratio", np.nan),
            "desi_gauss_ratio": obj.get("desi_gauss_ratio", np.nan),
            "broad_pct_change": obj.get("broad_pct_change", np.nan),
            "cl_verdict": obj.get("cl_verdict"),
            "mag_ranges": mag_summary,
        })

    candidate_columns = [
        "ra", "dec", "lsst_oid", "ztf_id", "redshift",
        "broad_line", "narrow_line", "sdss_type", "desi_type",
        "sdss_gauss_ratio", "desi_gauss_ratio", "broad_pct_change",
        "cl_verdict", "mag_ranges",
    ]
    pd.DataFrame(candidate_rows, columns=candidate_columns).to_csv(
        OUTPUT_CANDIDATE_CSV, index=False
    )

    print(f"\nAll figures saved to: {OUTPUT_PDF}")
    print(f"Candidate-only figures saved to: {OUTPUT_CANDIDATE_PDF}")
    print(f"Candidate CSV saved to: {OUTPUT_CANDIDATE_CSV}  ({len(candidate_objects)} rows)")

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DONE.  Rows processed: {len(all_objects)}")
    print()

    for obj in all_objects:
        z_str   = "n/a" if not np.isfinite(obj["z"]) else f"{obj['z']:.4f}"
        verdict = obj.get("cl_verdict", "no spectra")
        pct     = obj.get("broad_pct_change", np.nan)
        pct_str = "n/a" if not (isinstance(pct, float) and np.isfinite(pct)) else f"{pct:+.1f}%"

        print(f"  RA={obj['ra']:.4f} Dec={obj['dec']:.4f}"
              f"  LSST={obj['lsst_oid'] or '—'}  ZTF={obj['ztf_id'] or '—'}"
              f"  |  lines={obj.get('broad_line','n/a')}/{obj.get('narrow_line','n/a')}"
              f"  |  verdict: {verdict}"
              f"  |  broad Δflux={pct_str}  z={z_str}")

        mag_ranges = obj.get("mag_ranges", {})
        if mag_ranges:
            parts = [f"{k}: {v['min']:.2f}–{v['max']:.2f} (med {v['median']:.2f}, N={v['N']}, Δmag={v['delta']:.2f})"
                     for k, v in mag_ranges.items()]
            print("    mags: " + "  |  ".join(parts))
        else:
            print("    mags: (none)")
        print()
