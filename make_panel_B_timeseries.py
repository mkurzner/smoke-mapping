#!/usr/bin/env python3
"""
Panel B — Hourly PM2.5 emissions vs. hours since ignition (no title)

Reads a CSV of emissions and produces a simple time series figure:
  x-axis: hours since ignition
  y-axis: PM2.5 emissions (tons per hour)

Assumptions
-----------
- The CSV contains one emissions column (e.g., "PM2.5_emitted") and
  optionally a column for hours since ignition (e.g., "hour").
- If no hour column is present, we fallback to 0..N-1.
- Emissions units may be tons/hour or kg/hour; we can force or auto-detect.

Usage
-----
python make_panel_B_timeseries.py \
  --csv data/bluesky/.../csvs/fire_emissions.csv \
  --outdir figures \
  --units auto \
  --outfile panel_B_emissions_timeseries.png
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Candidate column names for robust CSV parsing ----
CAND_EMISS_COLS = [
    "PM2.5_emitted", "pm25_emitted", "pm2_5_emitted", "PM25_emitted",
    "pm25", "PM25", "emissions", "emissions_pm25", "pm25_kgph", "pm25_tph"
]
CAND_HOUR_COLS = ["hour", "hoursinceignition", "timestep", "tstep", "time", "h"]


def _find_emissions_col(df):
    """
    Return the name of a PM2.5 emissions column.

    Strategy:
      1) Exact matches against common variants.
      2) Fuzzy fallback: columns containing 'pm' + ('25' or '2.5') + 'emit/...' etc.
    """
    # exact pass
    for c in df.columns:
        key = c.strip().lower().replace(" ", "").replace("_", "")
        for want in CAND_EMISS_COLS:
            wantk = want.strip().lower().replace(" ", "").replace("_", "")
            if key == wantk:
                return c
    # fuzzy pass
    for c in df.columns:
        key = c.strip().lower()
        if "pm" in key and ("25" in key or "2.5" in key) and ("emit" in key or "emiss" in key or "tph" in key or "kg" in key):
            return c
    raise ValueError("Could not find a PM2.5 emissions column in the CSV.")


def _to_tons_per_hour(series, force_units):
    """
    Convert emissions series to tons per hour.

    Parameters
    ----------
    series : pd.Series
        Emissions values in kg/h or t/h (unknown if force_units='auto').
    force_units : {'kg','t','auto'}
        - 'kg': treat as kg/h and convert to t/h.
        - 't' : treat as tons/h (no change).
        - 'auto': guess by magnitude (simple heuristic).

    Returns
    -------
    tph : pd.Series
        Emissions in tons per hour.
    label : str
        A units label for the y-axis.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if force_units == "kg":
        return s / 1000.0, "t h$^{-1}$"
    if force_units == "t":
        return s, "t h$^{-1}$"
    # auto: crude heuristic—if typical values are large, assume kg/h
    p95 = np.nanpercentile(s, 95) if len(s) else 0.0
    if p95 > 50:
        return s / 1000.0, "t h$^{-1}$"
    return s, "t h$^{-1}$"


def main():
    ap = argparse.ArgumentParser(
        description="Make Panel B: hourly PM2.5 emissions (t/h) vs hours since ignition (no title)."
    )
    ap.add_argument("--csv", required=True, help="Path to fire_emissions.csv")
    ap.add_argument("--outdir", default="figures", help="Output directory")
    ap.add_argument("--units", choices=["kg", "t", "auto"], default="auto",
                    help="Units of emissions column. 'auto' tries to guess.")
    ap.add_argument("--outfile", default="panel_B_emissions_timeseries.png",
                    help="Output filename (PNG)")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.csv)

    # Identify columns
    ecol = _find_emissions_col(df)
    hcol = next((c for c in CAND_HOUR_COLS if c in df.columns), None)

    # Hours since ignition
    hours = (np.arange(len(df), dtype=int) if hcol is None
             else pd.to_numeric(df[hcol], errors="coerce").fillna(0).astype(int).values)

    # Convert emissions to tons/hour
    tph, units_label = _to_tons_per_hour(df[ecol], args.units)

    # Plot
    fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=args.dpi)
    ax.plot(hours, tph, lw=2, marker="o", ms=3)
    ax.set_xlabel("Hours since ignition", fontsize=14)
    ax.set_ylabel(f"PM$_{{2.5}}$ emissions ({units_label})", fontsize=14)
    ax.grid(True, ls=":", alpha=0.5)

    outpath = outdir / args.outfile
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {outpath}")


if __name__ == "__main__":
    main()
