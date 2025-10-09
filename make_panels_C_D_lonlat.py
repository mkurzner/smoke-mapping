#!/usr/bin/env python3
"""
Panels C & D — lon/lat plots using BlueSky/HYSPLIT concentration output

C: Peak 3-hour mean surface PM2.5 (linear color; white contours), with
   labeled markers (Ignition/Kelowna/Penticton) and directional edge arrows
   for Seattle and Calgary. Optional auto-zoom or manual bbox.

D: Exceedance-hours >= 25 µg m^-3 on Day 1 (linear color; white contours),
   with the same annotations.

Assumptions
-----------
- NetCDF contains a surface PM2.5 field (variable "PM25") with a time-like
  dimension (e.g., "time" or "TSTEP") and two horizontal dims.
- grid_info.json provides bbox = [lon_min, lat_min, lon_max, lat_max] or
  bbox.minx/miny/maxx/maxy.

Usage
-----
python make_panels_C_D_lonlat.py \
  --nc data/bluesky/.../hysplit_conc.nc \
  --gridinfo data/bluesky/.../grid_info.json \
  --outdir figures \
  [--autozoom] [--xmin -121 --xmax -118 --ymin 49 --ymax 50] \
  [--cmax 200] [--hours-vmax 12] [--dpi 200]
"""
import json
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import argparse


# ---- default markers (lon, lat) ----
POINTS = [
    ("Ignition",  -119.6700, 49.9350),
    ("Kelowna",   -119.4966, 49.8863),
    ("Penticton", -119.5937, 49.4991),
]

# Directional “off-map” references (lon, lat)
SEATTLE = ("Seattle", -122.3321, 47.6062)
CALGARY = ("Calgary", -114.0719, 51.0447)


# ---------------- helpers ----------------
def lonlat_from_gridinfo(grid_json_path, nx, ny):
    """Return (XX, YY) 2-D lon/lat grids from BlueSky grid_info.json."""
    with open(grid_json_path) as f:
        g = json.load(f)
    bbox = g.get("bbox", g)
    if isinstance(bbox, dict):
        lon_min, lon_max = float(bbox["minx"]), float(bbox["maxx"])
        lat_min, lat_max = float(bbox["miny"]), float(bbox["maxy"])
    else:
        lon_min, lat_min, lon_max, lat_max = map(float, bbox)
    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    return np.meshgrid(lons, lats)


def pick_pm25(ds):
    """Find PM2.5 concentration and return as (time, y, x)."""
    da = ds["PM25"]
    dims = list(da.dims)

    # time dim
    tdim = next((d for d in dims if d.lower() in ("tstep", "time")), None)
    if tdim is None:
        raise RuntimeError(f"Couldn't find time-like dimension among {dims}")

    # peel vertical dim if present; keep surface (index 0)
    spatial = [d for d in dims if d != tdim]
    vdim = next((d for d in spatial if d.lower() in ("lay", "level", "lev", "z", "height", "layers")), None)
    if vdim:
        da = da.isel({vdim: 0})
        spatial = [d for d in spatial if d != vdim]

    if len(spatial) != 2:
        raise RuntimeError(f"Expected 2 horizontal dims, got {spatial}")

    ydim, xdim = spatial
    return da.transpose(tdim, ydim, xdim), tdim, ydim, xdim


def add_points(ax):
    """Plot primary points with small white-outlined crimson markers and labels."""
    for name, lon, lat in POINTS:
        ax.scatter(lon, lat, s=55, c="crimson", edgecolors="white",
                   linewidths=1.2, zorder=6)
        ax.annotate(
            name, (lon, lat),
            xytext=(10, 10), textcoords="offset points",
            fontsize=12, color="white",
            bbox=dict(facecolor="black", alpha=0.4, pad=1.5, edgecolor="none"),
            zorder=7
        )


def add_edge_arrow(ax, label, lon, lat, xmin, xmax, ymin, ymax):
    """
    Place a label near the nearest figure edge with an arrow pointing toward the
    true (lon,lat) direction off-panel. Keeps text inside the figure.
    """
    pad = 0.15
    d = {
        "left":   abs(lon - xmin),
        "right":  abs(lon - xmax),
        "bottom": abs(lat - ymin),
        "top":    abs(lat - ymax),
    }
    side = min(d, key=d.get)

    if side == "left":
        x = xmin + pad; y = np.clip(lat, ymin + pad, ymax - pad)
        ax.annotate(label, (x, y), xytext=(x - 0.6, y),
                    textcoords="data", ha="left", va="center",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.5, edgecolor="none"),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="white"),
                    zorder=8)
    elif side == "right":
        x = xmax - pad; y = np.clip(lat, ymin + pad, ymax - pad)
        ax.annotate(label, (x, y), xytext=(x + 0.6, y),
                    textcoords="data", ha="right", va="center",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.5, edgecolor="none"),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="white"),
                    zorder=8)
    elif side == "bottom":
        x = np.clip(lon, xmin + pad, xmax - pad); y = ymin + pad
        ax.annotate(label, (x, y), xytext=(x, y - 0.6),
                    textcoords="data", ha="center", va="bottom",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.5, edgecolor="none"),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="white"),
                    zorder=8)
    else:  # top
        x = np.clip(lon, xmin + pad, xmax - pad); y = ymax - pad
        ax.annotate(label, (x, y), xytext=(x, y + 0.6),
                    textcoords="data", ha="center", va="top",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.5, edgecolor="none"),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="white"),
                    zorder=8)


def _autozoom_bbox(c3_np, XX, YY):
    """
    Compute a bounding box that tightly contains the plume (by dilating a small
    thresholded mask). Returns (xmin, xmax, ymin, ymax).
    """
    if np.any(c3_np > 0):
        thr = max(1.0, float(np.nanpercentile(c3_np[c3_np > 0], 5)))
        mask = binary_dilation(c3_np > thr, iterations=8)
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        XXs, YYs = XX[y0:y1+1, x0:x1+1], YY[y0:y1+1, x0:x1+1]
        return float(XXs.min()), float(XXs.max()), float(YYs.min()), float(YYs.max())
    # fallback to full frame
    return float(XX.min()), float(XX.max()), float(YY.min()), float(YY.max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc", required=True, help="Path to HYSPLIT concentration NetCDF")
    p.add_argument("--gridinfo", required=True, help="Path to BlueSky grid_info.json")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--autozoom", action="store_true", help="Auto-zoom to plume")
    p.add_argument("--xmin", type=float); p.add_argument("--xmax", type=float)
    p.add_argument("--ymin", type=float); p.add_argument("--ymax", type=float)
    p.add_argument("--cmax", type=float, default=200.0, help="Max color for Panel C")
    p.add_argument("--hours-vmax", type=float, default=12.0, help="Max color for Panel D")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    OUTDIR = Path(args.outdir); OUTDIR.mkdir(exist_ok=True)

    # Load data
    ds = xr.open_dataset(args.nc, decode_times=False)
    conc, tdim, ydim, xdim = pick_pm25(ds)

    # Panel C metric: peak 3-hr mean across time (2D)
    c3 = conc.rolling({tdim: 3}, min_periods=3).mean().max(tdim)
    c3_np = np.asarray(c3)

    # Panel D metric: exceedance-hours (>= 25 µg m^-3) across time
    hours = (conc >= 25.0).sum(tdim).astype("float32").values

    # Lon/lat grids
    ny, nx = c3_np.shape
    XX, YY = lonlat_from_gridinfo(args.gridinfo, nx, ny)

    # Determine plotting bbox
    if args.autozoom:
        xmin, xmax, ymin, ymax = _autozoom_bbox(c3_np, XX, YY)
    else:
        xmin = args.xmin if args.xmin is not None else float(XX.min())
        xmax = args.xmax if args.xmax is not None else float(XX.max())
        ymin = args.ymin if args.ymin is not None else float(YY.min())
        ymax = args.ymax if args.ymax is not None else float(YY.max())

    # Color/contour levels
    vmax_c = float(np.nanpercentile(c3_np, 99)) if np.isfinite(np.nanmax(c3_np)) else args.cmax
    vmax_c = max(args.cmax, vmax_c)  # ensure at least cmax
    levels_c = [5, 10, 25, 50, 100, 150, 200]
    levels_d = [1, 2, 3, 4, 6, 8, 12]

    # Slice arrays to bbox
    xmask = (XX[0, :] >= xmin) & (XX[0, :] <= xmax)
    ymask = (YY[:, 0] >= ymin) & (YY[:, 0] <= ymax)
    XXs, YYs = XX[ymask][:, xmask], YY[ymask][:, xmask]
    C3s = c3_np[ymask][:, xmask]
    Hs = hours[ymask][:, xmask]

    # ---------- Panel C ----------
    fig, ax = plt.subplots(figsize=(10, 7), dpi=args.dpi)
    pm = ax.pcolormesh(XXs, YYs, np.clip(C3s, 0, vmax_c),
                       shading="auto", cmap="viridis", vmin=0, vmax=vmax_c)
    cbar = fig.colorbar(pm, ax=ax)
    cbar.set_label(r"Peak 3-hr mean PM$_{2.5}$ ($\mu$g m$^{-3}$)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    cs = ax.contour(XXs, YYs, C3s, levels=levels_c, colors="w", linewidths=0.9)
    ax.clabel(cs, fmt="%d", inline=True, fontsize=12)

    add_points(ax)
    add_edge_arrow(ax, *SEATTLE, xmin, xmax, ymin, ymax)
    add_edge_arrow(ax, *CALGARY, xmin, xmax, ymin, ymax)

    ax.set_xlabel("Longitude", fontsize=14); ax.set_ylabel("Latitude", fontsize=14)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    c_path = OUTDIR / "panel_C_peak3hr_lonlat_linear.png"
    fig.tight_layout(); fig.savefig(c_path); plt.close(fig)

    # ---------- Panel D ----------
    fig, ax = plt.subplots(figsize=(10, 7), dpi=args.dpi)
    hm = ax.pcolormesh(XXs, YYs, Hs, shading="auto", cmap="viridis", vmin=0, vmax=args.hours_vmax)
    cbar = fig.colorbar(hm, ax=ax)
    cbar.set_label(r"Hours $\geq 25$ $\mu$g m$^{-3}$", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    cs2 = ax.contour(XXs, YYs, Hs, levels=levels_d, colors="w", linewidths=0.9)
    ax.clabel(cs2, fmt=lambda v: f"{int(round(v))} h", inline=True, fontsize=12)

    add_points(ax)
    add_edge_arrow(ax, *SEATTLE, xmin, xmax, ymin, ymax)
    add_edge_arrow(ax, *CALGARY, xmin, xmax, ymin, ymax)

    ax.set_xlabel("Longitude", fontsize=14); ax.set_ylabel("Latitude", fontsize=14)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    d_path = OUTDIR / "panel_D_exceedance_hours_lonlat_linear.png"
    fig.tight_layout(); fig.savefig(d_path); plt.close(fig)

    print("Wrote:")
    print(" ", c_path)
    print(" ", d_path)


if __name__ == "__main__":
    main()
