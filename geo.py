"""IZV project 2025 – Task 1 (variant Nehody).

This module provides utilities for creating a GeoDataFrame from accident/location
DataFrames and for generating two required geographic visualizations.

The evaluation imports this file as a library. Any side effects (I/O, plotting)
MUST be guarded by ``if __name__ == '__main__':``.

Author: (fill in if needed)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import MultiPoint
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing required dependencies for geo tasks. Install geopandas + shapely."
    ) from exc

try:
    import contextily as ctx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing required dependency 'contextily' for basemap tiles."
    ) from exc

try:
    from sklearn.cluster import DBSCAN
except ImportError as exc:  # pragma: no cover
    raise ImportError("Missing required dependency 'scikit-learn'.") from exc


# ----------------------------
# Configuration (chosen region)
# ----------------------------
# We work with Jihomoravský kraj (JHM). In the provided datasets, it is typically
# encoded either as the string code "JHM" (column 'region') or as integer 16
# (column 'p4a').
SELECTED_REGION_CODE_STR = "JHM"
SELECTED_REGION_CODE_INT = 16
SELECTED_REGION_NAME = "Jihomoravský kraj"


@dataclass(frozen=True)
class _RegionSpec:
    col: str
    value: object
    label: str


def _get_region_spec(df: pd.DataFrame) -> _RegionSpec:
    """Pick a single region to use in plots.

    The assignment requires choosing one region (kraj). We prefer JHM if present;
    otherwise we fall back to the most frequent region value.
    """

    if "region" in df.columns:
        if (df["region"] == SELECTED_REGION_CODE_STR).any():
            return _RegionSpec("region", SELECTED_REGION_CODE_STR, SELECTED_REGION_NAME)
        # fallback to mode
        mode_val = df["region"].mode(dropna=True)
        val = mode_val.iloc[0] if not mode_val.empty else df["region"].dropna().iloc[0]
        return _RegionSpec("region", val, str(val))

    if "p4a" in df.columns:
        if (df["p4a"] == SELECTED_REGION_CODE_INT).any():
            return _RegionSpec("p4a", SELECTED_REGION_CODE_INT, SELECTED_REGION_NAME)
        mode_val = df["p4a"].replace(-1, np.nan).dropna().mode()
        val = int(mode_val.iloc[0]) if not mode_val.empty else int(df["p4a"].iloc[0])
        return _RegionSpec("p4a", val, f"region={val}")

    raise ValueError("Unable to identify region column ('region' or 'p4a').")


def _get_date_series(df: pd.DataFrame) -> pd.Series:
    """Return accident date as datetime64[ns] series."""

    if "date" in df.columns and np.issubdtype(df["date"].dtype, np.datetime64):
        return df["date"]

    if "p2a" in df.columns:
        # The provided dataset uses DD.MM.YYYY.
        return pd.to_datetime(df["p2a"], format="%d.%m.%Y", errors="coerce")

    raise ValueError("Unable to identify date column ('date' or 'p2a').")


# ---------------------------------
# Required public API (Task 1 – geo)
# ---------------------------------

def make_geo(
    df_accidents: pd.DataFrame, df_locations: pd.DataFrame
) -> "gpd.GeoDataFrame":
    """Create a GeoDataFrame by merging accidents with locations.

    The returned GeoDataFrame:
    - contains *all regions* (no filtering by kraj),
    - uses CRS EPSG:5514 (S-JTSK / Krovak East North),
    - removes rows with unknown position (d/e == 0 or missing),
    - fixes swapped coordinates: if d < e, swap x/y.

    Parameters
    ----------
    df_accidents:
        DataFrame from `accidents.pkl.gz`.
    df_locations:
        DataFrame from `locations.pkl.gz`.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with Point geometries.
    """

    if "p1" not in df_accidents.columns or "p1" not in df_locations.columns:
        raise ValueError("Both input frames must contain key column 'p1'.")

    if "d" not in df_locations.columns or "e" not in df_locations.columns:
        raise ValueError("Locations frame must contain columns 'd' and 'e'.")

    merged = df_accidents.merge(
        df_locations[["p1", "d", "e"] + [c for c in df_locations.columns if c not in {"p1", "d", "e"}]],
        on="p1",
        how="left",
        suffixes=("", "_loc"),
    )

    # Unknown/invalid coordinates: zeros and NaNs.
    d = pd.to_numeric(merged["d"], errors="coerce")
    e = pd.to_numeric(merged["e"], errors="coerce")

    # Drop explicit "unknown" zeros.
    mask_valid = d.notna() & e.notna() & (d != 0) & (e != 0)
    merged = merged.loc[mask_valid].copy()
    d = d.loc[mask_valid].astype(float)
    e = e.loc[mask_valid].astype(float)

    # Fix swapped coordinates: for S-JTSK in CZ, x (d) is usually "less negative" than y (e).
    # If d < e, it likely indicates swapped axes.
    swap = d < e
    d_fixed = d.where(~swap, e)
    e_fixed = e.where(~swap, d)

    merged["d"] = d_fixed
    merged["e"] = e_fixed

    gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["d"], merged["e"]),
        crs="EPSG:5514",
    )

    # Sanity filter: keep only points that fall into a broad CZ bounding box in WGS84.
    # This mainly removes extreme outliers (e.g., wrong swaps, null-like coordinates).
    wgs = gdf.to_crs("EPSG:4326")
    lon = wgs.geometry.x
    lat = wgs.geometry.y
    cz_bbox = lon.between(11.5, 19.5) & lat.between(47.5, 51.5)
    gdf = gdf.loc[cz_bbox].copy()

    return gdf


def plot_geo(
    gdf: "gpd.GeoDataFrame", fig_location: str = None, show_figure: bool = False
):
    """Plot positions of wildlife-caused accidents in a chosen region for 2023 and 2024.

    Creates a figure with two subplots (years 2023 and 2024), using a basemap.

    Parameters
    ----------
    gdf:
        GeoDataFrame returned by :func:`make_geo`.
    fig_location:
        Output image path (e.g. "geo1.png"). If None, the figure is not saved.
    show_figure:
        If True, display the figure. Otherwise, close it after saving.
    """

    region = _get_region_spec(gdf)
    dates = _get_date_series(gdf)
    years = dates.dt.year

    sel = (
        (gdf[region.col] == region.value)
        & (gdf.get("p10", pd.Series(index=gdf.index, data=np.nan)) == 4)
        & years.isin([2023, 2024])
    )
    data = gdf.loc[sel].copy()
    data["_year"] = years.loc[sel]

    # Use Web Mercator for contextily tiles.
    data_3857 = data.to_crs("EPSG:3857")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Consistent extent across both years.
    if not data_3857.empty:
        xmin, ymin, xmax, ymax = data_3857.total_bounds
        pad_x = (xmax - xmin) * 0.05
        pad_y = (ymax - ymin) * 0.05
        extent = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)
    else:
        extent = None

    for ax, year in zip(axes, [2023, 2024], strict=True):
        subset = data_3857.loc[data_3857["_year"] == year]
        if subset.empty:
            ax.set_title(f"{region.label} – zvěř (p10=4), {year} (0 nehod)")
            ax.axis("off")
            continue

        subset.plot(
            ax=ax,
            markersize=10,
            color="#D55E00",
            alpha=0.6,
            linewidth=0,
        )

        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            crs=subset.crs,
            attribution_size=6,
        )
        ax.set_title(f"{region.label} – zvěř (p10=4), {year} (n={len(subset)})")
        ax.set_axis_off()
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

    if fig_location:
        fig.savefig(fig_location, dpi=200, bbox_inches="tight")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


def plot_cluster(
    gdf: "gpd.GeoDataFrame", fig_location: str = None, show_figure: bool = False
):
    """Cluster alcohol-related accidents in a chosen region and visualize hotspots.

    We plot points for accidents with significant alcohol involvement (p11 >= 4).
    Clusters are computed via DBSCAN in S-JTSK (meters), then visualized as convex
    hull polygons shaded by accident counts.

    Notes on method choice (required by assignment):
    - KMeans requires choosing K and tends to create spherical clusters even in
      elongated road networks.
    - Agglomerative clustering often over-splits dense city centers.
    - DBSCAN is robust to noise and naturally separates dense hotspots from sparse
      isolated accidents. We therefore use DBSCAN with parameters chosen to keep
      the visualization readable (few dozen clusters, plus noise).

    Parameters
    ----------
    gdf:
        GeoDataFrame returned by :func:`make_geo`.
    fig_location:
        Output image path (e.g. "geo2.png"). If None, the figure is not saved.
    show_figure:
        If True, display the figure. Otherwise, close it after saving.
    """

    region = _get_region_spec(gdf)

    sel = (gdf[region.col] == region.value) & (gdf.get("p11", -1) >= 4)
    data = gdf.loc[sel].copy()

    if data.empty:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title(f"{region.label} – alkohol (p11>=4): 0 nehod")
        ax.axis("off")
        if fig_location:
            fig.savefig(fig_location, dpi=200, bbox_inches="tight")
        if show_figure:
            plt.show()
        else:
            plt.close(fig)
        return

    # DBSCAN in meters (EPSG:5514). Parameter heuristics:
    # - eps: ~1.2–2.0 km tends to capture urban hotspots without merging the whole region
    # - min_samples: scale with N to avoid tiny clusters
    n = len(data)
    eps = 1800 if n > 2000 else 1400
    min_samples = max(12, int(round(n * 0.01)))

    coords = np.column_stack([data.geometry.x.to_numpy(), data.geometry.y.to_numpy()])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    data["_cluster"] = labels

    clustered = data.loc[data["_cluster"] >= 0].copy()

    # Build convex hull polygons per cluster.
    hull_rows = []
    for cid, grp in clustered.groupby("_cluster"):
        pts = list(grp.geometry)
        hull = MultiPoint(pts).convex_hull
        # Buffer slightly to make thin hulls visible.
        hull = hull.buffer(80)
        hull_rows.append({"cluster": int(cid), "count": len(grp), "geometry": hull})

    hulls = gpd.GeoDataFrame(hull_rows, crs=data.crs)

    # Plot in 3857 for basemap.
    data_3857 = data.to_crs("EPSG:3857")
    hulls_3857 = hulls.to_crs("EPSG:3857")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)

    # Color polygons by counts.
    cmap = plt.get_cmap("viridis")
    vmin = int(hulls_3857["count"].min())
    vmax = int(hulls_3857["count"].max())

    hulls_3857.plot(
        ax=ax,
        column="count",
        cmap=cmap,
        alpha=0.35,
        edgecolor="#222222",
        linewidth=0.8,
        legend=True,
        legend_kwds={"label": "Počet nehod v oblasti (cluster)", "shrink": 0.8},
    )

    # Plot points (including noise) on top.
    data_3857.plot(
        ax=ax,
        markersize=6,
        color="#000000",
        alpha=0.45,
        linewidth=0,
    )

    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron,
        crs=data_3857.crs,
        attribution_size=6,
    )

    ax.set_title(
        f"{region.label} – oblasti s alkoholem (p11>=4), DBSCAN eps={eps} m, min_samples={min_samples}"
    )
    ax.set_axis_off()

    # Set extent to region points (ignore possible misclassified region points).
    xmin, ymin, xmax, ymax = data_3857.total_bounds
    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    if fig_location:
        fig.savefig(fig_location, dpi=200, bbox_inches="tight")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    # Optional local demo (expects *.pkl.gz files next to this script).
    acc = pd.read_pickle("accidents.pkl.gz")
    loc = pd.read_pickle("locations.pkl.gz")
    gdf0 = make_geo(acc, loc)
    plot_geo(gdf0, fig_location="geo1.png", show_figure=False)
    plot_cluster(gdf0, fig_location="geo2.png", show_figure=False)
    print("Generated geo1.png and geo2.png")
