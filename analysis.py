#!/usr/bin/env python3
"""Basic exploration utilities for the IZV accident dataset (2023-2025).

The module intentionally keeps all logic inside functions so that it can be
imported by the grader. Use :func:`load_data` to read data frames from the zip
archive, :func:`parse_data` to prepare the accident table, and the ``plot_*``
functions to render the required visualisations.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, List
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

YEARS = (2023, 2024, 2025)
REGION_MAP = {
    0: "PHA",
    1: "STC",
    2: "JHC",
    3: "PLK",
    4: "ULK",
    5: "HKK",
    6: "JHM",
    7: "MSK",
    14: "OLK",
    15: "ZLK",
    16: "VYS",
    17: "PAK",
    18: "LBK",
    19: "KVK",
}
REGION_ORDER = list(REGION_MAP.values())
STATE_LABELS = {
    3: "Únava nebo spánek",
    4: "Onemocnění",
    5: "Pod vlivem alkoholu",
    6: "Pod vlivem jiných látek",
    7: "Invalidita",
    8: "Sebevražda",
    9: "Řidič zemřel",
}
STATE_ORDER = list(STATE_LABELS.keys())
INJURY_LABELS = {
    1: "Usmrcení",
    2: "Těžké zranění",
    3: "Lehké zranění",
    4: "Bez zranění",
}
INJURY_ORDER = [4, 3, 2, 1]
CONDITION_LABELS = {
    0: "Nezjištěno",
    1: "Střízlivý",
    2: "Do 0,24‰",
    3: "0,24–0,5‰",
    4: "0,5–1,0‰",
    5: "1,0‰ a více",
    6: "Podezření jiné látky",
    7: "Odmítl dech",
    8: "Odmítl krev",
    9: "Jiné ovlivnění",
}
CONDITION_ORDER = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]


def load_data(filename: str, ds: str) -> pd.DataFrame:
    """Load a given dataset from the IZV archive for all configured years.

    Parameters
    ----------
    filename:
        Path to the ``.zip`` archive downloaded from IZV (do not download here).
    ds:
        Name of the dataset without the ``I`` prefix (e.g. ``"nehody"``).
    """

    dataset = ds.strip()
    if not dataset:
        raise ValueError("Dataset identifier must be a non-empty string")

    frames: List[pd.DataFrame] = []
    with zipfile.ZipFile(filename, mode="r") as archive:
        members = {name.lower(): name for name in archive.namelist()}
        for year in YEARS:
            wanted = [
                f"{year}/I{dataset}.xls",
                f"I{dataset}_{year}.xls",
                f"{dataset}/{year}/I{dataset}.xls",
            ]
            member = _resolve_member(members, wanted)
            frames.append(_read_member(archive, member, year))

    data = pd.concat(frames, ignore_index=True)
    data = data.loc[:, ~data.columns.str.startswith("Unnamed")]
    data = data.rename(columns={"p14*100": "p14"})
    return data


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Create a cleaned accident table and attach ``date`` and ``region``."""

    result = df.copy()
    result = result.dropna(subset=["p1", "p2a"])  # ensure IDs and dates exist
    result["p1"] = result["p1"].astype(str)
    result["date"] = pd.to_datetime(result["p2a"], format="%d.%m.%Y", errors="coerce")

    region_col = _get_region_column(result)
    codes = pd.to_numeric(result[region_col], errors="coerce").astype("Int64")
    result["region"] = pd.Categorical(
        codes.map(REGION_MAP), categories=REGION_ORDER, ordered=True
    )

    result = result.drop_duplicates(subset="p1")
    if verbose:
        size_mb = result.memory_usage(deep=True).sum() / 1_000_000
        print(f"new_size={size_mb:.1f} MB")
    return result


def plot_state(
    df: pd.DataFrame,
    df_vehicles: pd.DataFrame,
    fig_location: str | None = None,
    show_figure: bool = False,
) -> None:
    """Plot counts of accidents per region split by driver state (p57)."""

    required = {"p1", "region"}
    if not required.issubset(df.columns):
        raise ValueError("Data frame must contain parsed accidents with 'region'.")
    if "p57" not in df_vehicles.columns:
        raise ValueError("Vehicle data must contain column 'p57'.")

    accidents = df[["p1", "region"]].copy()
    vehicles = df_vehicles[["p1", "p57"]].copy()
    accidents["p1"] = accidents["p1"].astype(str)
    vehicles["p1"] = vehicles["p1"].astype(str)
    merged = accidents.merge(vehicles, on="p1", how="inner")
    merged["p57"] = pd.to_numeric(merged["p57"], errors="coerce")
    merged = merged[merged["p57"].between(3, 9)]
    merged["driver_state"] = merged["p57"].astype(int).map(STATE_LABELS)
    merged = merged.dropna(subset=["driver_state", "region"])

    region_levels = _ordered_levels(merged["region"], REGION_ORDER)
    state_levels = [STATE_LABELS[s] for s in STATE_ORDER if STATE_LABELS[s] in merged["driver_state"].unique()]
    if not region_levels or not state_levels:
        raise ValueError("No data available for the requested driver states.")

    index = pd.MultiIndex.from_product(
        [state_levels, region_levels], names=["driver_state", "region"]
    )
    counts = (
        merged.groupby(["driver_state", "region"]).size().reindex(index, fill_value=0).reset_index(name="count")
    )

    n_states = len(state_levels)
    cols = 2
    rows = (n_states + cols - 1) // cols
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), sharey=False)
    axes = axes.flatten()
    palette = sns.color_palette("Set2", len(region_levels))
    region_palette = dict(zip(region_levels, palette))

    for idx, state in enumerate(state_levels):
        ax = axes[idx]
        state_data = counts[counts["driver_state"] == state].copy()
        sns.barplot(
            data=state_data,
            x="region",
            y="count",
            hue="region",
            palette=region_palette,
            dodge=False,
            ax=ax,
        )
        ax.set_title(state, fontsize=12, fontweight="bold")
        ax.set_xlabel("Kraj")
        ax.set_ylabel("Počet nehod")
        ax.tick_params(axis="x", rotation=45)
        ax.set_facecolor("#f8f9fb")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    for j in range(len(state_levels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Počty nehod podle stavu řidiče", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _finalize_figure(fig, fig_location, show_figure)


def plot_alcohol(
    df: pd.DataFrame,
    df_consequences: pd.DataFrame,
    fig_location: str | None = None,
    show_figure: bool = False,
) -> None:
    """Visualise counts of alcohol-related consequences per region and year."""

    needed_cols = {"p1", "date", "region", "p11"}
    if not needed_cols.issubset(df.columns):
        raise ValueError("Parsed accidents must contain 'p1', 'date', 'region', and 'p11'.")
    if "p59g" not in df_consequences.columns:
        raise ValueError("Consequences data must contain column 'p59g'.")

    accidents = df[list(needed_cols)].copy()
    consequences = df_consequences[["p1", "p59g"]].copy()
    accidents["p1"] = accidents["p1"].astype(str)
    consequences["p1"] = consequences["p1"].astype(str)
    merged = accidents.merge(consequences, on="p1", how="inner")
    merged = merged.dropna(subset=["date", "region", "p59g"])
    merged = merged[merged["p11"] >= 3]
    merged = merged[merged["date"].dt.month <= 10]
    merged["year"] = merged["date"].dt.year
    merged["injury_type"] = merged["p59g"].astype(int).map(INJURY_LABELS)
    merged = merged.dropna(subset=["injury_type"])

    region_levels = [r for r in REGION_ORDER if r in merged["region"].unique()]
    if not region_levels:
        raise ValueError("No regions available after filtering.")

    aggregated = (
        merged.groupby(["injury_type", "region", "year"]).size().reset_index(name="count")
    )

    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, injury in zip(axes, [INJURY_LABELS[i] for i in INJURY_ORDER]):
        injury_data = aggregated[aggregated["injury_type"] == injury]
        if injury_data.empty:
            ax.text(0.5, 0.5, "Žádná data", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            continue

        pivot = injury_data.pivot_table(
            index="region",
            columns="year",
            values="count",
            fill_value=0,
        ).reindex(region_levels)
        sns.heatmap(
            pivot,
            cmap="YlOrRd",
            annot=True,
            fmt="d",
            cbar_kws={"label": "Počet následků"},
            linewidths=0.4,
            ax=ax,
        )
        ax.set_title(injury, fontweight="bold")
        ax.set_xlabel("Rok")
        ax.set_ylabel("Kraj")

    fig.tight_layout()
    fig.suptitle(
        "Následky nehod pod vlivem alkoholu (do října)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    _finalize_figure(fig, fig_location, show_figure)


def plot_conditions(
    df: pd.DataFrame,
    fig_location: str | None = None,
    show_figure: bool = False,
) -> None:
    """Plot monthly counts for four regions grouped by driver condition (p11)."""

    if not {"date", "region", "p11"}.issubset(df.columns):
        raise ValueError("Parsed accidents must include 'date', 'region', and 'p11'.")

    available = [r for r in REGION_ORDER if r in df["region"].unique()]
    preferred = [r for r in ("JHM", "MSK", "OLK", "ZLK") if r in available]
    remaining = [r for r in available if r not in preferred]
    selected = (preferred + remaining)[:4]
    if len(selected) < 4:
        raise ValueError("Not enough regions with data to build the plot.")

    subset = df[df["region"].isin(selected)].copy()
    subset = subset.dropna(subset=["date"])
    subset["condition"] = subset["p11"].map(CONDITION_LABELS)
    missing_mask = subset["condition"].isna()
    if missing_mask.any():
        subset.loc[missing_mask, "condition"] = subset.loc[missing_mask, "p11"].apply(
            lambda value: f"Stav {int(value)}" if pd.notna(value) else "Neznámý"
        )
    subset = subset[(subset["date"] >= "2023-01-01") & (subset["date"] < "2025-01-01")]
    subset["month"] = subset["date"].dt.to_period("M")

    monthly = (
        subset.groupby(["region", "condition", "month"]).size().reset_index(name="count")
    )
    monthly["month_start"] = monthly["month"].dt.to_timestamp()

    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    palette = sns.color_palette("tab20", len(CONDITION_ORDER))
    color_map = {
        CONDITION_LABELS.get(code, f"Stav {code}"): palette[idx]
        for idx, code in enumerate(CONDITION_ORDER)
    }

    legend_handles: dict[str, plt.Line2D] = {}
    for ax, region in zip(axes, selected):
        region_data = monthly[monthly["region"] == region]
        for condition, color in color_map.items():
            cond_data = region_data[region_data["condition"] == condition]
            if cond_data.empty:
                continue
            (line,) = ax.plot(
                cond_data["month_start"],
                cond_data["count"],
                label=condition,
                color=color,
                linewidth=2,
                marker="o",
                markersize=3,
            )
            legend_handles.setdefault(condition, line)
        ax.set_title(f"Kraj {region}", fontweight="bold")
        ax.set_xlabel("Měsíc")
        ax.set_ylabel("Počet nehod")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01"))

    if legend_handles:
        handles = list(legend_handles.values())
        labels = list(legend_handles.keys())
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.suptitle(
        "Vývoj počtu nehod podle stavu řidiče (p11)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    _finalize_figure(fig, fig_location, show_figure)


def _resolve_member(members: dict[str, str], candidates: Iterable[str]) -> str:
    for candidate in candidates:
        key = candidate.lower()
        if key in members:
            return members[key]
    raise FileNotFoundError(f"None of {list(candidates)} found in archive")


def _read_member(archive: zipfile.ZipFile, member: str, year: int) -> pd.DataFrame:
    with archive.open(member) as handle:
        tables = pd.read_html(
            io.BytesIO(handle.read()),
            encoding="cp1250",
            decimal=",",
            thousands="\xa0",
        )
    table = tables[0]
    table["year"] = year
    return table


def _get_region_column(df: pd.DataFrame) -> str:
    best_column = None
    best_match_count = -1
    for column in ("p4a", "p5a"):
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        matches = numeric.dropna().isin(REGION_MAP.keys()).sum()
        if matches > best_match_count:
            best_match_count = matches
            best_column = column
    if best_column is None or best_match_count <= 0:
        raise KeyError("Region column (p4a/p5a) not found in dataframe")
    return best_column


def _ordered_levels(series: pd.Series, order: List[str]) -> List[str]:
    return [region for region in order if region in series.unique()]


def _finalize_figure(fig: plt.Figure, fig_location: str | None, show: bool) -> None:
    if fig_location:
        path = Path(fig_location)
        if path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # Optional quick smoke-test to ensure the loaders work when run manually.
    print("analysis.py module ready. Import and call the functions from graders.")
