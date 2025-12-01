#!/usr/bin/env python3
# coding=utf-8

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile
import io
from pathlib import Path
from typing import Iterable, Dict

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

YEARS = (2023, 2024, 2025)
MBYTE = 1_000_000

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

STATE_MAP = {
    3: "Únava nebo spánek",
    4: "Onemocnění",
    5: "Pod vlivem alkoholu",
    6: "Pod vlivem jiných látek",
    7: "Invalidita",
    8: "Sebevražda",
    9: "Řidič zemřel",
}
STATE_ORDER = [3, 4, 5, 6, 7, 8, 9]

INJURY_MAP = {
    1: "Usmrcení",
    2: "Těžké zranění",
    3: "Lehké zranění",
    4: "Bez zranění",
}
INJURY_ORDER = [4, 3, 2, 1]

CONDITION_MAP = {
    0: "Nezjištěno",
    1: "Střízlivý",
    2: "< 0,24 ‰",
    3: "0,24–0,5 ‰",
    4: "0,5–1,0 ‰",
    5: "≥ 1,0 ‰",
    6: "Jiné látky",
    7: "Odmítl dech",
    8: "Odmítl krev",
    9: "Jiné ovlivnění",
}
CONDITION_ORDER = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]


# Ukol 1: nacteni dat ze ZIP souboru
def load_data(filename: str, ds: str) -> pd.DataFrame:
    """Načte tabulku z archivu a spojí roky 2023–2025."""

    dataset = ds.strip()
    if not dataset:
        raise ValueError("Dataset identifier nesmí být prázdný.")

    frames = []
    with zipfile.ZipFile(filename, "r") as archive:
        members = {name.lower(): name for name in archive.namelist()}
        for year in YEARS:
            member = _pick_member(members, dataset, year)
            frames.append(_read_member(archive, member, year))

    df = pd.concat(frames, ignore_index=True)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    if "p14*100" in df.columns and "p14" not in df.columns:
        df = df.rename(columns={"p14*100": "p14"})
    return df


# Ukol 2: zpracovani dat
def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Vyčistí tabulku nehod, doplní datum a zkratku kraje."""

    result = df.copy()
    result = result.dropna(subset=["p1"])
    result["p1"] = result["p1"].astype(str)
    result["date"] = pd.to_datetime(result["p2a"], format="%d.%m.%Y", errors="coerce")

    region_codes = _extract_region_codes(result)
    result["region"] = pd.Categorical(
        region_codes.map(REGION_MAP), categories=REGION_ORDER, ordered=True
    )

    result = result.dropna(subset=["date", "region"]).drop_duplicates("p1")
    if verbose:
        size_mb = result.memory_usage(deep=True).sum() / MBYTE
        print(f"new_size={size_mb:.1f} MB")
    return result


# Ukol 3: počty nehod v jednotlivých regionech podle stavu řidiče
def plot_state(
    df: pd.DataFrame,
    df_vehicles: pd.DataFrame,
    fig_location: str | None = None,
    show_figure: bool = False,
) -> None:
    """Vykreslí počty nehod per kraj zvlášť pro jednotlivé stavy řidiče (p57)."""

    if not {"p1", "region"}.issubset(df.columns):
        raise ValueError("Data frame df musí obsahovat sloupce 'p1' a 'region'.")
    if "p57" not in df_vehicles.columns:
        raise ValueError("Data frame df_vehicles musí obsahovat sloupec 'p57'.")

    merged = _merge_on_p1(df[["p1", "region"]], df_vehicles[["p1", "p57"]])
    merged["p57"] = pd.to_numeric(merged["p57"], errors="coerce")
    merged = merged[merged["p57"].between(3, 9)]
    merged["state"] = merged["p57"].astype(int).map(STATE_MAP)
    merged = merged.dropna(subset=["state", "region"])

    region_levels = [r for r in REGION_ORDER if r in merged["region"].unique()]
    state_levels = [STATE_MAP[s] for s in STATE_ORDER if STATE_MAP.get(s) in merged["state"].unique()]
    if not region_levels or not state_levels:
        raise ValueError("Není co zobrazit, zkontrolujte vstupní data.")

    index = pd.MultiIndex.from_product([state_levels, region_levels], names=["state", "region"])
    counts = (
        merged.groupby(["state", "region"], observed=False)
        .size()
        .reindex(index, fill_value=0)
        .reset_index(name="accidents")
    )

    cols = 2
    rows = max(3, int(np.ceil(len(state_levels) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharey=False)
    axes = axes.flatten()
    palette = sns.color_palette("crest", len(region_levels))
    region_palette = dict(zip(region_levels, palette))

    for idx, state in enumerate(state_levels):
        if idx >= len(axes):
            break
        ax = axes[idx]
        state_data = counts[counts["state"] == state]
        sns.barplot(
            data=state_data,
            x="region",
            y="accidents",
            hue="region",
            palette=region_palette,
            dodge=False,
            ax=ax,
        )
        ax.set_title(state, fontsize=12, fontweight="bold")
        ax.set_xlabel("Kraj")
        ax.set_ylabel("Počet nehod")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.set_facecolor("#f6f7fb")
        legend = ax.get_legend()
        if legend:
            legend.remove()

    for ax in axes[len(state_levels) :]:
        ax.axis("off")

    fig.suptitle("Počty nehod podle stavu řidiče", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_or_show(fig, fig_location, show_figure)


# Ukol4: alkohol a roky v krajích
def plot_alcohol(
    df: pd.DataFrame,
    df_consequences: pd.DataFrame,
    fig_location: str | None = None,
    show_figure: bool = False,
) -> None:
    """Heatmapy následků nehod pod vlivem alkoholu (p11 ≥ 3)."""

    needed = {"p1", "date", "region", "p11"}
    if not needed.issubset(df.columns):
        raise ValueError("df musí být výstup parse_data s p11/date/region.")
    if "p59g" not in df_consequences.columns:
        raise ValueError("df_consequences musí obsahovat sloupec 'p59g'.")

    merged = _merge_on_p1(df[list(needed)], df_consequences[["p1", "p59g"]])
    merged = merged.dropna(subset=["date", "region", "p59g"])
    merged = merged[merged["p11"] >= 3]
    merged = merged[merged["date"].dt.month <= 10]
    merged["year"] = merged["date"].dt.year
    merged["injury"] = merged["p59g"].astype(int).map(INJURY_MAP)
    merged = merged.dropna(subset=["injury"])

    region_levels = [r for r in REGION_ORDER if r in merged["region"].unique()]
    if not region_levels:
        raise ValueError("Není k dispozici žádný kraj po zfiltrování.")

    aggregated = (
        merged.groupby(["injury", "region", "year"], observed=False)
        .size()
        .reset_index(name="count")
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    sns.set_theme(style="white")

    for ax, injury in zip(axes, [INJURY_MAP[i] for i in INJURY_ORDER]):
        injury_data = aggregated[aggregated["injury"] == injury]
        if injury_data.empty:
            ax.text(0.5, 0.5, "Žádná data", ha="center", va="center", fontsize=12)
            ax.axis("off")
            continue

        pivot = injury_data.pivot_table(
            index="region",
            columns="year",
            values="count",
            fill_value=0,
            observed=False,
        ).reindex(region_levels, fill_value=0)
        pivot = pivot.astype(int)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            annot=True,
            fmt="g",
            linewidths=0.5,
            cbar_kws={"label": "Počet následků"},
        )
        ax.set_title(injury, fontweight="bold")
        ax.set_xlabel("Rok")
        ax.set_ylabel("Kraj")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(
        "Následky nehod pod vlivem alkoholu (do října)",
        fontsize=16,
        fontweight="bold",
    )
    _save_or_show(fig, fig_location, show_figure)


# Ukol 5: Podmínky v čase
def plot_conditions(
    df: pd.DataFrame,
    fig_location: str | None = None,
    show_figure: bool = False,
) -> None:
    """Čárový graf měsíčních počtů nehod dle podmínek (p11) pro 4 kraje."""

    required = {"date", "region", "p11"}
    if not required.issubset(df.columns):
        raise ValueError("df musí obsahovat sloupce date/region/p11.")

    available = [r for r in ("JHM", "MSK", "OLK", "ZLK") if r in df["region"].unique()]
    if len(available) < 4:
        fallback = [r for r in REGION_ORDER if r in df["region"].unique() and r not in available]
        available.extend(fallback)
    selected = available[:4]
    if len(selected) < 4:
        raise ValueError("K vykreslení jsou potřeba data alespoň pro čtyři kraje.")

    subset = df[df["region"].isin(selected)].dropna(subset=["date"])
    subset = subset[(subset["date"] >= "2023-01-01") & (subset["date"] < "2025-01-01")]
    subset["condition"] = subset["p11"].map(CONDITION_MAP)
    subset["condition"] = subset["condition"].fillna(subset["p11"].apply(lambda x: f"Stav {int(x)}"))
    subset["month"] = subset["date"].dt.to_period("M")

    monthly = (
        pd.crosstab([subset["region"], subset["month"]], subset["condition"])
        .stack()
        .rename("count")
        .reset_index()
    )
    monthly["month_start"] = monthly["month"].dt.to_timestamp()

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    palette = sns.color_palette("tab20", len(CONDITION_ORDER))
    condition_colors = {
        CONDITION_MAP.get(code, f"Stav {code}"): palette[idx]
        for idx, code in enumerate(CONDITION_ORDER)
    }
    legend_handles: Dict[str, Line2D] = {}

    for ax, region in zip(axes, selected):
        region_data = monthly[monthly["region"] == region]
        for condition, color in condition_colors.items():
            cond = region_data[region_data["condition"] == condition]
            if cond.empty:
                continue
            (line,) = ax.plot(
                cond["month_start"],
                cond["count"],
                label=condition,
                color=color,
                linewidth=2,
                marker="o",
                markersize=3,
            )
            legend_handles.setdefault(condition, line)
        ax.set_title(f"Kraj {region}", fontweight="bold")
        ax.set_xlabel("Datum")
        ax.set_ylabel("Počet nehod")
        ax.set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01"))
        ax.grid(True, linestyle="--", alpha=0.4)

    if legend_handles:
        handles = list(legend_handles.values())
        labels = list(legend_handles.keys())
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.suptitle("Vývoj počtu nehod podle stavu řidiče (p11)", fontsize=16, fontweight="bold", y=1.02)
    _save_or_show(fig, fig_location, show_figure)


def _pick_member(members: Dict[str, str], dataset: str, year: int) -> str:
    variants = {
        dataset,
        dataset.lower(),
        dataset.upper(),
        dataset.capitalize(),
    }
    candidates = []
    for variant in variants:
        prefix = f"I{variant}"
        candidates.extend(
            [
                f"{year}/{prefix}.xls",
                f"{prefix}_{year}.xls",
                f"data/{year}/{prefix}.xls",
                f"{variant}/{year}/{prefix}.xls",
            ]
        )
    for candidate in candidates:
        key = candidate.lower()
        if key in members:
            return members[key]
    raise FileNotFoundError(f"Soubor pro dataset '{dataset}' a rok {year} nebyl v archivu nalezen.")


def _read_member(archive: zipfile.ZipFile, member: str, year: int) -> pd.DataFrame:
    with archive.open(member) as handle:
        buffer = io.BytesIO(handle.read())
    table = pd.read_html(buffer, encoding="cp1250", decimal=",", thousands=" ")[0]
    table["year"] = year
    return table


def _extract_region_codes(df: pd.DataFrame) -> pd.Series:
    best_codes = None
    best_matches = -1
    for column in ("p4a", "p5a"):
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        matches = numeric.dropna().isin(REGION_MAP.keys()).sum()
        if matches > best_matches:
            best_matches = matches
            best_codes = numeric.astype("Int64")
    if best_codes is None or best_matches <= 0:
        raise KeyError("Nelze najít sloupec s kódem kraje (p4a/p5a).")
    return best_codes


def _merge_on_p1(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged_left = left.copy()
    merged_right = right.copy()
    merged_left["p1"] = merged_left["p1"].astype(str)
    merged_right["p1"] = merged_right["p1"].astype(str)
    return merged_left.merge(merged_right, on="p1", how="inner")


def _save_or_show(fig: plt.Figure, fig_location: str | None, show_figure: bool) -> None:
    if fig_location:
        path = Path(fig_location)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    if show_figure:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    print("Modul analysis.py byl naimportován. Spusťte jednotlivé funkce v testech.")
