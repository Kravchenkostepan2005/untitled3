"""IZV project 2025 – Task 3 (variant Nehody).

This script reads the provided *.pkl.gz datasets (expected in the same directory
as this file) and:

- generates a figure `fig.pdf` used in the report,
- generates a one-page A4 report `doc.pdf` (graph + table + text),
- prints table data and computed values to standard output.

The assignment evaluates primarily the correctness and interpretability of the
analysis; the code aims to be deterministic and self-contained.

Author: (fill in if needed)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_accidents(data_dir: Path) -> pd.DataFrame:
    """Load accidents dataframe and ensure a usable datetime column."""

    acc = pd.read_pickle(data_dir / "accidents.pkl.gz")

    if "date" in acc.columns and np.issubdtype(acc["date"].dtype, np.datetime64):
        acc_date = acc["date"]
    else:
        acc_date = pd.to_datetime(acc["p2a"], format="%d.%m.%Y", errors="coerce")

    acc = acc.copy()
    acc["__date"] = acc_date
    acc["__year"] = acc_date.dt.year

    # Prefer a human-readable region code if available.
    if "region" not in acc.columns and "p4a" in acc.columns:
        # Standard IZV mapping used in previous parts.
        p4a_map = {
            0: "PHA",
            1: "STC",
            2: "JHC",
            3: "PLK",
            4: "KVK",
            5: "ULK",
            6: "LBK",
            7: "HKK",
            14: "PAK",
            15: "VYS",
            16: "JHM",
            17: "OLK",
            18: "ZLK",
            19: "MSK",
        }
        acc["region"] = acc["p4a"].map(p4a_map).fillna("UNK")

    return acc


def _road_type_labels() -> dict[int, str]:
    """Return readable labels for p36 (road type) codes."""

    return {
        0: "neznámé",
        1: "dálnice",
        2: "silnice I.",
        3: "silnice II.",
        6: "silnice III.",
        7: "účelová",
        8: "jiné",
    }


def _compute_region_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute table: accidents, deaths, death rate per 10k by region."""

    out = (
        df.groupby("region")
        .agg(
            accidents=("p1", "size"),
            deaths=("p13a", "sum"),
            alcohol_accidents=("p11", lambda s: int((s >= 4).sum())),
        )
        .reset_index()
    )
    out["death_rate_per_10k"] = (out["deaths"] / out["accidents"]) * 10000
    out["alcohol_share"] = out["alcohol_accidents"] / out["accidents"]
    return out.sort_values("death_rate_per_10k", ascending=False)


def _make_fig(df: pd.DataFrame, out_path: Path) -> None:
    """Create and save the main graph used in the report."""

    labels = _road_type_labels()

    d = df.copy()
    d["road"] = d["p36"].map(labels).fillna(d["p36"].astype(str).radd("p36="))
    d["alcohol"] = np.where(d["p11"] >= 4, "alkohol (p11>=4)", "bez alkoholu (p11<4)")
    d["death"] = (d["p13a"] > 0).astype(int)

    # Death rate per 10k accidents by road type, split by alcohol.
    grp = (
        d.groupby(["road", "alcohol"])  # type: ignore[call-arg]
        .agg(accidents=("p1", "size"), deaths=("death", "sum"))
        .reset_index()
    )
    grp["death_rate_per_10k"] = (grp["deaths"] / grp["accidents"]) * 10000

    # Keep readable ordering: known codes first.
    order = [labels[k] for k in [1, 2, 3, 6, 7, 8] if k in labels]
    grp["road"] = pd.Categorical(grp["road"], categories=order, ordered=True)
    grp = grp.sort_values("road")

    # Pivot for grouped bar plot.
    piv = grp.pivot(index="road", columns="alcohol", values="death_rate_per_10k").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    piv.plot(kind="bar", ax=ax, width=0.8, color=["#0072B2", "#D55E00"])

    ax.set_title("Úmrtnost nehod dle typu komunikace (2023–2024)\nporovnání alkoholu vs. bez alkoholu")
    ax.set_xlabel("Typ komunikace (p36)")
    ax.set_ylabel("Nehody s úmrtím na 10 000 nehod")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Skupina")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _make_doc_pdf(df: pd.DataFrame, table: pd.DataFrame, fig_path: Path, out_path: Path) -> None:
    """Generate a one-page A4 PDF report (graph + table + text)."""

    # --- computed values for the text ---
    n_total = int(len(df))
    deaths_total = float(df["p13a"].sum())
    alcohol_n = int((df["p11"] >= 4).sum())
    alcohol_share = alcohol_n / n_total if n_total else 0.0

    alcohol_deaths = float(df.loc[df["p11"] >= 4, "p13a"].sum())
    non_alcohol_deaths = float(df.loc[df["p11"] < 4, "p13a"].sum())
    non_alcohol_n = n_total - alcohol_n

    death_rate_alc = (alcohol_deaths / alcohol_n * 10000) if alcohol_n else 0.0
    death_rate_non = (non_alcohol_deaths / non_alcohol_n * 10000) if non_alcohol_n else 0.0

    top_region = table.iloc[0]

    # --- layout ---
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 0.9, 0.9])

    # Graph (render again into the PDF; fig.pdf is also saved separately by doc.py)
    ax0 = fig.add_subplot(gs[0, 0])
    img = plt.imread(fig_path) if fig_path.suffix.lower() in {".png", ".jpg", ".jpeg"} else None
    if img is not None:
        ax0.imshow(img)
        ax0.axis("off")
    else:
        # If fig_path is vector (pdf), we rebuild the plot directly for PDF consistency.
        # This keeps the report self-contained without needing PDF embedding.
        plt.close(fig)
        fig = plt.figure(figsize=(8.27, 11.69))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 0.9, 0.9])
        ax0 = fig.add_subplot(gs[0, 0])
        # Recreate plot directly (same as _make_fig, but to an existing Axes).
        labels = _road_type_labels()
        d = df.copy()
        d["road"] = d["p36"].map(labels).fillna(d["p36"].astype(str).radd("p36="))
        d["alcohol"] = np.where(d["p11"] >= 4, "alkohol (p11>=4)", "bez alkoholu (p11<4)")
        d["death"] = (d["p13a"] > 0).astype(int)
        grp = (
            d.groupby(["road", "alcohol"])  # type: ignore[call-arg]
            .agg(accidents=("p1", "size"), deaths=("death", "sum"))
            .reset_index()
        )
        grp["death_rate_per_10k"] = (grp["deaths"] / grp["accidents"]) * 10000
        order = [labels[k] for k in [1, 2, 3, 6, 7, 8] if k in labels]
        grp["road"] = pd.Categorical(grp["road"], categories=order, ordered=True)
        grp = grp.sort_values("road")
        piv = grp.pivot(index="road", columns="alcohol", values="death_rate_per_10k").fillna(0)
        piv.plot(kind="bar", ax=ax0, width=0.8, color=["#0072B2", "#D55E00"])
        ax0.set_title("Úmrtnost nehod dle typu komunikace (2023–2024)")
        ax0.set_xlabel("Typ komunikace (p36)")
        ax0.set_ylabel("Nehody s úmrtím na 10 000 nehod")
        ax0.grid(axis="y", alpha=0.3)
        ax0.legend(title="Skupina")

    # Table (top 10 regions by death rate)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axis("off")

    t = table.head(10).copy()
    t["deaths"] = t["deaths"].round(0).astype(int)
    t["death_rate_per_10k"] = t["death_rate_per_10k"].round(2)
    t["alcohol_share"] = (t["alcohol_share"] * 100).round(1)

    col_labels = [
        "Region",
        "Nehod",
        "Úmrtí",
        "Úmrtí / 10k",
        "Alkohol (%)",
    ]
    cell_text = t[["region", "accidents", "deaths", "death_rate_per_10k", "alcohol_share"]].values.tolist()

    tab = ax1.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 1.3)
    ax1.set_title("Top 10 regionů dle úmrtnosti (2023–2024)", pad=10)

    # Text block (>= 10 sentences + includes computed values)
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.axis("off")

    text = (
        "Tato zpráva shrnuje vybrané ukazatele nehodovosti v letech 2023–2024. "
        f"V analyzovaném období evidujeme celkem {n_total} nehod. "
        f"Celkový počet usmrcených osob je {int(deaths_total)}. "
        f"Nehod s významnou mírou alkoholu (p11>=4) bylo {alcohol_n}, což je {alcohol_share:.2%} všech nehod. "
        "Alkohol ovšem nesouvisí jen s četností, ale i se závažností následků. "
        f"Úmrtnost u nehod s alkoholem je {death_rate_alc:.2f} na 10 000 nehod, zatímco bez alkoholu {death_rate_non:.2f} na 10 000. "
        "Rozdíly mezi typy komunikací (p36) ukazují, že prostředí a rychlosti na pozemních komunikacích mění riziko fatálních následků. "
        "V grafu porovnáváme úmrtnost podle typu komunikace zvlášť pro nehody s alkoholem a bez alkoholu. "
        "Tabulka doplňuje pohled o regionální rozdíly a umožňuje rychle identifikovat oblasti s vyšší úmrtností. "
        f"Nejvyšší úmrtnost v tabulce má region {top_region['region']} ({top_region['death_rate_per_10k']:.2f} úmrtí na 10 000 nehod). "
        "Výsledky je vhodné interpretovat s ohledem na intenzitu provozu a strukturu komunikací v jednotlivých regionech."
    )

    ax2.text(0.0, 1.0, text, va="top", ha="left", fontsize=10, wrap=True)

    fig.suptitle("IZV 2025 – Vlastní analýza: Alkohol, typ komunikace a úmrtnost", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data_dir = Path(__file__).resolve().parent

    acc = _load_accidents(data_dir)

    # Focus on the two years highlighted in the geo task.
    df = acc.loc[acc["__year"].isin([2023, 2024])].copy()

    # --- outputs for grading ---
    table = _compute_region_table(df)

    # Figure used in report
    fig_path = data_dir / "fig.pdf"
    _make_fig(df, fig_path)

    # One-page report
    doc_path = data_dir / "doc.pdf"
    _make_doc_pdf(df, table, fig_path, doc_path)

    # --- STDOUT: table data ---
    print("region\taccidents\tdeaths\tdeath_rate_per_10k\talcohol_share")
    for _, r in table.head(10).iterrows():
        print(
            f"{r['region']}\t{int(r['accidents'])}\t{int(round(r['deaths']))}\t"
            f"{float(r['death_rate_per_10k']):.2f}\t{float(r['alcohol_share']):.4f}"
        )

    # --- STDOUT: values used in the text ---
    n_total = int(len(df))
    deaths_total = int(round(float(df["p13a"].sum())))
    alcohol_n = int((df["p11"] >= 4).sum())
    alcohol_share = alcohol_n / n_total if n_total else 0.0

    print("\nnehod celkem (2023–2024):", n_total)
    print("usmrcených celkem (2023–2024):", deaths_total)
    print("nehod s alkoholem (p11>=4):", alcohol_n)
    print("podíl nehod s alkoholem:", f"{alcohol_share:.2%}")


if __name__ == "__main__":
    main()
