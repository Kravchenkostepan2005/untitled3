#!/usr/bin/env python3
# coding=utf-8
"""
Analýza dat nehodovosti Policie ČR.

Skript načítá, zpracovává a vizualizuje data o nehodovosti v České republice
v letech 2023-2025. Vytváří tři typy grafů pro analýzu nehod.
"""

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile
import os
import requests
import io

STATE_ORDER = [
    "Únava, spánek",
    "Onemocnění",
    "Pod vlivem alkoholu",
    "Pod vlivem jiných látek",
    "Invalidita",
    "Sebevražda",
    "Řidič zemřel",
]
STATE_MAP = {
    3: "Únava, spánek",
    4: "Onemocnění",
    5: "Pod vlivem alkoholu",
    6: "Pod vlivem jiných látek",
    7: "Invalidita",
    8: "Sebevražda",
    9: "Řidič zemřel",
}
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
INJURY_MAP = {
    1: "Usmrcení",
    2: "Těžké zranění",
    3: "Lehké zranění",
    4: "Bez zranění",
}
CONSEQUENCE_ORDER = ["Bez zranění", "Těžké zranění", "Lehké zranění", "Usmrcení"]
YEAR_COLORS = {2023: "#1f77b4", 2024: "#ff7f0e", 2025: "#2ca02c"}
CONDITION_MAP = {
    1: "neztížené",
    2: "mlha",
    3: "na počátku deště",
    4: "déšť",
    5: "sněžení",
    6: "náledí",
    7: "vítr",
}
CONDITION_ORDER = [
    "déšť",
    "mlha",
    "na počátku deště",
    "neztížené",
    "náledí",
    "sněžení",
    "vítr",
    "jiné",
]
PREFERRED_REGIONS = ["JHM", "MSK", "OLK", "ZLK"]


def load_data(filename: str, ds: str) -> pd.DataFrame:
    """Načte data ze ZIP souboru a spojí data pro roky 2023-2025."""
    zip_content = _download_zip(filename)
    frames = []
    with zipfile.ZipFile(zip_content, 'r') as zip_ref:
        for year in (2023, 2024, 2025):
            df_year = _load_year_data(zip_ref, year, ds)
            if df_year is not None:
                frames.append(df_year.assign(year=year))
    if not frames:
        raise ValueError(f"Data pro dataset '{ds}' nebyla nalezena")
    result = pd.concat(frames, ignore_index=True)
    return result.loc[:, ~result.columns.str.contains('^Unnamed')]


def _download_zip(filename: str):
    """Stáhne ZIP soubor z URL nebo použije lokální."""
    try:
        response = requests.get(filename, timeout=30)
        response.raise_for_status()
        return io.BytesIO(response.content)
    except Exception:
        if os.path.exists("data_23_25.zip"):
            return "data_23_25.zip"
        raise


def _load_year_data(zip_ref, year: int, ds: str):
    """Načte data pro konkrétní rok ze ZIP archivu."""
    for name in (f"{year}/I{ds}.xls", f"{year}/{ds}.xls"):
        if name in zip_ref.namelist():
            with zip_ref.open(name) as file:
                tables = pd.read_html(io.BytesIO(file.read()), encoding='cp1250', decimal=',', thousands=' ')
                return tables[0]
    return None


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Zpracuje a vyčistí data o nehodách."""
    result = df.copy()
    result['date'] = pd.to_datetime(result['p2a'], format='%d.%m.%Y', errors='coerce')
    result['region'] = pd.to_numeric(result['p4a'], errors='coerce').map(REGION_MAP)
    result = result.drop_duplicates(subset=['p1'])
    if verbose:
        size_mb = result.memory_usage(deep=True).sum() / 1e6
        print(f"new_size={size_mb:.1f} MB")
    return result


def plot_state(df: pd.DataFrame, df_vehicles: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """Vytvoří graf počtu nehod podle stavu řidiče."""
    merged = pd.merge(df, df_vehicles, on='p1', how='inner')
    filtered = merged[merged['p57'].between(3, 9)].copy()
    filtered['driver_state'] = filtered['p57'].map(STATE_MAP)
    plot_data = filtered.groupby(['region', 'driver_state']).size().reset_index(name='count')
    _create_state_plot(plot_data, fig_location, show_figure)


def _create_state_plot(data: pd.DataFrame, fig_location: str, show_figure: bool):
    """Vytvoří graf stavu řidiče."""
    regions = sorted(data['region'].dropna().unique())
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        if idx < len(STATE_ORDER):
            _plot_state_bars(ax, data, STATE_ORDER[idx], regions)
        else:
            ax.set_visible(False)
    plt.suptitle('Počty nehod podle stavu řidiče v jednotlivých krajích', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    _save_or_show(fig, fig_location, show_figure)


def _plot_state_bars(ax, data: pd.DataFrame, state: str, regions):
    """Vykreslí sloupcový graf pro jeden stav řidiče."""
    state_data = data[data['driver_state'] == state].set_index('region')['count']
    counts = state_data.reindex(regions, fill_value=0).tolist()
    colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
    bars = ax.bar(range(len(regions)), counts, color=colors, edgecolor='black')
    ax.set_title(state, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Kraj', fontsize=10)
    ax.set_ylabel('Počet nehod', fontsize=10)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    _annotate_bars(ax, bars)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#f5f5f5')


def _annotate_bars(ax, bars):
    """Popíše hodnoty nad sloupci."""
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)


def plot_alcohol(df: pd.DataFrame, df_consequences: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """Vytvoří graf následků nehod pod vlivem alkoholu."""
    merged = pd.merge(df, df_consequences, on='p1', suffixes=('', '_cons'))
    filtered = merged[(merged['p11'] >= 3) & (merged['date'] < '2025-11-01')].copy()
    filtered['year'] = filtered['date'].dt.year
    filtered['injury_type'] = filtered['p59g'].map(INJURY_MAP)
    plot_data = filtered.groupby(['region', 'year', 'injury_type']).size().reset_index(name='count')
    _create_alcohol_plot(plot_data, fig_location, show_figure)


def _create_alcohol_plot(data: pd.DataFrame, fig_location: str, show_figure: bool):
    """Vytvoří graf alkoholu a následků."""
    regions = sorted(data['region'].dropna().unique())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        if idx < len(CONSEQUENCE_ORDER):
            _plot_alcohol_bars(ax, data, CONSEQUENCE_ORDER[idx], regions)
        else:
            ax.set_visible(False)
    plt.suptitle("Následky nehod pod vlivem alkoholu v krajích a letech", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    _save_or_show(fig, fig_location, show_figure)


def _plot_alcohol_bars(ax, data: pd.DataFrame, consequence: str, regions):
    """Vykreslí sloupcový graf pro jeden typ následků."""
    cons_data = data[data['injury_type'] == consequence]
    table = cons_data.pivot_table(index='region', columns='year', values='count', fill_value=0).reindex(regions, fill_value=0)
    x_positions, _ = _draw_year_bars(ax, table, regions)
    ax.set_title(f'Následky nehody: {consequence}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Kraj', fontsize=10); ax.set_ylabel('Počet nehod pod vlivem', fontsize=10)
    ax.set_xticks(x_positions); ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=9)
    if not cons_data.empty:
        ax.set_ylim(0, cons_data['count'].max() * 1.15)
    ax.grid(True, alpha=0.3, axis='y'); ax.set_facecolor('#f5f5f5')


def _draw_year_bars(ax, table: pd.DataFrame, regions):
    """Vykreslí sloupce pro jednotlivé roky."""
    years = sorted(table.columns) if len(table.columns) else []
    x_positions = np.arange(len(regions))
    for i, year in enumerate(years):
        ax.bar(x_positions + (i - len(years) / 2 + 0.5) * 0.25, table[year].values, width=0.25,
               color=YEAR_COLORS.get(year, '#888888'), edgecolor='black', linewidth=0.5, label=str(year))
    if years:
        ax.legend(title='Rok', fontsize=9, title_fontsize=10, loc='upper right')
    return x_positions, years


def plot_conditions(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """Vytvoří graf počtu nehod podle povětrnostních podmínek."""
    regions = _select_four_regions(df)
    filtered = df[df['region'].isin(regions)].copy()
    filtered['condition'] = filtered['p18'].map(CONDITION_MAP).fillna("jiné")
    monthly = _prepare_monthly_data(filtered)
    _create_conditions_plot(monthly, regions, fig_location, show_figure)


def _select_four_regions(df: pd.DataFrame):
    """Vybere 4 kraje pro analýzu."""
    available = sorted(df['region'].dropna().unique())
    selected = [r for r in PREFERRED_REGIONS if r in available][:4]
    for region in available:
        if len(selected) >= 4:
            break
        if region not in selected:
            selected.append(region)
    return selected


def _prepare_monthly_data(df: pd.DataFrame):
    """Připraví měsíční agregovaná data."""
    df = df.copy()
    df['month_year'] = df['date'].dt.to_period('M')
    monthly = df.groupby(['region', 'month_year', 'condition']).size().reset_index(name='count')
    monthly['date'] = monthly['month_year'].dt.to_timestamp()
    mask = (monthly['date'] >= '2023-01-01') & (monthly['date'] <= '2025-01-01')
    return monthly[mask]


def _create_conditions_plot(data: pd.DataFrame, regions, fig_location: str, show_figure: bool):
    """Vytvoří graf povětrnostních podmínek."""
    n_cols = 2
    n_rows = (len(regions) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), squeeze=False)
    color_dict = dict(zip(CONDITION_ORDER, plt.cm.Set2(np.linspace(0, 1, len(CONDITION_ORDER)))))
    for idx, region in enumerate(regions):
        row, col = divmod(idx, n_cols)
        _plot_conditions_lines(axes[row, col], data, region, color_dict)
    for idx in range(len(regions), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    _add_conditions_legend(fig, color_dict)
    plt.suptitle('Počty nehod podle povětrnostních podmínek ve vybraných krajích', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])
    _save_or_show(fig, fig_location, show_figure)


def _plot_conditions_lines(ax, data: pd.DataFrame, region: str, color_dict):
    """Vykreslí čárový graf pro jeden kraj."""
    region_data = data[data['region'] == region]
    if region_data.empty:
        ax.text(0.5, 0.5, 'Žádná data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Kraj: {region}', fontsize=14, fontweight='bold', pad=10)
        ax.set_facecolor('#f5f5f5')
        return
    for condition in CONDITION_ORDER:
        subset = region_data[region_data['condition'] == condition].sort_values('date')
        if not subset.empty:
            ax.plot(subset['date'], subset['count'], label=condition, color=color_dict[condition],
                    linewidth=2, marker='o', markersize=4, markevery=3)
    _style_condition_axes(ax, region, region_data)


def _style_condition_axes(ax, region: str, region_data: pd.DataFrame):
    """Nastaví vzhled os pro graf podmínek."""
    ax.set_title(f'Kraj: {region}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Datum', fontsize=11); ax.set_ylabel('Počet nehod', fontsize=11)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%y'))
    dates = pd.date_range(start='2023-01-01', end='2025-01-01', freq='3MS')
    ax.set_xticks(dates); ax.set_xticklabels([d.strftime('%m/%y') for d in dates], rotation=45, ha='right', fontsize=10)
    if len(region_data) > 0:
        y_max = max(600, np.ceil(region_data['count'].max() / 100) * 100)
        ax.set_yticks(np.arange(0, y_max + 100, 100)); ax.set_ylim(0, y_max)
    ax.set_xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2025-01-01'))
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5); ax.set_facecolor('#f9f9f9')


def _add_conditions_legend(fig, color_dict):
    """Přidá legendu pro graf podmínek."""
    handles = [plt.Line2D([0], [0], color=color_dict[c], lw=2, marker='o', markersize=6)
               for c in CONDITION_ORDER if c in color_dict]
    fig.legend(handles, CONDITION_ORDER, loc='lower center', ncol=len(CONDITION_ORDER),
               fontsize=10, bbox_to_anchor=(0.5, 0.02), frameon=True)


def _save_or_show(fig, fig_location: str, show_figure: bool):
    """Uloží nebo zobrazí graf."""
    if fig_location:
        plt.savefig(fig_location, dpi=300, bbox_inches='tight')
    if show_figure:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    data_url = "https://ehw.fit.vutbr.cz/izv/data_23_25.zip"
    try:
        df_nehody = load_data(data_url, "nehody")
        df_nasledky = load_data(data_url, "nasledky")
        df_vozidla = load_data(data_url, "Vozidla")
        df_parsed = parse_data(df_nehody, verbose=True)
        plot_state(df_parsed, df_vozidla, "01_state.png")
        plot_alcohol(df_parsed, df_nasledky, "02_alcohol.png")
        plot_conditions(df_parsed, "03_conditions.png")
        print("Analýza dokončena úspěšně!")
    except Exception as e:
        print(f"Chyba: {e}")
        import traceback
        traceback.print_exc()
