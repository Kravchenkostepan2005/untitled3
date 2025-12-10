#!/usr/bin/python3.13
# coding=utf-8
# %%%
import os
import re
import warnings

import contextily
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, box, MultiPoint
from shapely.ops import unary_union
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def make_geo(df_accidents: pd.DataFrame, df_locations: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
        Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani.
        Pozor na mozne prohozeni d a e!

        Args:
            df_accidents: DataFrame ze souboru accidents.pkl.gz
            df_locations: DataFrame ze souboru locations.pkl.gz

        Returns:
            geopandas.GeoDataFrame s geometrií bodů a sloupci z obou DataFrame
        """
    # Spojení dat pomocí klíče p1 (ID nehody)
    df = pd.merge(df_accidents, df_locations[['p1', 'd', 'e']], on='p1', how='inner')

    # Odstranění nehod s neznámými souřadnicemi
    df = df[(df['d'] != 0) & (df['e'] != 0)].copy()

    # Prohození souřadnic, pokud d < e (oprava prohozených os)
    swap_mask = df['d'] < df['e']
    df.loc[swap_mask, ['d', 'e']] = df.loc[swap_mask, ['e', 'd']].values

    # Vytvoření GeoDataFrame s geometrií bodů
    gdf = geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df['d'], df['e']),
        crs='EPSG:5514'  # S-JTSK pro Českou republiku
    )

    # Přidání sloupce s rokem z data nehody (p2a formát DD.MM.YYYY)
    if 'p2a' in gdf.columns:
        # Formát je DD.MM.YYYY
        try:
            gdf['date'] = pd.to_datetime(gdf['p2a'], format='%d.%m.%Y', errors='coerce')
            gdf['year'] = gdf['date'].dt.year
        except Exception as exc:
            print(f"Chyba při převodu data: {exc}")

            # Pokud selže, zkusíme extrahovat rok pomocí regulárního výrazu
            def extract_year(x):
                if pd.isna(x):
                    return np.nan
                str_val = str(x)
                match = re.search(r'(\d{4})', str_val)
                if match:
                    try:
                        return int(match.group(1))
                    except Exception:
                        return np.nan
                return np.nan

            gdf['year'] = gdf['p2a'].apply(extract_year)

    # Debug: výpis prvních 5 let pro kontrolu
    if 'year' in gdf.columns:
        print(f"Prvních 5 let z extrahovaných dat: {gdf['year'].head().tolist()}")

    return gdf


def get_kraj_boundary(gdf: geopandas.GeoDataFrame, kraj_kod: int) -> geopandas.GeoDataFrame | None:
    """
    Vytvoří přibližné hranice kraje z dostupných dat
    """
    # Filtrace bodů v daném kraji
    kraj_points = gdf[gdf['p5a'] == kraj_kod].copy()

    if len(kraj_points) == 0:
        return None

    # Převod do jednotného CRS (5514)
    kraj_points = kraj_points.to_crs(epsg=5514)

    # Vytvoření obálky bodů v kraji
    point_list = [geom for geom in kraj_points.geometry if geom is not None]
    if len(point_list) == 0:
        return None

    multipoint = MultiPoint(point_list)
    hull = multipoint.convex_hull
    if hull.is_empty:
        return None

    # Rozšíření o 5 km (5000 metrů) kolem konvexní obálky
    polygon = hull.buffer(5000)

    # Vytvoření GeoDataFrame
    kraj_boundary = geopandas.GeoDataFrame(
        {'kraj': [kraj_kod], 'geometry': [polygon]},
        crs='EPSG:5514'
    )

    # Převod do Web Mercator pro zobrazení
    kraj_boundary = kraj_boundary.to_crs(epsg=3857)

    return kraj_boundary


def _apply_bounds(ax: plt.Axes, bounds: np.ndarray) -> None:
    """Nastaví limity os podle hranic kraje a zabrání autoscale."""
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect('equal', adjustable='box')
    ax.set_autoscale_on(False)


def _add_basemap(ax: plt.Axes, bounds: np.ndarray) -> None:
    """Přidá podkladovou mapu bez resetu rozsahu os."""
    min_x, min_y, max_x, max_y = bounds
    _apply_bounds(ax, bounds)
    try:
        img, ext = contextily.bounds2img(
            min_x,
            min_y,
            max_x,
            max_y,
            epsg=3857,
            source=contextily.providers.CartoDB.Positron,
        )
        ax.imshow(img, extent=ext, origin='upper')
    except Exception:
        try:
            contextily.add_basemap(
                ax,
                source=contextily.providers.CartoDB.Positron,
                crs='EPSG:3857',
                reset_extent=False,
            )
        except TypeError:
            contextily.add_basemap(
                ax,
                source=contextily.providers.CartoDB.Positron,
                crs='EPSG:3857',
            )
    _apply_bounds(ax, bounds)


def _mask_outside_kraj(ax: plt.Axes, kraj_boundary: geopandas.GeoDataFrame) -> None:
    """Zabarví oblast mimo vybraný kraj na bílo, aby zůstal viditelný jen on."""
    if kraj_boundary is None or kraj_boundary.empty:
        return

    union = unary_union(kraj_boundary.geometry)
    if union.is_empty:
        return

    min_x, min_y, max_x, max_y = kraj_boundary.total_bounds
    outer = box(min_x - 1000, min_y - 1000, max_x + 1000, max_y + 1000)
    mask_geom = outer.difference(union.buffer(0))

    if mask_geom.is_empty:
        return

    geopandas.GeoSeries([mask_geom], crs=kraj_boundary.crs).plot(
        ax=ax,
        color='white',
        edgecolor='none',
        alpha=1.0,
        zorder=5,
    )


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str | None = None,
             show_figure: bool = False):
    """
        Vykresleni grafu s nehodami se zvěří pro roky 2023-2024.
        Používá kraj 2 (Středočeský kraj) - zobrazí se POUZE tento kraj.
        """
    KRAJ_KOD = 2
    print(f"\n=== Vytváření grafu s nehodami se zvěří pro kraj {KRAJ_KOD} ===")

    # Získání hranic kraje 2
    kraj_boundary = get_kraj_boundary(gdf, KRAJ_KOD)
    if kraj_boundary is None:
        print(f"⚠ Nelze získat hranice kraje {KRAJ_KOD}")
        return

    # Filtrace a příprava dat (váš stávající kód zůstává)
    gdf_kraj = gdf[gdf['p5a'] == KRAJ_KOD].copy()
    gdf_zver = gdf_kraj[gdf_kraj['p10'] == 4].copy()
    gdf_2023 = gdf_zver[gdf_zver['year'] == 2023].copy()
    gdf_2024 = gdf_zver[gdf_zver['year'] == 2024].copy()

    kraj_boundary = kraj_boundary.to_crs(epsg=3857)
    if len(gdf_2023) > 0:
        gdf_2023 = gdf_2023.to_crs(epsg=3857)
    if len(gdf_2024) > 0:
        gdf_2024 = gdf_2024.to_crs(epsg=3857)

    # Vytvoření grafu
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bounds = kraj_boundary.total_bounds

    # ---- Podgraf pro rok 2023 ----
    _add_basemap(ax1, bounds)
    _mask_outside_kraj(ax1, kraj_boundary)

    kraj_boundary.boundary.plot(ax=ax1, color='black', linewidth=2)
    if len(gdf_2023) > 0:
        gdf_2023.plot(ax=ax1, markersize=15, color='red', alpha=0.7,
                       edgecolor='darkred', linewidth=0.5)

    ax1.set_title(f'Nehody způsobené zvěří - Kraj {KRAJ_KOD} - 2023\n({len(gdf_2023)} nehod)')
    ax1.set_xlabel('Východní délka [m]')
    ax1.set_ylabel('Severní šířka [m]')
    ax1.grid(True, alpha=0.3, linestyle='--')
    _apply_bounds(ax1, bounds)

    # ---- Podgraf pro rok 2024 ----
    _add_basemap(ax2, bounds)
    _mask_outside_kraj(ax2, kraj_boundary)

    kraj_boundary.boundary.plot(ax=ax2, color='black', linewidth=2)
    if len(gdf_2024) > 0:
        gdf_2024.plot(ax=ax2, markersize=15, color='blue', alpha=0.7,
                       edgecolor='darkblue', linewidth=0.5)

    ax2.set_title(f'Nehody způsobené zvěří - Kraj {KRAJ_KOD} - 2024\n({len(gdf_2024)} nehod)')
    ax2.set_xlabel('Východní délka [m]')
    ax2.set_ylabel('Severní šířka [m]')
    ax2.grid(True, alpha=0.3, linestyle='--')
    _apply_bounds(ax2, bounds)

    plt.tight_layout()
    if fig_location:
        plt.savefig(fig_location, dpi=300, bbox_inches='tight')
        print(f"✓ Graf uložen do {fig_location}")
    if show_figure:
        plt.show()
    else:
        plt.close(fig)



def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str | None = None,
                 show_figure: bool = False):
    """
            Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru.

            Používá kraj 2 (Středočeský kraj) - zobrazí se POUZE tento kraj.

            Args:
                gdf: GeoDataFrame vytvořený funkcí make_geo
                fig_location: Cesta pro uložení grafu
                show_figure: Zda zobrazit graf
            """
    # Používáme kraj 2
    KRAJ_KOD = 2

    print(f"\n=== Vytváření cluster grafu s nehodami s alkoholem pro kraj {KRAJ_KOD} ===")

    # Získání hranic kraje 2
    kraj_boundary = get_kraj_boundary(gdf, KRAJ_KOD)

    if kraj_boundary is None:
        print(f"⚠ Nelze získat hranice kraje {KRAJ_KOD}")
        return

    # Filtrace na kraj 2 a nehody s alkoholem (p11 >= 4)
    gdf_kraj = gdf[gdf['p5a'] == KRAJ_KOD].copy()
    gdf_alkohol = gdf_kraj[gdf_kraj['p11'] >= 4].copy()

    print(f"Celkem nehod v kraji {KRAJ_KOD}: {len(gdf_kraj)}")
    print(f"Nehody s alkoholem (p11 >= 4) v kraji {KRAJ_KOD}: {len(gdf_alkohol)}")

    kraj_boundary = kraj_boundary.to_crs(epsg=3857)
    bounds = kraj_boundary.total_bounds

    if len(gdf_alkohol) == 0:
        print(f"⚠ V kraji {KRAJ_KOD} nejsou žádné nehody s alkoholem (p11 >= 4)")
        print("Vytvářím prázdný graf...")

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        _add_basemap(ax, bounds)
        _mask_outside_kraj(ax, kraj_boundary)

        kraj_boundary.boundary.plot(ax=ax, color='black', linewidth=2)
        kraj_boundary.plot(ax=ax, color='lightgray', alpha=0.3, edgecolor='black', linewidth=1)

        ax.text(0.5, 0.5, f'Žádné nehody s alkoholem (p11 >= 4)\nv kraji {KRAJ_KOD}',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Nehody s alkoholem (p11 >= 4) v kraji {KRAJ_KOD}')
        ax.set_xlabel('Východní délka [m]')
        ax.set_ylabel('Severní šířka [m]')
        _apply_bounds(ax, bounds)
        plt.tight_layout()

        if fig_location:
            plt.savefig(fig_location, dpi=300, bbox_inches='tight')
            print(f"✓ Prázdný graf uložen do {fig_location}")

        if show_figure:
            plt.show()
        else:
            plt.close(fig)
        return

    gdf_alkohol = gdf_alkohol.to_crs(epsg=3857)

    if len(gdf_alkohol) < 10:
        print(f"⚠ V kraji {KRAJ_KOD} je příliš málo nehod s alkoholem ({len(gdf_alkohol)}), vytvářím jednoduchý graf")
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        _add_basemap(ax, bounds)
        _mask_outside_kraj(ax, kraj_boundary)

        kraj_boundary.boundary.plot(ax=ax, color='black', linewidth=2)
        kraj_boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=1, alpha=0.5)

        gdf_alkohol.plot(ax=ax, markersize=20, color='red', alpha=0.7,
                         edgecolor='black', linewidth=0.5)

        ax.set_title(f'Nehody s alkoholem (p11 >= 4) v kraji {KRAJ_KOD}\n({len(gdf_alkohol)} nehod)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Východní délka [m]')
        ax.set_ylabel('Severní šířka [m]')
        ax.grid(True, alpha=0.3, linestyle='--')
        _apply_bounds(ax, bounds)
        plt.tight_layout()

        if fig_location:
            plt.savefig(fig_location, dpi=300, bbox_inches='tight')
            print(f"✓ Jednoduchý graf uložen do {fig_location}")

        if show_figure:
            plt.show()
        else:
            plt.close(fig)
        return

    # Extrakce souřadnic pro shlukování
    coords = np.array(list(zip(gdf_alkohol.geometry.x, gdf_alkohol.geometry.y)))

    # Normalizace souřadnic pro lepší shlukování
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    # Volba počtu clusterů
    n_clusters = min(8, max(2, len(gdf_alkohol) // 10))
    print(f"Počet clusterů: {n_clusters}")

    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(coords_scaled)

    gdf_alkohol['cluster'] = cluster_labels

    # Vytvoření konvexních obalů pro clustery
    polygons = []
    cluster_info = []

    for cluster_id in range(n_clusters):
        cluster_points = gdf_alkohol[gdf_alkohol['cluster'] == cluster_id]

        if len(cluster_points) >= 3:
            try:
                points = np.array([[p.x, p.y] for p in cluster_points.geometry])
                hull = ConvexHull(points)

                hull_points = points[hull.vertices]
                polygon = Polygon(hull_points)

                polygons.append(polygon)
                cluster_info.append({
                    'cluster': cluster_id,
                    'count': len(cluster_points),
                    'centroid': polygon.centroid
                })
            except Exception:
                if len(points) > 0:
                    min_x, min_y = points.min(axis=0)
                    max_x, max_y = points.max(axis=0)
                    polygon = Polygon([
                        (min_x, min_y), (max_x, min_y),
                        (max_x, max_y), (min_x, max_y)
                    ])
                    polygons.append(polygon)
                    cluster_info.append({
                        'cluster': cluster_id,
                        'count': len(cluster_points),
                        'centroid': polygon.centroid
                    })
        elif len(cluster_points) > 0:
            points = np.array([[p.x, p.y] for p in cluster_points.geometry])
            min_x, min_y = points.min(axis=0)
            max_x, max_y = points.max(axis=0)
            polygon = Polygon([
                (min_x, min_y), (max_x, min_y),
                (max_x, max_y), (min_x, max_y)
            ])
            polygons.append(polygon)
            cluster_info.append({
                'cluster': cluster_id,
                'count': len(cluster_points),
                'centroid': polygon.centroid
            })

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    _add_basemap(ax, bounds)
    _mask_outside_kraj(ax, kraj_boundary)

    kraj_boundary.boundary.plot(ax=ax, color='black', linewidth=2)
    kraj_boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=1, alpha=0.5)

    if len(gdf_alkohol) > 0:
        ax.scatter(gdf_alkohol.geometry.x, gdf_alkohol.geometry.y,
                   c=gdf_alkohol['cluster'], cmap='tab20',
                   s=25, alpha=0.7, edgecolors='black', linewidth=0.5)

    if len(polygons) > 0:
        counts = [info['count'] for info in cluster_info]
        norm = plt.Normalize(vmin=min(counts), vmax=max(counts))
        cmap = plt.cm.YlOrRd

        for i, polygon in enumerate(polygons):
            color = cmap(norm(cluster_info[i]['count']))
            geopandas.GeoSeries([polygon]).plot(ax=ax, alpha=0.3,
                                                color=color,
                                                edgecolor='darkred',
                                                linewidth=1.5)

            centroid = cluster_info[i]['centroid']
            ax.text(centroid.x, centroid.y,
                    str(cluster_info[i]['count']),
                    fontsize=10, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8,
                              edgecolor='black'))

    ax.set_title(f'Nehody s alkoholem (p11 >= 4) v kraji {KRAJ_KOD}\n'
                 f'Shlukováno do {n_clusters} oblastí (Celkem: {len(gdf_alkohol)} nehod)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Východní délka [m]')
    ax.set_ylabel('Severní šířka [m]')
    ax.grid(True, alpha=0.3, linestyle='--')

    if len(cluster_info) > 1:
        counts = [info['count'] for info in cluster_info]
        norm = plt.Normalize(vmin=min(counts), vmax=max(counts))
        cmap = plt.cm.YlOrRd
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Počet nehod v oblasti', fontsize=10)

    _apply_bounds(ax, bounds)
    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location, dpi=300, bbox_inches='tight')
        print(f"✓ Cluster graf uložen do {fig_location}")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    print("Načítání dat...")
    df_accidents = pd.read_pickle("accidents.pkl.gz")
    df_locations = pd.read_pickle("locations.pkl.gz")

    print(f"Načteno nehod: {len(df_accidents)}")
    print(f"Načteno lokací: {len(df_locations)}")

    # Analýza sloupce p2a (datum)
    print("\n=== Analýza sloupce p2a (datum nehody) ===")
    print(f"Typ dat: {df_accidents['p2a'].dtype}")
    print(f"Prvních 5 hodnot: {df_accidents['p2a'].head().tolist()}")

    # Extrahujeme rok
    def extract_year_from_string(x):
        if pd.isna(x):
            return None
        str_val = str(x)
        match = re.search(r'(\d{4})', str_val)
        if match:
            return int(match.group(1))
        return None

    years_in_data = df_accidents['p2a'].apply(extract_year_from_string).dropna().unique()
    print(f"Roky v datech (odhad): {sorted(years_in_data)}")

    gdf = make_geo(df_accidents, df_locations)

    # Základní informace o datech
    print(f"\n=== Základní informace ===")
    print(f"Celkový počet nehod: {len(gdf)}")

    if 'year' in gdf.columns:
        roky = sorted(gdf['year'].dropna().unique())
        print(f"Roky v datech: {roky}")
        for rok in roky:
            pocet = len(gdf[gdf['year'] == rok])
            print(f"  {rok}: {pocet} nehod")
    else:
        print("⚠ Sloupec 'year' nebyl vytvořen")

    # Přehled nehod podle krajů
    print("\n=== Nehody podle krajů ===")
    kraj_counts = gdf['p5a'].value_counts().sort_index()
    for kraj, pocet in kraj_counts.items():
        print(f"Kraj {kraj}: {pocet} nehod")

    # Detailní informace pro kraj 2
    print(f"\n=== Detailní informace pro kraj 2 ===")
    gdf_kraj2 = gdf[gdf['p5a'] == 2].copy()
    print(f"Celkem nehod v kraji 2: {len(gdf_kraj2)}")

    # Nehody se zvěří v kraji 2
    zver_kraj2 = gdf_kraj2[gdf_kraj2['p10'] == 4]
    print(f"Nehody se zvěří v kraji 2: {len(zver_kraj2)}")

    if 'year' in zver_kraj2.columns:
        zver_roky = sorted(zver_kraj2['year'].dropna().unique())
        print(f"Roky s nehodami se zvěří v kraji 2: {zver_roky}")
        for rok in zver_roky:
            pocet = len(zver_kraj2[zver_kraj2['year'] == rok])
            print(f"  {rok}: {pocet} nehod")

    # Nehody s alkoholem v kraji 2
    alkohol_kraj2 = gdf_kraj2[gdf_kraj2['p11'] >= 4]
    print(f"Nehody s alkoholem (p11 >= 4) v kraji 2: {len(alkohol_kraj2)}")

    print("\n=== Vytváření geografického grafu (zvěř) ===")
    plot_geo(gdf, "geo1.png", True)

    print("\n=== Vytváření cluster grafu (alkohol) ===")
    plot_cluster(gdf, "geo2.png", True)

    if os.path.exists("geo1.png"):
        print("\n✓ Soubor geo1.png byl vytvořen")
    if os.path.exists("geo2.png"):
        print("✓ Soubor geo2.png byl vytvořen")

    print("\n=== Hotovo ===")

    assert os.path.exists("geo1.png")
    assert os.path.exists("geo2.png")
