"""
==============================================================================
Distillation Column — Exploratory Data Analysis & Feature Engineering
==============================================================================
Author : Chemical Process + ML Engineering
Dataset: Binary distillation column (benzene/toluene proxy, 1 000 rows)

Pipeline
--------
1. Load dataset & audit quality
2. Missing-value analysis
3. Descriptive statistics
4. Feature engineering  (relative volatility, reflux-to-feed ratio, etc.)
5. Min-Max normalisation
6. Correlation heatmaps  (raw features  +  engineered features)

Dependencies
------------
    pip install pandas numpy matplotlib seaborn scikit-learn
==============================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from   sklearn.preprocessing import MinMaxScaler

# ── Plot aesthetics ───────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE   = "coolwarm"
FIG_DPI   = 150
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 1 — LOAD DATA
# =============================================================================

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the distillation column CSV into a DataFrame and print a quick audit.

    Parameters
    ----------
    path : str
        File-system path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    df = pd.read_csv(path)

    print("=" * 70)
    print("DATASET LOADED")
    print("=" * 70)
    print(f"  Shape   : {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
    print(f"  Columns : {list(df.columns)}\n")
    print(df.head(3).to_string(index=False))
    print()
    return df


# =============================================================================
# SECTION 2 — MISSING VALUE ANALYSIS
# =============================================================================

def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-column missing-value statistics and visualise them.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Summary table with count and percentage of nulls per column.
    """
    null_count  = df.isnull().sum()
    null_pct    = (null_count / len(df) * 100).round(2)
    dtype_info  = df.dtypes

    summary = pd.DataFrame({
        "dtype"       : dtype_info,
        "null_count"  : null_count,
        "null_pct_%"  : null_pct,
    })

    print("=" * 70)
    print("MISSING VALUE AUDIT")
    print("=" * 70)
    print(summary.to_string())

    total_missing = null_count.sum()
    if total_missing == 0:
        print("\n  ✓  No missing values detected — dataset is complete.\n")
    else:
        print(f"\n  ⚠  {total_missing} missing values found — consider imputation.\n")

        # Bar chart of null percentages (only when nulls exist)
        fig, ax = plt.subplots(figsize=(10, 4))
        null_pct[null_pct > 0].sort_values().plot.barh(ax=ax, color="#E07B54")
        ax.set_xlabel("Missing values (%)")
        ax.set_title("Missing Value Percentage by Column")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/missing_values.png", dpi=FIG_DPI)
        plt.close()

    return summary


# =============================================================================
# SECTION 3 — DESCRIPTIVE STATISTICS
# =============================================================================

def describe_dataset(df: pd.DataFrame) -> None:
    """
    Print extended descriptive statistics (mean, std, quartiles, skewness,
    kurtosis) for every numerical column.

    Parameters
    ----------
    df : pd.DataFrame
    """
    stats = df.describe().T
    stats["skewness"] = df.skew(numeric_only=True)
    stats["kurtosis"] = df.kurt(numeric_only=True)

    print("=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(stats.round(4).to_string())
    print()


# =============================================================================
# SECTION 4 — FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive physically motivated features from raw distillation variables.

    New features
    ~~~~~~~~~~~~
    relative_volatility_est
        Estimated α using the Fenske-style ratio of distillate to bottoms odds.
        α_est = [x_D / (1 - x_D)] / [x_B / (1 - x_B)]
        A higher value means easier separation.

    reflux_to_feed_ratio
        R / F  — normalises reflux effort against throughput.
        Captures how much internal reflux is used per unit of feed.

    separation_factor
        (x_D - z_F) / (z_F - x_B)  — measures enrichment symmetry.
        Values > 1 indicate a heavy rectifying burden relative to stripping.

    tray_utilisation
        N_theoretical_min / N_actual  (Fenske proxy).
        Values near 1.0 → column is operating close to its theoretical limit.
        log(α_est) / log(N_trays + 1) is used as a tractable proxy.

    energy_per_kmol
        Energy consumption [kW] per unit feed flow [kmol/h].
        A measure of specific energy intensity.

    pressure_temperature_index
        P [kPa] × T_feed [°C]  (interaction term).
        Captures combined thermodynamic load on the condenser/reboiler.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (must contain the standard column names).

    Returns
    -------
    pd.DataFrame
        Extended DataFrame with 6 additional engineered columns.
    """
    fe = df.copy()

    # ── 4.1  Relative volatility estimate  (from separation endpoints) ────────
    x_D  = fe["distillate_purity_molfrac"].clip(1e-4, 1 - 1e-4)
    x_B  = fe["bottoms_composition_molfrac"].clip(1e-4, 1 - 1e-4)
    odds_D = x_D  / (1 - x_D)
    odds_B = x_B  / (1 - x_B)
    fe["relative_volatility_est"] = (odds_D / odds_B).round(4)

    # ── 4.2  Reflux-to-feed ratio  ────────────────────────────────────────────
    fe["reflux_to_feed_ratio"] = (
        fe["reflux_ratio"] / fe["feed_flow_rate_kmolph"]
    ).round(6)

    # ── 4.3  Separation factor  ───────────────────────────────────────────────
    z_F  = fe["feed_composition_molfrac"].clip(1e-4, 1 - 1e-4)
    num  = (x_D - z_F).clip(1e-6)   # rectifying enrichment
    den  = (z_F - x_B).clip(1e-6)   # stripping enrichment
    fe["separation_factor"] = (num / den).round(4)

    # ── 4.4  Tray utilisation index  (log-ratio proxy) ───────────────────────
    # Uses ln(α_est) / ln(N_T + 1) as a dimensionless loading metric
    alpha_safe = fe["relative_volatility_est"].clip(1.01)
    fe["tray_utilisation"] = (
        np.log(alpha_safe) / np.log(fe["num_trays"] + 1)
    ).round(4)

    # ── 4.5  Specific energy intensity  ──────────────────────────────────────
    fe["energy_per_kmol"] = (
        fe["energy_consumption_kW"] / fe["feed_flow_rate_kmolph"]
    ).round(4)

    # ── 4.6  Pressure–temperature interaction term  ───────────────────────────
    fe["pressure_temp_index"] = (
        fe["column_pressure_kPa"] * fe["feed_temperature_C"]
    ).round(2)

    # ── Summary ───────────────────────────────────────────────────────────────
    new_cols = [
        "relative_volatility_est", "reflux_to_feed_ratio",
        "separation_factor",        "tray_utilisation",
        "energy_per_kmol",          "pressure_temp_index",
    ]
    print("=" * 70)
    print("ENGINEERED FEATURES — Summary Statistics")
    print("=" * 70)
    print(fe[new_cols].describe().round(4).to_string())
    print()

    return fe


# =============================================================================
# SECTION 5 — NORMALISATION
# =============================================================================

def normalise(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply Min-Max scaling to all numerical columns and return both the
    scaled DataFrame and the fitted scaler (for inverse-transform later).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    (df_scaled, scaler) : tuple
        df_scaled — all columns scaled to [0, 1]
        scaler    — fitted sklearn MinMaxScaler
    """
    scaler    = MinMaxScaler()
    arr_scaled = scaler.fit_transform(df.select_dtypes(include=np.number))
    df_scaled  = pd.DataFrame(arr_scaled, columns=df.select_dtypes(include=np.number).columns)

    print("=" * 70)
    print("NORMALISATION (Min-Max  →  [0, 1])")
    print("=" * 70)
    print(df_scaled.describe().round(4).to_string())
    print()

    return df_scaled, scaler


# =============================================================================
# SECTION 6 — VISUALISATIONS
# =============================================================================

# ── Helper ────────────────────────────────────────────────────────────────────

def _draw_heatmap(
    corr   : pd.DataFrame,
    title  : str,
    fname  : str,
    figsize: tuple = (14, 11),
    annot  : bool  = True,
) -> None:
    """
    Render a styled lower-triangle correlation heatmap and save to disk.

    Parameters
    ----------
    corr    : pd.DataFrame   Correlation matrix (square).
    title   : str            Figure title.
    fname   : str            Output filename (saved inside OUT_DIR).
    figsize : tuple          Matplotlib figure size.
    annot   : bool           Whether to annotate each cell with its value.
    """
    mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask       = mask,
        cmap       = PALETTE,
        vmin       = -1,
        vmax       = 1,
        center     = 0,
        annot      = annot,
        fmt        = ".2f",
        linewidths = 0.4,
        linecolor  = "white",
        square     = True,
        cbar_kws   = {"shrink": 0.75, "label": "Pearson r"},
        ax         = ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()
    path = f"{OUT_DIR}/{fname}"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── 6A.  Raw-feature heatmap ──────────────────────────────────────────────────

def plot_raw_heatmap(df_scaled: pd.DataFrame) -> None:
    """
    Pearson correlation heatmap for the original (scaled) feature set.

    Parameters
    ----------
    df_scaled : pd.DataFrame   Min-Max-scaled dataset.
    """
    print("=" * 70)
    print("HEATMAP 1 — Raw Features (after Min-Max scaling)")
    print("=" * 70)
    corr = df_scaled.corr()
    _draw_heatmap(
        corr,
        title  = "Correlation Heatmap — Raw Distillation Features (Min-Max Scaled)",
        fname  = "heatmap_raw_features.png",
        figsize= (13, 10),
    )


# ── 6B.  Engineered-feature heatmap ──────────────────────────────────────────

def plot_engineered_heatmap(df_scaled: pd.DataFrame) -> None:
    """
    Pearson correlation heatmap highlighting engineered features alongside
    the four output variables.

    Parameters
    ----------
    df_scaled : pd.DataFrame   Min-Max-scaled full dataset (raw + engineered).
    """
    print("=" * 70)
    print("HEATMAP 2 — Engineered Features vs Outputs")
    print("=" * 70)

    eng_cols = [
        "relative_volatility_est", "reflux_to_feed_ratio",
        "separation_factor",        "tray_utilisation",
        "energy_per_kmol",          "pressure_temp_index",
    ]
    output_cols = [
        "distillate_purity_molfrac",   "bottoms_composition_molfrac",
        "energy_consumption_kW",        "column_efficiency_pct",
    ]
    cols = [c for c in eng_cols + output_cols if c in df_scaled.columns]
    corr = df_scaled[cols].corr()

    _draw_heatmap(
        corr,
        title  = "Correlation Heatmap — Engineered Features vs Output Variables",
        fname  = "heatmap_engineered_features.png",
        figsize= (13, 10),
    )


# ── 6C.  Distribution grid of engineered features ────────────────────────────

def plot_feature_distributions(df: pd.DataFrame) -> None:
    """
    KDE + histogram grid for all engineered features (unscaled values).

    Parameters
    ----------
    df : pd.DataFrame   Extended dataset (raw + engineered, unscaled).
    """
    eng_cols = [
        "relative_volatility_est", "reflux_to_feed_ratio",
        "separation_factor",        "tray_utilisation",
        "energy_per_kmol",          "pressure_temp_index",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, col in zip(axes, eng_cols):
        data = df[col].dropna()
        sns.histplot(data, kde=True, ax=ax, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.set_title(col.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        # Annotate mean and std
        ax.axvline(data.mean(), color="#E07B54", lw=1.5, ls="--", label=f"μ={data.mean():.3f}")
        ax.legend(fontsize=8)

    fig.suptitle("Engineered Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{OUT_DIR}/engineered_feature_distributions.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── 6D.  Pairplot: engineered features coloured by distillate purity ──────────

def plot_pairplot(df: pd.DataFrame) -> None:
    """
    Seaborn pair-plot of key engineered features, colour-coded by distillate
    purity quartile to expose separation quality clusters.

    Parameters
    ----------
    df : pd.DataFrame   Extended unscaled dataset.
    """
    # Bin distillate purity into quartile labels
    df_plot = df.copy()
    df_plot["purity_quartile"] = pd.qcut(
        df_plot["distillate_purity_molfrac"],
        q      = 4,
        labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
    )

    pair_cols = [
        "relative_volatility_est", "separation_factor",
        "reflux_to_feed_ratio",    "energy_per_kmol",
        "purity_quartile",
    ]

    g = sns.pairplot(
        df_plot[pair_cols],
        hue         = "purity_quartile",
        diag_kind   = "kde",
        plot_kws    = {"alpha": 0.45, "s": 18, "linewidth": 0},
        diag_kws    = {"linewidth": 1.5},
        palette     = "Set2",
        corner      = True,
    )
    g.figure.suptitle(
        "Pair-Plot: Engineered Features  —  coloured by Distillate Purity Quartile",
        y=1.01, fontsize=13, fontweight="bold",
    )
    path = f"{OUT_DIR}/pairplot_engineered.png"
    g.figure.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── 6E.  Clustermap (hierarchical clustering of full feature set) ─────────────

def plot_clustermap(df_scaled: pd.DataFrame) -> None:
    """
    Seaborn clustermap that groups correlated features via hierarchical
    clustering — useful for identifying redundant or complementary variables.

    Parameters
    ----------
    df_scaled : pd.DataFrame   Min-Max-scaled full dataset.
    """
    corr = df_scaled.corr()

    g = sns.clustermap(
        corr,
        cmap        = PALETTE,
        vmin        = -1,
        vmax        = 1,
        center      = 0,
        annot       = True,
        fmt         = ".2f",
        linewidths  = 0.3,
        figsize     = (15, 13),
        dendrogram_ratio = 0.12,
        cbar_pos    = (0.02, 0.82, 0.03, 0.15),
    )
    g.figure.suptitle(
        "Hierarchical Clustermap — All Features (Raw + Engineered)",
        y=1.01, fontsize=14, fontweight="bold",
    )
    path = f"{OUT_DIR}/clustermap_all_features.png"
    g.figure.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_PATH = "distillation_synthetic_dataset.csv"

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df_raw = load_dataset(DATA_PATH)

    # ── 2. Missing-value audit ────────────────────────────────────────────────
    audit_missing(df_raw)

    # ── 3. Descriptive statistics ─────────────────────────────────────────────
    describe_dataset(df_raw)

    # ── 4. Feature engineering ────────────────────────────────────────────────
    df_eng = engineer_features(df_raw)

    # ── 5. Normalise (full engineered frame) ─────────────────────────────────
    df_scaled, scaler = normalise(df_eng)

    # ── 6. Visualisations ─────────────────────────────────────────────────────
    print("=" * 70)
    print("GENERATING VISUALISATIONS")
    print("=" * 70)

    plot_raw_heatmap(df_scaled)
    plot_engineered_heatmap(df_scaled)
    plot_feature_distributions(df_eng)
    plot_pairplot(df_eng)
    plot_clustermap(df_scaled)

    print("\n✓  All outputs saved to the 'outputs/' directory.")

    # ── 7. Save processed dataset ─────────────────────────────────────────────
    out_csv = f"{OUT_DIR}/distillation_processed.csv"
    df_eng.to_csv(out_csv, index=False)
    print(f"✓  Processed dataset (raw + engineered) saved → {out_csv}\n")


if __name__ == "__main__":
    main()
