"""
==============================================================================
Distillation Column — Multi-Target Regression Modelling
==============================================================================
Author : Machine Learning Engineering
Targets: distillate_purity_molfrac  |  energy_consumption_kW

Pipeline
--------
1.  Load processed dataset
2.  Define features & targets
3.  Train / test split  (80 / 20, stratified bins for purity target)
4.  Model zoo
        (a) Linear Regression         — baseline
        (b) Random Forest Regressor   — ensemble / non-linear
        (c) Gradient Boosting (GBRT)  — boosted ensemble
5.  Evaluation  (R², RMSE, MAE) on both targets
6.  Model comparison table + winner selection
7.  Residual & prediction plots
8.  Feature importance charts
9.  Persist best model as .pkl

Dependencies
------------
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
==============================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import warnings
import time
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.linear_model      import LinearRegression
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput       import MultiOutputRegressor
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.metrics           import r2_score, mean_squared_error, mean_absolute_error

# ── Reproducibility & output dir ──────────────────────────────────────────────
SEED    = 42
np.random.seed(SEED)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
FIG_DPI = 150

# =============================================================================
# SECTION 1 — LOAD DATA
# =============================================================================

def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the processed distillation dataset and separate features from targets.

    Features used
    ~~~~~~~~~~~~~
    Raw operational inputs + engineered features from the EDA step.
    Output variables that would cause data leakage are excluded from X:
        - bottoms_composition_molfrac  (direct material-balance proxy of x_D)
        - energy_per_kmol              (energy_consumption / feed — leaks target)

    Parameters
    ----------
    path : str  Path to CSV file.

    Returns
    -------
    (X, y) : tuple of DataFrames
    """
    df = pd.read_csv(path)

    # ── Input features ────────────────────────────────────────────────────────
    FEATURE_COLS = [
        "feed_composition_molfrac",
        "feed_temperature_C",
        "reflux_ratio",
        "column_pressure_kPa",
        "num_trays",
        "feed_flow_rate_kmolph",
        # Engineered (no leakage)
        "relative_volatility_est",
        "reflux_to_feed_ratio",
        "separation_factor",
        "tray_utilisation",
        "pressure_temp_index",
    ]

    # ── Targets ───────────────────────────────────────────────────────────────
    TARGET_COLS = [
        "distillate_purity_molfrac",
        "energy_consumption_kW",
    ]

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COLS].copy()

    print("=" * 70)
    print("DATA LOADED")
    print("=" * 70)
    print(f"  Features : {X.shape[1]}  →  {list(X.columns)}")
    print(f"  Targets  : {y.shape[1]}  →  {list(y.columns)}")
    print(f"  Rows     : {len(df):,}\n")

    return X, y


# =============================================================================
# SECTION 2 — TRAIN / TEST SPLIT
# =============================================================================

def split_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float = 0.20,
) -> tuple:
    """
    Stratified 80/20 split.

    Purity is binned into 10 quantile groups so each split has representative
    coverage across the full purity range.

    Parameters
    ----------
    X         : feature DataFrame
    y         : target DataFrame
    test_size : fraction reserved for testing

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # Stratification bins on distillate purity
    purity_bins = pd.qcut(y["distillate_purity_molfrac"], q=10, labels=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = SEED,
        stratify     = purity_bins,
    )

    print("=" * 70)
    print("TRAIN / TEST SPLIT  (80 / 20)")
    print("=" * 70)
    print(f"  Train : {len(X_train):,} rows")
    print(f"  Test  : {len(X_test):,} rows\n")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 3 — MODEL DEFINITIONS
# =============================================================================

def build_model_zoo(seed: int = SEED) -> dict:
    """
    Return a dictionary of named sklearn Pipelines.

    Each pipeline applies StandardScaler followed by the regressor.
    Multi-output is handled natively by RF and GBRT wrappers.
    LinearRegression also handles multi-output natively.

    Parameters
    ----------
    seed : int  Random state for reproducibility.

    Returns
    -------
    dict  {name: Pipeline}
    """
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),

        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators = 300,
                    max_depth    = None,
                    min_samples_leaf = 2,
                    n_jobs       = -1,
                    random_state = seed,
                )
            )),
        ]),

        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators     = 300,
                    learning_rate    = 0.05,
                    max_depth        = 5,
                    subsample        = 0.8,
                    min_samples_leaf = 4,
                    random_state     = seed,
                )
            )),
        ]),
    }

    return models


# =============================================================================
# SECTION 4 — TRAINING & EVALUATION
# =============================================================================

def evaluate(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    target_names: list[str],
) -> pd.DataFrame:
    """
    Compute R², RMSE, and MAE for every target column.

    Parameters
    ----------
    y_true       : ground-truth targets
    y_pred       : model predictions (2D array)
    target_names : list of target column names

    Returns
    -------
    pd.DataFrame  with columns [target, R², RMSE, MAE]
    """
    records = []
    for i, name in enumerate(target_names):
        yt = y_true.iloc[:, i].values
        yp = y_pred[:, i]
        records.append({
            "target" : name,
            "R²"     : r2_score(yt, yp),
            "RMSE"   : mean_squared_error(yt, yp) ** 0.5,
            "MAE"    : mean_absolute_error(yt, yp),
        })
    return pd.DataFrame(records)


def train_and_evaluate(
    models    : dict,
    X_train   : pd.DataFrame,
    X_test    : pd.DataFrame,
    y_train   : pd.DataFrame,
    y_test    : pd.DataFrame,
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Fit each model, measure wall-clock time, evaluate on the test set, and
    collect 5-fold cross-validation R² for the purity target.

    Parameters
    ----------
    models  : model zoo dict
    X_train, X_test, y_train, y_test : split datasets

    Returns
    -------
    (fitted_models, predictions, results_df)
    """
    target_names  = list(y_train.columns)
    fitted_models = {}
    predictions   = {}
    all_metrics   = []

    print("=" * 70)
    print("TRAINING & EVALUATION")
    print("=" * 70)

    for name, pipeline in models.items():
        print(f"\n  ── {name} ──")

        # Train
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"     Train time : {train_time:.2f}s")

        # Predict
        y_pred = pipeline.predict(X_test)

        # Metrics
        metrics_df = evaluate(y_test, y_pred, target_names)
        for _, row in metrics_df.iterrows():
            all_metrics.append({
                "Model"  : name,
                "Target" : row["target"],
                "R²"     : round(row["R²"],   4),
                "RMSE"   : round(row["RMSE"],  4),
                "MAE"    : round(row["MAE"],   4),
                "Train_s": round(train_time,   2),
            })
            print(f"     [{row['target']}]  "
                  f"R²={row['R²']:.4f}  "
                  f"RMSE={row['RMSE']:.4f}  "
                  f"MAE={row['MAE']:.4f}")

        fitted_models[name] = pipeline
        predictions[name]   = y_pred

    results_df = pd.DataFrame(all_metrics)
    return fitted_models, predictions, results_df


# =============================================================================
# SECTION 5 — MODEL COMPARISON & WINNER SELECTION
# =============================================================================

def select_best_model(
    results_df    : pd.DataFrame,
    fitted_models : dict,
) -> tuple[str, object]:
    """
    Rank models by average R² across both targets and print a leaderboard.

    The winner is the model with the highest mean R².

    Parameters
    ----------
    results_df    : evaluation metrics DataFrame
    fitted_models : dict of fitted pipelines

    Returns
    -------
    (best_name, best_pipeline) : str, sklearn Pipeline
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON LEADERBOARD")
    print("=" * 70)

    # Average R² across targets per model
    leaderboard = (
        results_df.groupby("Model")[["R²", "RMSE", "MAE"]]
        .mean()
        .round(4)
        .sort_values("R²", ascending=False)
        .rename(columns={"R²": "Avg R²", "RMSE": "Avg RMSE", "MAE": "Avg MAE"})
    )
    print(leaderboard.to_string())

    best_name = leaderboard.index[0]
    print(f"\n  🏆  Best model : {best_name}  "
          f"(Avg R² = {leaderboard.loc[best_name, 'Avg R²']:.4f})\n")

    return best_name, fitted_models[best_name]


# =============================================================================
# SECTION 6 — VISUALISATIONS
# =============================================================================

def plot_metric_comparison(results_df: pd.DataFrame) -> None:
    """
    Side-by-side bar charts of R², RMSE, and MAE for each model × target.
    """
    targets = results_df["Target"].unique()
    metrics = ["R²", "RMSE", "MAE"]
    colors  = {"Linear Regression": "#4C72B0",
                "Random Forest":    "#55A868",
                "Gradient Boosting":"#C44E52"}

    fig, axes = plt.subplots(len(targets), len(metrics),
                             figsize=(15, 4 * len(targets)))

    for r, target in enumerate(targets):
        sub = results_df[results_df["Target"] == target]
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            bars = ax.bar(
                sub["Model"], sub[metric],
                color=[colors[m] for m in sub["Model"]],
                edgecolor="white", linewidth=0.6,
            )
            ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
            ax.set_title(f"{metric}  —  {target.replace('_',' ').title()}",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=15, labelsize=8)
            if metric == "R²":
                ax.set_ylim(0, 1.12)

    fig.suptitle("Model Performance Comparison", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{OUT_DIR}/model_comparison.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_predictions(
    predictions   : dict,
    y_test        : pd.DataFrame,
    best_name     : str,
) -> None:
    """
    Actual vs Predicted scatter plots for every model × target.
    Perfect-fit diagonal drawn for reference.
    """
    targets = list(y_test.columns)
    models  = list(predictions.keys())
    n_rows, n_cols = len(targets), len(models)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5 * n_rows))

    palette = {"Linear Regression": "#4C72B0",
               "Random Forest":     "#55A868",
               "Gradient Boosting": "#C44E52"}

    for r, target in enumerate(targets):
        y_true_col = y_test[target].values
        for c, name in enumerate(models):
            ax     = axes[r, c]
            y_pred = predictions[name][:, r]
            color  = palette[name]

            ax.scatter(y_true_col, y_pred, alpha=0.35, s=18,
                       color=color, edgecolors="none")

            # Perfect-fit line
            lo, hi = min(y_true_col.min(), y_pred.min()), \
                     max(y_true_col.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect fit")

            r2 = r2_score(y_true_col, y_pred)
            ax.set_title(f"{name}\n{target.replace('_',' ').title()}\n"
                         f"R²={r2:.4f}",
                         fontsize=9, fontweight="bold")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.legend(fontsize=7)

            # Gold border for best model
            if name == best_name:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#E0A800")
                    spine.set_linewidth(2.5)

    fig.suptitle("Actual vs Predicted  (gold border = best model)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{OUT_DIR}/actual_vs_predicted.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_residuals(
    predictions : dict,
    y_test      : pd.DataFrame,
    best_name   : str,
) -> None:
    """
    Residual (error) distribution histograms with KDE for each model × target.
    """
    targets = list(y_test.columns)
    models  = list(predictions.keys())

    fig, axes = plt.subplots(len(targets), len(models),
                             figsize=(6 * len(models), 4 * len(targets)))

    palette = {"Linear Regression": "#4C72B0",
               "Random Forest":     "#55A868",
               "Gradient Boosting": "#C44E52"}

    for r, target in enumerate(targets):
        y_true_col = y_test[target].values
        for c, name in enumerate(models):
            ax        = axes[r, c]
            residuals = y_true_col - predictions[name][:, r]
            color     = palette[name]

            sns.histplot(residuals, kde=True, ax=ax, color=color,
                         edgecolor="white", linewidth=0.3, bins=30)
            ax.axvline(0, color="black", lw=1.2, ls="--")
            ax.axvline(residuals.mean(), color="#E07B54", lw=1.2, ls=":",
                       label=f"μ={residuals.mean():.3f}")
            ax.set_title(f"{name}\n{target.replace('_',' ').title()}",
                         fontsize=9, fontweight="bold")
            ax.set_xlabel("Residual (Actual − Predicted)")
            ax.legend(fontsize=7)

            if name == best_name:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#E0A800")
                    spine.set_linewidth(2.5)

    fig.suptitle("Residual Distributions  (gold border = best model)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{OUT_DIR}/residuals.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_feature_importance(
    fitted_models : dict,
    X_train       : pd.DataFrame,
    best_name     : str,
) -> None:
    """
    Horizontal bar charts of feature importances for tree-based models.
    Linear Regression uses |coefficient| as a proxy.
    """
    feature_names = X_train.columns.tolist()
    targets       = ["distillate_purity_molfrac", "energy_consumption_kW"]
    n_targets     = len(targets)

    palette = {"Linear Regression": "#4C72B0",
               "Random Forest":     "#55A868",
               "Gradient Boosting": "#C44E52"}

    for name, pipeline in fitted_models.items():
        fig, axes = plt.subplots(1, n_targets,
                                 figsize=(14, max(5, len(feature_names) * 0.45)))
        color = palette[name]

        for ax, target_idx, target_name in zip(
            axes, range(n_targets), targets
        ):
            regressor = pipeline.named_steps["model"]

            if name == "Linear Regression":
                # Coefficient magnitudes (standardised features → comparable)
                coefs = np.abs(regressor.coef_[target_idx])
                importance = coefs / coefs.sum()
                xlabel = "|Coefficient| (normalised)"
            else:
                # MultiOutputRegressor stores per-estimator objects
                sub_est    = regressor.estimators_[target_idx]
                importance = sub_est.feature_importances_
                xlabel     = "Feature Importance (mean decrease in impurity)"

            # Sort descending
            order      = np.argsort(importance)
            sorted_imp = importance[order]
            sorted_fea = [feature_names[i] for i in order]

            bars = ax.barh(sorted_fea, sorted_imp,
                           color=color, edgecolor="white", linewidth=0.4)
            ax.set_title(f"{target_name.replace('_',' ').title()}",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=8)
            ax.tick_params(axis="y", labelsize=8)

            # Annotate values
            for bar, val in zip(bars, sorted_imp):
                ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=7)

        border = "#E0A800" if name == best_name else "#AAAAAA"
        fig.suptitle(f"Feature Importance — {name}  "
                     + ("🏆" if name == best_name else ""),
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = f"{OUT_DIR}/feature_importance_{name.replace(' ','_').lower()}.png"
        plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")


# =============================================================================
# SECTION 7 — PERSIST BEST MODEL
# =============================================================================

def save_model(
    pipeline   : object,
    model_name : str,
    X_train    : pd.DataFrame,
    y_train    : pd.DataFrame,
) -> str:
    """
    Serialise the best pipeline to disk as a .pkl file using joblib.

    A metadata sidecar (JSON) is also written alongside the model recording
    the feature names, target names, and training set size.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline
    model_name : human-readable model name
    X_train    : training features  (for metadata)
    y_train    : training targets   (for metadata)

    Returns
    -------
    str  Path to saved .pkl file
    """
    import json, datetime

    safe_name  = model_name.replace(" ", "_").lower()
    pkl_path   = f"{OUT_DIR}/best_model_{safe_name}.pkl"
    meta_path  = f"{OUT_DIR}/best_model_{safe_name}_metadata.json"

    joblib.dump(pipeline, pkl_path, compress=3)

    metadata = {
        "model_name"     : model_name,
        "saved_at"       : datetime.datetime.utcnow().isoformat() + "Z",
        "feature_columns": list(X_train.columns),
        "target_columns" : list(y_train.columns),
        "train_rows"     : len(X_train),
        "pkl_path"       : pkl_path,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Model  saved → {pkl_path}")
    print(f"  Meta   saved → {meta_path}")
    return pkl_path


def verify_saved_model(pkl_path: str, X_test: pd.DataFrame) -> None:
    """
    Reload the saved model from disk and run a smoke-test prediction
    to confirm the file is intact and usable.

    Parameters
    ----------
    pkl_path : str          Path to the .pkl file.
    X_test   : pd.DataFrame A slice of test features.
    """
    loaded = joblib.load(pkl_path)
    sample_pred = loaded.predict(X_test.iloc[:5])
    print("\n  ✓  Model reload verified. Sample predictions:")
    print(f"     Purity  : {sample_pred[:, 0].round(4)}")
    print(f"     Energy  : {sample_pred[:, 1].round(2)}\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    DATA_PATH = "outputs/distillation_processed.csv"

    # 1. Load
    X, y = load_data(DATA_PATH)

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Build model zoo
    models = build_model_zoo()

    # 4. Train + evaluate
    fitted_models, predictions, results_df = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )

    # Print full results table
    print("\n" + "=" * 70)
    print("FULL METRICS TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # 5. Select winner
    best_name, best_pipeline = select_best_model(results_df, fitted_models)

    # 6. Visualise
    print("=" * 70)
    print("GENERATING VISUALISATIONS")
    print("=" * 70)
    plot_metric_comparison(results_df)
    plot_predictions(predictions, y_test, best_name)
    plot_residuals(predictions, y_test, best_name)
    plot_feature_importance(fitted_models, X_train, best_name)

    # 7. Save best model
    print("=" * 70)
    print("SAVING BEST MODEL")
    print("=" * 70)
    pkl_path = save_model(best_pipeline, best_name, X_train, y_train)
    verify_saved_model(pkl_path, X_test)

    print("✓  All outputs written to the 'outputs/' directory.")


if __name__ == "__main__":
    main()
