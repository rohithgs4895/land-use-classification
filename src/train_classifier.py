"""
Train Random Forest and Gradient Boosting classifiers for Phoenix land use.

Pipeline
--------
  Load CSV  ->  stratified split  ->  StandardScaler  ->  train two models
  ->  evaluate on validation set  ->  final test evaluation  ->  save artefacts

Models
------
  RandomForestClassifier           scikit-learn ensemble (parallel, n_jobs=-1)
  HistGradientBoostingClassifier   sklearn's fast histogram-based GB (200 iters)
    — equivalent to GradientBoosting(n_estimators=200, lr=0.1, max_depth=6) but
      10–50× faster on the 138k-row training set.

Target accuracy: ~87% (as per resume requirement)

Outputs
-------
  models/random_forest.pkl         RF model
  models/gradient_boosting.pkl     GB model
  models/scaler.pkl                fitted StandardScaler
  models/best_model.pkl            best of the two (by weighted F1 on val set)
  outputs/confusion_matrix_rf.png  normalised confusion matrix — Random Forest
  outputs/confusion_matrix_gb.png  normalised confusion matrix — Gradient Boosting
  outputs/feature_importance.png   top-20 feature importances (both models)

Usage
-----
    conda run -n arcgis-dl python src/train_classifier.py
"""

import pickle
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR  = Path(r"C:\Projects\land-use-classification")
FEAT_CSV     = PROJECT_DIR / "data"    / "features" / "training_features.csv"
MODELS_DIR   = PROJECT_DIR / "models"
OUTPUTS_DIR  = PROJECT_DIR / "outputs"

RF_PATH      = MODELS_DIR / "random_forest.pkl"
GB_PATH      = MODELS_DIR / "gradient_boosting.pkl"
SCALER_PATH  = MODELS_DIR / "scaler.pkl"
BEST_PATH    = MODELS_DIR / "best_model.pkl"

CM_RF_PATH   = OUTPUTS_DIR / "confusion_matrix_rf.png"
CM_GB_PATH   = OUTPUTS_DIR / "confusion_matrix_gb.png"
FI_PATH      = OUTPUTS_DIR / "feature_importance.png"

CLASS_NAMES  = ["Urban", "Vegetation", "Agricultural",
                "Bare_Soil", "Water", "Industrial"]

BAND_NAMES   = [
    "B02", "B03", "B04", "B08", "B11", "B12",
    "NDVI", "NDBI", "NDWI", "SAVI", "BSI",
    "EVI", "MNDWI", "UI", "NBI",
    "RedGreen", "SR", "SWIRRatio",
    "Elevation", "Slope", "Aspect_sin", "Aspect_cos",
    "PlanCurv", "ProfCurv", "TRI", "TPI",
    "NDVI_std", "NIR_mean", "NIR_std",
]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _divider(char: str = "-", width: int = 64) -> None:
    print(char * width)


def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f} min" if s >= 60 else f"{s:.1f} s"


# ---------------------------------------------------------------------------
# Step 1 — Load data
# ---------------------------------------------------------------------------
def load_data() -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    _divider("=")
    print("  STEP 1 - Load training features")
    _divider()

    df = pd.read_csv(FEAT_CSV)
    print(f"  Rows     : {len(df):,}")
    print(f"  Columns  : {len(df.columns)}")

    X = df[BAND_NAMES].values.astype(np.float32)
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    y = le.transform(df["class"].values)

    print()
    print(f"  {'Class':<16}  {'Count':>8}  {'%':>6}")
    _divider()
    for cls in CLASS_NAMES:
        mask = df["class"] == cls
        n    = mask.sum()
        print(f"  {cls:<16}  {n:>8,}  {n/len(df)*100:>5.1f}%")

    return X, y, le


# ---------------------------------------------------------------------------
# Step 2 — Split + scale
# ---------------------------------------------------------------------------
def split_and_scale(X: np.ndarray, y: np.ndarray) -> tuple:
    _divider("=")
    print("  STEP 2 - Stratified split (70 / 15 / 15) + StandardScaler")
    _divider()

    # Train vs temp (85% / 15%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    # Temp -> val / test (50 / 50 of the 30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.50, stratify=y_test, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    _save(scaler, SCALER_PATH)

    print(f"  Train : {len(X_train):,} samples  ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val   : {len(X_val):,} samples   ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test  : {len(X_test):,} samples   ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Scaler saved -> {SCALER_PATH.name}")

    return (X_train_s, y_train,
            X_val_s,   y_val,
            X_test_s,  y_test,
            scaler)


# ---------------------------------------------------------------------------
# Step 3 — Train models
# ---------------------------------------------------------------------------
def train_models(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Random Forest ──────────────────────────────────────────────────────
    _divider("=")
    print("  STEP 3a - Train Random Forest")
    _divider()
    print("  n_estimators=200  max_depth=20  class_weight='balanced'  n_jobs=-1")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators  = 200,
        max_depth     = 20,
        min_samples_leaf = 5,
        class_weight  = "balanced",
        n_jobs        = -1,
        random_state  = RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    print(f"  Training time : {_elapsed(t0)}")
    _save(rf, RF_PATH)
    print(f"  Saved -> {RF_PATH.name}")

    # ── Histogram-based Gradient Boosting ─────────────────────────────────
    _divider("=")
    print("  STEP 3b - Train Gradient Boosting  (HistGradientBoosting)")
    _divider()
    print("  max_iter=200  learning_rate=0.1  max_depth=6  class_weight='balanced'")
    print("  (HistGradientBoostingClassifier — sklearn's fast GB implementation)")
    t0 = time.time()
    gb = HistGradientBoostingClassifier(
        max_iter      = 200,
        learning_rate = 0.1,
        max_depth     = 6,
        min_samples_leaf = 20,
        class_weight  = "balanced",
        random_state  = RANDOM_STATE,
        verbose       = 0,
    )
    gb.fit(X_train, y_train)
    print(f"  Training time : {_elapsed(t0)}")
    _save(gb, GB_PATH)
    print(f"  Saved -> {GB_PATH.name}")

    return rf, gb


# ---------------------------------------------------------------------------
# Step 4 — Evaluate model
# ---------------------------------------------------------------------------
def evaluate(model, X: np.ndarray, y_true: np.ndarray,
             le: LabelEncoder, split: str) -> dict:
    y_pred   = model.predict(X)
    acc      = accuracy_score(y_true, y_pred)
    f1_w     = f1_score(y_true, y_pred, average="weighted")
    kappa    = cohen_kappa_score(y_true, y_pred)
    report   = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    return {
        "split":    split,
        "accuracy": acc,
        "f1_w":     f1_w,
        "kappa":    kappa,
        "y_true":   y_true,
        "y_pred":   y_pred,
        "report":   report,
    }


def print_eval(results: dict, model_name: str) -> None:
    r = results
    print(f"  {model_name}  [{r['split']}]")
    print(f"  Accuracy         : {r['accuracy']*100:6.2f}%")
    print(f"  F1 (weighted)    : {r['f1_w']*100:6.2f}%")
    print(f"  Cohen's Kappa    : {r['kappa']:6.4f}")
    print()
    print(f"  {'Class':<16}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>9}")
    _divider()
    rep = r["report"]
    for cls in CLASS_NAMES:
        cr = rep.get(cls, {})
        print(f"  {cls:<16}  {cr.get('precision',0)*100:>9.1f}%  "
              f"{cr.get('recall',0)*100:>7.1f}%  "
              f"{cr.get('f1-score',0)*100:>7.1f}%  "
              f"{int(cr.get('support',0)):>9,}")


# ---------------------------------------------------------------------------
# Step 5 — Confusion matrix plot
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Recall (row-normalised)"},
    )
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class",      fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path.name}")


# ---------------------------------------------------------------------------
# Step 6 — Feature importance plot
# ---------------------------------------------------------------------------
def plot_feature_importance(rf, gb, save_path: Path, top_n: int = 20) -> None:
    rf_imp = rf.feature_importances_

    # HistGB exposes feature_importances_ via permutation; fall back to zeros
    if hasattr(gb, "feature_importances_"):
        gb_imp = gb.feature_importances_
    else:
        gb_imp = np.zeros(len(BAND_NAMES))

    # Rank by mean importance
    mean_imp = (rf_imp + gb_imp) / 2
    idx      = np.argsort(mean_imp)[::-1][:top_n]
    names    = [BAND_NAMES[i] for i in idx]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, rf_imp[idx], width, label="Random Forest",    color="#2196F3", alpha=0.85)
    ax.bar(x + width/2, gb_imp[idx], width, label="Gradient Boosting", color="#FF5722", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Feature Importance (MDI)", fontsize=10)
    ax.set_title(f"Top {top_n} Feature Importances — RF vs Gradient Boosting", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    _divider("=")
    print("  Phoenix Land Use — Train Classifiers")
    _divider("=")
    print()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load & split ───────────────────────────────────────────────────────
    X, y, le = load_data()
    print()
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     scaler)  = split_and_scale(X, y)
    print()

    # ── Train ──────────────────────────────────────────────────────────────
    rf, gb = train_models(X_train, y_train)

    # ── Validation evaluation (model selection) ────────────────────────────
    _divider("=")
    print("  STEP 4 - Validation evaluation  (model selection)")
    _divider()
    rf_val = evaluate(rf, X_val, y_val, le, "validation")
    gb_val = evaluate(gb, X_val, y_val, le, "validation")
    print_eval(rf_val, "Random Forest")
    print()
    print_eval(gb_val, "Gradient Boosting")

    best_model_name = "Random Forest" if rf_val["f1_w"] >= gb_val["f1_w"] else "Gradient Boosting"
    best_model      = rf              if rf_val["f1_w"] >= gb_val["f1_w"] else gb
    _save(best_model, BEST_PATH)
    print()
    print(f"  Best model (val F1): {best_model_name}  "
          f"(RF={rf_val['f1_w']*100:.2f}%  GB={gb_val['f1_w']*100:.2f}%)")
    print(f"  Saved -> {BEST_PATH.name}")

    # ── Final test evaluation ──────────────────────────────────────────────
    _divider("=")
    print("  STEP 5 - Final test evaluation")
    _divider()
    rf_test = evaluate(rf, X_test, y_test, le, "test")
    gb_test = evaluate(gb, X_test, y_test, le, "test")

    print_eval(rf_test, "Random Forest")
    print()
    print_eval(gb_test, "Gradient Boosting")

    # ── Plots ──────────────────────────────────────────────────────────────
    _divider("=")
    print("  STEP 6 - Save plots")
    _divider()

    rf_acc = rf_test["accuracy"] * 100
    gb_acc = gb_test["accuracy"] * 100

    plot_confusion_matrix(
        rf_test["y_true"], rf_test["y_pred"],
        f"Random Forest — Phoenix Land Use  (Accuracy {rf_acc:.1f}%)",
        CM_RF_PATH,
    )
    plot_confusion_matrix(
        gb_test["y_true"], gb_test["y_pred"],
        f"Gradient Boosting — Phoenix Land Use  (Accuracy {gb_acc:.1f}%)",
        CM_GB_PATH,
    )
    plot_feature_importance(rf, gb, FI_PATH)

    # ── Final summary ──────────────────────────────────────────────────────
    print()
    _divider("=")
    print("  FINAL SUMMARY")
    _divider("=")
    print(f"  {'Metric':<22}  {'Random Forest':>15}  {'Grad. Boosting':>15}")
    _divider()
    for label, rf_val_r, gb_val_r in [
        ("Accuracy",        rf_test["accuracy"]*100, gb_test["accuracy"]*100),
        ("F1  (weighted)",  rf_test["f1_w"]*100,     gb_test["f1_w"]*100),
        ("Cohen's Kappa",   rf_test["kappa"],         gb_test["kappa"]),
    ]:
        rf_s = f"{rf_val_r:.4f}" if "Kappa" in label else f"{rf_val_r:.2f}%"
        gb_s = f"{gb_val_r:.4f}" if "Kappa" in label else f"{gb_val_r:.2f}%"
        print(f"  {label:<22}  {rf_s:>15}  {gb_s:>15}")

    _divider("=")
    print(f"  Best model         : {best_model_name}")
    _divider("=")
    print()
    print("  Saved artefacts:")
    for p in [RF_PATH, GB_PATH, SCALER_PATH, BEST_PATH,
              CM_RF_PATH, CM_GB_PATH, FI_PATH]:
        mb = p.stat().st_size / 1_048_576
        print(f"    {p.name:<36}  {mb:5.1f} MB")
    _divider("=")


if __name__ == "__main__":
    main()
