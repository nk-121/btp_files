"""Compare ANN pipeline with the Modified Pro-Energy baseline (Algorithm 1).

Usage:
    python proenergy_modified.py [datafile] [out_dir]

This script:
- loads and preprocesses the dataset using the existing pipeline
- loads the pre-trained ANN from  results/model.h5
- loads the pre-fitted scaler from results/scaler.pkl
- runs the Modified Pro-Energy algorithm (Algorithm 1 from paper):
    alpha=0.5, beta=0.2, D=10, thre=0.4
    with weather-condition classification, MAE-based day similarity,
    weighted profile averaging, and slot-by-slot error correction
- evaluates on test set and saves metrics, plots and predictions

Prerequisites:
    Run pipeline_ann.py first so that results/model.h5 and
    results/scaler.pkl exist.

FIXES applied vs original:
  FIX 1  K_window=6 sliding-window MAE (was full-day MAE, too coarse for hourly)
  FIX 2  Weather classification candidates are now USED to filter similar days
         (was computed but dead-code in original)
  FIX 3  WP = Σ Wⱼ·Edⱼ — removed spurious ÷(D_eff-1) that un-normalised
         already-normalised weights
  FIX 4  ρ = β·ρ + (1−β)·(pred[n-1] − actual[n-1])
         (original used E_hat_prev − day_vals[n-1] where E_hat_prev was
          initialised to day_vals[0], giving 0 on slot-1 and stale values
          thereafter; effectively zeroing out error correction)
  FIX 5  mpro_true is taken from the original y_test Series, NOT rebuilt
         from the padded daily_matrix, so ground-truth is never contaminated
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# try to reuse functions from pipeline_ann
try:
    from pipeline_ann import load_data, preprocess, TARGET_COLUMN
except Exception:
    TARGET_COLUMN = "Global CMP22 (vent/cor) [W/m^2]"
    from pipeline_ann import load_data, preprocess

import tensorflow as tf

# ── colour palette (consistent across all plots) ─────────────────────────────
C_TRUE = "#2563EB"   # blue   – ground truth
C_PRO  = "#F97316"   # orange – Pro-Energy
C_ANN  = "#16A34A"   # green  – ANN (our algorithm)
C_ERR_PRO = "#FCA5A5"
C_ERR_ANN = "#86EFAC"
STYLE  = "seaborn-v0_8-whitegrid"


# ═════════════════════════════════════════════════════════════════════════════
# Modified Pro-Energy Algorithm  (Algorithm 1 from paper)
#
# Parameters
# ----------
# alpha    : smoothing weight for EWMA prediction  (default 0.5)
# beta     : error correction momentum              (default 0.2)
# D        : number of similar historical days used (default 10)
# thre     : threshold for sunny/cloudy detection   (default 0.4)
# K_window : sliding-window size for intra-day MAE  (default 6)
#            — FIX 1: was full-day MAE in original, too coarse for hourly
#
# The algorithm operates on a *daily* energy profile matrix
#   E[day, slot]  shape (num_days, slots_per_day)
#
# Prediction for day d, slot n:
#   1. Classify each historical day as sunny / cloudy / mixed using thre
#      — FIX 2: classification now filters candidates (was dead-code)
#   2. Compute sliding-window MAE between the observed slots of day d and
#      each candidate historical day C  — FIX 1
#   3. Select the D most-similar days; compute weights W_j ∝ (1 − norm MAE)
#   4. Weighted-average profile:  WP = Σ Wⱼ · E_j[slot]
#      — FIX 3: removed spurious ÷(D-1) present in original
#   5. Running error correction:  ρ ← β·ρ + (1−β)·[pred(n−1) − actual(n−1)]
#      — FIX 4: original used stale E_hat_prev ≈ actual[0], making ρ ≈ 0
#   6. Final prediction:          Ê[d,n] ← α·actual(n−1) + (1−α)·WP − ρ
# ═════════════════════════════════════════════════════════════════════════════
class ModifiedProEnergy:
    """Corrected implementation of Algorithm 1 (Modified Pro-Energy)."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.2,
                 D: int = 10, thre: float = 0.4, K_window: int = 6):
        # FIX 1: K_window added (was absent in original)
        self.alpha    = alpha
        self.beta     = beta
        self.D        = D
        self.thre     = thre
        self.K_window = K_window          # FIX 1
        self._history: np.ndarray | None = None
        self._E_prime: float             = 0.0

    # ─────────────────────────────────────────────────────────────── fit ──
    def fit(self, series: pd.Series) -> "ModifiedProEnergy":
        """
        Build the historical day-profile matrix from a *sorted* hourly Series.

        Parameters
        ----------
        series : pd.Series with a DatetimeIndex (hourly or sub-hourly values).
        """
        series = series.sort_index()

        daily = [grp.values for _, grp in series.groupby(series.index.date)]

        lengths  = [len(d) for d in daily]
        slot_len = int(np.median(lengths))
        normed   = []
        for d in daily:
            if len(d) >= slot_len:
                normed.append(d[:slot_len])
            else:
                pad       = np.full(slot_len, d[-1] if len(d) else 0.0)
                pad[:len(d)] = d
                normed.append(pad)

        self._history  = np.array(normed, dtype=float)   # (num_days, slots)
        self._E_prime  = float(self._history.mean())
        self._slot_len = slot_len
        return self

    # ──────────────────────────────────────────────────────────── predict ──
    def predict(self, series: pd.Series) -> np.ndarray:
        """
        Predict each slot of every day in *series* using Algorithm 1.

        Returns a flat numpy array aligned with series (sorted by index).

        FIX 5: ground-truth values are read from the original series, never
        from the padded daily_matrix, so mpro_true == y_test.values exactly.
        """
        if self._history is None:
            raise RuntimeError("Call fit() before predict().")

        series  = series.sort_index()
        history = self._history        # (H, S)
        H, S    = history.shape

        alpha    = self.alpha
        beta     = self.beta
        D        = self.D
        thre     = self.thre
        K_window = self.K_window       # FIX 1

        E_prime  = self._E_prime

        # ── FIX 2: classify historical days ──────────────────────────────
        # classify_day(sum) → 'sunny' | 'cloudy' | 'mixed'
        day_sums = history.sum(axis=1)            # (H,)
        mean_E   = day_sums.mean()

        def classify(s):
            if s > (1 + thre) * mean_E:
                return "sunny"
            elif s < (1 - thre) * mean_E:
                return "cloudy"
            return "mixed"

        history_classes = np.array([classify(s) for s in day_sums])

        # group test data into days
        test_day_groups = list(series.groupby(series.index.date))
        predictions_all: list[float] = []

        for _, day_series in test_day_groups:
            day_vals = day_series.values.astype(float)
            n_slots  = len(day_vals)

            # ── FIX 2: filter candidates by weather class ─────────────────
            today_sum   = day_vals.sum()
            today_class = classify(today_sum)
            same_mask   = history_classes == today_class
            # fall back to all history if fewer than D same-class days exist
            candidate_idx = np.where(same_mask)[0] if same_mask.sum() >= D else np.arange(H)

            # ── Step 22-27: slot-by-slot prediction ───────────────────────
            rho            = 0.0
            pred_prev      = day_vals[0]      # initialise; overwritten each slot
            day_preds: list[float] = []

            for n in range(n_slots):
                slot_idx = min(n, S - 1)

                # ── FIX 1: sliding-window MAE instead of full-day MAE ──────
                start    = max(0, n - K_window)
                K_actual = n - start          # number of observed slots so far

                if K_actual == 0 or n == 0:
                    # no observed context yet — fall back to last history day
                    theta = float(history[candidate_idx[-1], slot_idx])
                else:
                    # observed window from *original* series (FIX 5)
                    obs_window  = day_vals[start:n]                   # (K_actual,)
                    hist_window = history[candidate_idx, start:n]     # (C, K_actual)
                    maes        = np.mean(np.abs(hist_window - obs_window), axis=1)

                    D_eff   = min(D, len(candidate_idx))
                    top_pos = np.argsort(maes)[:D_eff]
                    top_idx = candidate_idx[top_pos]
                    top_mae = maes[top_pos]

                    # ── FIX 3: weights W_j, NO ÷(D-1) ────────────────────
                    mae_sum = top_mae.sum()
                    if mae_sum == 0:
                        weights = np.ones(D_eff) / D_eff
                    else:
                        weights = 1.0 - top_mae / mae_sum
                        w_sum   = weights.sum()
                        weights = weights / w_sum if w_sum > 0 else np.ones(D_eff) / D_eff

                    # WP = Σ Wⱼ · E_j[slot]   (FIX 3: no ÷(D-1))
                    theta = float(np.sum(weights * history[top_idx, slot_idx]))

                # ── FIX 4: error correction uses actual pred vs actual ──────
                if n > 0:
                    rho = beta * rho + (1 - beta) * (pred_prev - day_vals[n - 1])

                E_prev  = day_vals[n - 1] if n > 0 else day_vals[0]
                E_hat_n = alpha * E_prev + (1 - alpha) * theta - rho
                E_hat_n = max(0.0, E_hat_n)   # irradiance cannot be negative

                day_preds.append(E_hat_n)
                pred_prev = E_hat_n            # FIX 4: track prediction, not actual

            predictions_all.extend(day_preds)

        return np.array(predictions_all[: len(series)], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# ─────────────────────────────────────────────────────────────────────────────
def _rolling(series, index, window=5):
    """Return a rolling-mean Series with a safe fallback."""
    try:
        return pd.Series(series, index=index).rolling(window=window, center=True, min_periods=1).mean()
    except Exception:
        return pd.Series(series).rolling(window=window, center=True, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 – Time-series comparison (main overview)
# ─────────────────────────────────────────────────────────────────────────────
def plot_timeseries(index, y_true, ann_pred, pro_pred, out_dir):
    df_plot = pd.DataFrame(
        {"actual": y_true, "ann": ann_pred, "pro": pro_pred},
        index=index,
    ).sort_index()

    def sm(col, w=5):
        return df_plot[col].rolling(window=w, center=True, min_periods=1).mean()

    with plt.style.context(STYLE):
        # (a) Full test-set view
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        ax.plot(df_plot.index, sm("actual"), label="Ground Truth", color=C_TRUE, lw=1.8, zorder=3)
        ax.plot(df_plot.index, sm("pro"),    label="Pro-Energy",   color=C_PRO,  lw=1.4, ls="--", zorder=2)
        ax.plot(df_plot.index, sm("ann"),    label="Our ANN",      color=C_ANN,  lw=1.4, ls="-.", zorder=2)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel(TARGET_COLUMN, fontsize=10)
        ax.set_title("Time-series Comparison  |  Ground Truth vs Models (full test set)",
                     fontsize=12, fontweight="bold")
        ax.legend(frameon=True, fontsize=9, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "01_timeseries_full.png"))
        plt.close(fig)

        # (b) Zoom: one representative day (highest-irradiance day)
        best_day = df_plot["actual"].resample("D").mean().idxmax().date()
        day_df   = df_plot[df_plot.index.date == best_day]

        if len(day_df) >= 3:
            fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150)
            x_min = np.arange(len(day_df))
            ax.plot(x_min, day_df["actual"].values, label="Ground Truth", color=C_TRUE, lw=2,   marker="o", ms=3)
            ax.plot(x_min, day_df["pro"].values,    label="Pro-Energy",   color=C_PRO,  lw=1.6, marker="s", ms=3)
            ax.plot(x_min, day_df["ann"].values,    label="Our ANN",      color=C_ANN,  lw=1.6, marker="^", ms=3)
            ax.set_xticks(x_min[::max(1, len(x_min)//8)])
            ax.set_xticklabels(
                [str(t.time())[:5] for t in day_df.index[::max(1, len(x_min)//8)]],
                rotation=30, fontsize=8,
            )
            ax.set_xlabel("Time of day", fontsize=10)
            ax.set_ylabel("Energy [W/m²]", fontsize=10)
            ax.set_title(f"Our Algorithm vs Pro-Energy  |  Day {best_day}  (paper Fig 3a style)",
                         fontsize=11, fontweight="bold")
            ax.legend(frameon=False, fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "01b_timeseries_one_day.png"))
            plt.close(fig)

        # (c) Low-variability and high-variability day windows
        daily_std = df_plot["actual"].resample("D").std().dropna()

        if len(daily_std) >= 2:
            smooth_day  = daily_std.idxmin().date()
            varying_day = daily_std.idxmax().date()

            for win_label, chosen_day, desc in [
                ("smooth",  smooth_day,  "Low variability  —  steady / sunny weather"),
                ("varying", varying_day, "High variability  —  patchy / cloudy weather"),
            ]:
                day_chunk = df_plot[df_plot.index.date == chosen_day]
                if len(day_chunk) < 3:
                    continue

                fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
                ax.plot(day_chunk.index, day_chunk["actual"].values,
                        label="Ground Truth", color=C_TRUE, lw=2,   marker="o", ms=3)
                ax.plot(day_chunk.index, day_chunk["pro"].values,
                        label="Pro-Energy",   color=C_PRO,  lw=1.6, marker="s", ms=3, ls="--")
                ax.plot(day_chunk.index, day_chunk["ann"].values,
                        label="Our ANN",      color=C_ANN,  lw=1.6, marker="^", ms=3, ls="-.")

                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)

                std_val = daily_std[pd.Timestamp(chosen_day)]
                ax.set_xlabel(f"Time of day  [{chosen_day}]", fontsize=10)
                ax.set_ylabel("Solar Irradiance [W/m²]", fontsize=10)
                ax.set_title(
                    f"{desc}\n"
                    f"Date: {chosen_day}   |   Irradiance std dev = {std_val:.1f} W/m²",
                    fontsize=10, fontweight="bold"
                )
                ax.legend(frameon=False, fontsize=9)
                ax.grid(axis="y", linestyle="--", alpha=0.3)
                ax.annotate(
                    "← each point = 1 hourly reading",
                    xy=(0.01, 0.04), xycoords="axes fraction",
                    fontsize=7, color="grey"
                )
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"01c_timeseries_window_{win_label}.png"))
                plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 – Residuals over time (signed error)
# ─────────────────────────────────────────────────────────────────────────────
def plot_residuals(index, y_true, ann_pred, pro_pred, out_dir):
    df_r = pd.DataFrame(
        {"actual": y_true, "ann": ann_pred, "pro": pro_pred}, index=index
    ).sort_index()
    res_ann = df_r["ann"].values - df_r["actual"].values
    res_pro = df_r["pro"].values - df_r["actual"].values

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), dpi=150, sharex=True)

        for ax, res, color, label in zip(
            axes,
            [res_pro, res_ann],
            [C_PRO,   C_ANN],
            ["Pro-Energy Residuals", "ANN Residuals"],
        ):
            t = df_r.index
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.fill_between(t, res, 0, where=(res > 0),  alpha=0.35, color=color, label="Over-prediction")
            ax.fill_between(t, res, 0, where=(res <= 0), alpha=0.35, color=color, label="Under-prediction")
            ax.plot(t, res, color=color, lw=0.8, alpha=0.7)
            ax.set_ylabel("Error (W/m²)", fontsize=9)
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            ax.legend(fontsize=8, frameon=False)

        axes[-1].set_xlabel("Sample index", fontsize=10)
        fig.suptitle("Residuals over Time  (Predicted − Actual)", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "02_residuals_over_time.png"))
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 – Scatter: Predicted vs Actual
# ─────────────────────────────────────────────────────────────────────────────
def plot_scatter(y_true, ann_pred, pro_pred, ann_m, pro_m, out_dir):
    lo = min(y_true.min(), ann_pred.min(), pro_pred.min()) * 0.95
    hi = max(y_true.max(), ann_pred.max(), pro_pred.max()) * 1.05
    diag = np.linspace(lo, hi, 200)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)

        for ax, pred, color, label, m in zip(
            axes,
            [pro_pred, ann_pred],
            [C_PRO,    C_ANN],
            ["Pro-Energy",  "Our ANN"],
            [pro_m,    ann_m],
        ):
            ax.scatter(y_true, pred, alpha=0.35, s=10, color=color, edgecolors="none")
            ax.plot(diag, diag, "k--", lw=1.2, label="Perfect fit")
            coef = np.polyfit(y_true, pred, 1)
            ax.plot(diag, np.polyval(coef, diag), color=color, lw=1.8, label="Fit line")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("Actual (W/m²)", fontsize=10)
            ax.set_ylabel("Predicted (W/m²)", fontsize=10)
            ax.set_title(f"{label}\nR²={m['r2']:.3f}  RMSE={m['rmse']:.1f}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.set_aspect("equal", "box")

        fig.suptitle("Predicted vs Actual", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "03_scatter_predicted_vs_actual.png"))
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 – Absolute-error distributions (histogram + KDE)
# ─────────────────────────────────────────────────────────────────────────────
def plot_error_distribution(y_true, ann_pred, pro_pred, out_dir):
    abs_ann = np.abs(ann_pred - y_true)
    abs_pro = np.abs(pro_pred - y_true)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        bins = np.linspace(0, max(abs_ann.max(), abs_pro.max()) * 1.05, 45)
        ax.hist(abs_pro, bins=bins, alpha=0.55, color=C_PRO, label="Pro-Energy |error|", density=True)
        ax.hist(abs_ann, bins=bins, alpha=0.55, color=C_ANN, label="ANN |error|",        density=True)
        ax.axvline(abs_pro.mean(), color=C_PRO, ls="--", lw=1.5, label=f"Pro-Energy MAE={abs_pro.mean():.1f}")
        ax.axvline(abs_ann.mean(), color=C_ANN, ls="--", lw=1.5, label=f"ANN MAE={abs_ann.mean():.1f}")
        ax.set_xlabel("Absolute Error (W/m²)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title("Absolute Error Distribution", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "04_error_distribution.png"))
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 – Metrics bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_bar(ann_m, pro_m, out_dir):
    metric_names = ["RMSE", "MAE", "MSE (÷100)", "R²  (×100)"]
    ann_vals = [ann_m["rmse"], ann_m["mae"], ann_m["mse"] / 100, ann_m["r2"] * 100]
    pro_vals = [pro_m["rmse"], pro_m["mae"], pro_m["mse"] / 100, pro_m["r2"] * 100]

    x = np.arange(len(metric_names))
    width = 0.35

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
        bars_pro = ax.bar(x - width / 2, pro_vals, width, label="Pro-Energy", color=C_PRO, alpha=0.85)
        bars_ann = ax.bar(x + width / 2, ann_vals, width, label="Our ANN",    color=C_ANN, alpha=0.85)

        for bars in (bars_pro, bars_ann):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title("Model Performance Metrics", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "05_metrics_bar.png"))
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 – Cumulative error (CDF of absolute errors)
# ─────────────────────────────────────────────────────────────────────────────
def plot_cdf(y_true, ann_pred, pro_pred, out_dir):
    abs_ann = np.sort(np.abs(ann_pred - y_true))
    abs_pro = np.sort(np.abs(pro_pred - y_true))
    cdf = np.linspace(0, 1, len(abs_ann))

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.plot(abs_pro, cdf, color=C_PRO, lw=2,   label="Pro-Energy")
        ax.plot(abs_ann, cdf, color=C_ANN, lw=2,   label="Our ANN")
        ax.axhline(0.90, color="grey", ls=":", lw=1, label="90th percentile")
        ax.set_xlabel("Absolute Error (W/m²)", fontsize=10)
        ax.set_ylabel("Cumulative Fraction", fontsize=10)
        ax.set_title("CDF of Absolute Errors", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "06_cdf_errors.png"))
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 – Summary dashboard (all panels in one figure)
# ─────────────────────────────────────────────────────────────────────────────
def plot_dashboard(index, y_true, ann_pred, pro_pred, ann_m, pro_m, out_dir):
    df_s = pd.DataFrame(
        {"actual": y_true, "ann": ann_pred, "pro": pro_pred}, index=index
    ).sort_index()
    s_true = df_s["actual"].rolling(5, center=True, min_periods=1).mean()
    s_ann  = df_s["ann"].rolling(5, center=True, min_periods=1).mean()
    s_pro  = df_s["pro"].rolling(5, center=True, min_periods=1).mean()

    res_ann = ann_pred - y_true
    res_pro = pro_pred - y_true
    abs_ann = np.abs(res_ann)
    abs_pro = np.abs(res_pro)
    lo = min(y_true.min(), ann_pred.min(), pro_pred.min()) * 0.95
    hi = max(y_true.max(), ann_pred.max(), pro_pred.max()) * 1.05
    diag = np.linspace(lo, hi, 200)

    with plt.style.context(STYLE):
        fig = plt.figure(figsize=(18, 12), dpi=150)
        fig.suptitle("ANN vs Modified ProEnergy  —  Full Comparison Dashboard",
                     fontsize=14, fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

        ax_ts = fig.add_subplot(gs[0, :])
        ax_ts.plot(s_true.index, s_true.values, color=C_TRUE, lw=2,   label="Ground Truth", zorder=3)
        ax_ts.plot(s_pro.index,  s_pro.values,  color=C_PRO,  lw=1.5, ls="--", label="Pro-Energy", zorder=2)
        ax_ts.plot(s_ann.index,  s_ann.values,  color=C_ANN,  lw=1.5, ls="-.", label="Our ANN",    zorder=2)
        ax_ts.set_xlabel("Sample index"); ax_ts.set_ylabel("W/m²")
        ax_ts.set_title("Time-series Overview", fontweight="bold")
        ax_ts.legend(frameon=True, fontsize=9)

        ax_s1 = fig.add_subplot(gs[1, 0])
        ax_s1.scatter(y_true, pro_pred, alpha=0.3, s=8, color=C_PRO, edgecolors="none")
        ax_s1.plot(diag, diag, "k--", lw=1)
        ax_s1.plot(diag, np.polyval(np.polyfit(y_true, pro_pred, 1), diag), color=C_PRO, lw=1.8)
        ax_s1.set_xlim(lo, hi); ax_s1.set_ylim(lo, hi); ax_s1.set_aspect("equal", "box")
        ax_s1.set_xlabel("Actual"); ax_s1.set_ylabel("Predicted")
        ax_s1.set_title(f"Pro-Energy Scatter\nR²={pro_m['r2']:.3f}", fontweight="bold")

        ax_s2 = fig.add_subplot(gs[1, 1])
        ax_s2.scatter(y_true, ann_pred, alpha=0.3, s=8, color=C_ANN, edgecolors="none")
        ax_s2.plot(diag, diag, "k--", lw=1)
        ax_s2.plot(diag, np.polyval(np.polyfit(y_true, ann_pred, 1), diag), color=C_ANN, lw=1.8)
        ax_s2.set_xlim(lo, hi); ax_s2.set_ylim(lo, hi); ax_s2.set_aspect("equal", "box")
        ax_s2.set_xlabel("Actual"); ax_s2.set_ylabel("Predicted")
        ax_s2.set_title(f"ANN Scatter\nR²={ann_m['r2']:.3f}", fontweight="bold")

        ax_mb = fig.add_subplot(gs[1, 2])
        names = ["RMSE", "MAE"]
        vals_pro = [pro_m["rmse"], pro_m["mae"]]
        vals_ann = [ann_m["rmse"], ann_m["mae"]]
        x = np.arange(len(names)); w = 0.35
        b1 = ax_mb.bar(x - w/2, vals_pro, w, color=C_PRO, alpha=0.85, label="Pro-Energy")
        b2 = ax_mb.bar(x + w/2, vals_ann, w, color=C_ANN, alpha=0.85, label="ANN")
        for bars in (b1, b2):
            for bar in bars:
                ax_mb.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                           f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
        ax_mb.set_xticks(x); ax_mb.set_xticklabels(names)
        ax_mb.set_title("RMSE & MAE", fontweight="bold"); ax_mb.legend(fontsize=8)

        ax_r1 = fig.add_subplot(gs[2, 0])
        ax_r1.axhline(0, color="black", lw=0.8, ls="--")
        ax_r1.fill_between(range(len(res_pro)), res_pro, 0, alpha=0.4, color=C_PRO)
        ax_r1.plot(res_pro, color=C_PRO, lw=0.7, alpha=0.8)
        ax_r1.set_xlabel("Sample"); ax_r1.set_ylabel("Error (W/m²)")
        ax_r1.set_title("Pro-Energy Residuals", fontweight="bold")

        ax_r2 = fig.add_subplot(gs[2, 1])
        ax_r2.axhline(0, color="black", lw=0.8, ls="--")
        ax_r2.fill_between(range(len(res_ann)), res_ann, 0, alpha=0.4, color=C_ANN)
        ax_r2.plot(res_ann, color=C_ANN, lw=0.7, alpha=0.8)
        ax_r2.set_xlabel("Sample"); ax_r2.set_ylabel("Error (W/m²)")
        ax_r2.set_title("ANN Residuals", fontweight="bold")

        ax_cdf = fig.add_subplot(gs[2, 2])
        cdf = np.linspace(0, 1, len(abs_ann))
        ax_cdf.plot(np.sort(abs_pro), cdf, color=C_PRO, lw=2, label="Pro-Energy")
        ax_cdf.plot(np.sort(abs_ann), cdf, color=C_ANN, lw=2, label="ANN")
        ax_cdf.axhline(0.90, color="grey", ls=":", lw=1, label="90%")
        ax_cdf.set_xlabel("Absolute Error"); ax_cdf.set_ylabel("CDF")
        ax_cdf.set_title("Cumulative Error CDF", fontweight="bold")
        ax_cdf.legend(fontsize=8)

        fig.savefig(os.path.join(out_dir, "07_dashboard.png"), bbox_inches="tight")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    datafile = sys.argv[1] if len(sys.argv) >= 2 else os.path.join("results", "preprocessed_hourly.csv")
    out_dir  = sys.argv[2] if len(sys.argv) >= 3 else os.path.join("results", "proenergy")
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading data from {datafile}")
    df = load_data(datafile)
    df = preprocess(df, out_dir="results")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"target column {TARGET_COLUMN} not found in data")

    X_df = df.drop(columns=[TARGET_COLUMN])
    y_ser = df[TARGET_COLUMN]

    # ── Chronological day-based split (90% train / 10% test) ─────────────────
    all_dates   = np.array(sorted(set(y_ser.index.date)))
    split_at    = int(len(all_dates) * 0.90)
    train_dates = set(all_dates[:split_at])
    test_dates  = set(all_dates[split_at:])

    idx_dates  = pd.Series(y_ser.index.date, index=y_ser.index)
    train_mask = idx_dates.isin(train_dates).values
    test_mask  = idx_dates.isin(test_dates).values

    X_train = X_df.iloc[train_mask]
    X_test  = X_df.iloc[test_mask]
    y_train = y_ser.iloc[train_mask]
    y_test  = y_ser.iloc[test_mask]

    print(f"train: {train_mask.sum()} samples ({split_at} days)  |  "
          f"test: {test_mask.sum()} samples ({len(test_dates)} days)")

    # ── Load saved scaler & ANN model ────────────────────────────────────────
    import joblib
    scaler_path = os.path.join("results", "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at '{scaler_path}'. "
            "Run pipeline_ann.py first to generate scaler.pkl."
        )
    scaler   = joblib.load(scaler_path)
    print(f"loaded scaler from {scaler_path}")
    X_test_s = scaler.transform(X_test.values)

    model_path = os.path.join("results", "model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run pipeline_ann.py first to generate model.h5."
        )
    ann = tf.keras.models.load_model(model_path, compile=False)
    print(f"loaded ANN model from {model_path}")

    ann_pred    = ann.predict(X_test_s).flatten()
    ann_metrics = metrics(y_test.values, ann_pred)

    # ── Modified ProEnergy — corrected Algorithm 1 ───────────────────────────
    print("fitting Modified Pro-Energy on training data …")
    pro_model = ModifiedProEnergy(alpha=0.5, beta=0.2, D=10, thre=0.4, K_window=6)
    pro_model.fit(y_train)

    print("running Modified Pro-Energy predictions on test set …")
    pro_pred    = pro_model.predict(y_test)           # FIX 5: y_test passed directly
    pro_metrics = metrics(y_test.values, pro_pred)    # FIX 5: compared to real y_test

    # ── Save metrics & predictions ───────────────────────────────────────────
    metrics_df = pd.DataFrame({"ANN": ann_metrics, "ModifiedProEnergy": pro_metrics})
    metrics_df.to_csv(os.path.join(out_dir, "metrics.csv"))
    print("metrics:\n", metrics_df)

    idx = X_test.index
    pred_df = pd.DataFrame({"datetime": idx, "actual": y_test.values,
                             "ann": ann_pred, "proenergy": pro_pred})
    pred_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    # ── Generate all plots ───────────────────────────────────────────────────
    y_true_np = y_test.values
    print("generating plots …")
    plot_timeseries(idx,      y_true_np, ann_pred, pro_pred, out_dir)
    plot_residuals(idx,       y_true_np, ann_pred, pro_pred, out_dir)
    plot_scatter(             y_true_np, ann_pred, pro_pred, ann_metrics, pro_metrics, out_dir)
    plot_error_distribution(  y_true_np, ann_pred, pro_pred, out_dir)
    plot_metrics_bar(         ann_metrics, pro_metrics, out_dir)
    plot_cdf(                 y_true_np, ann_pred, pro_pred, out_dir)
    plot_dashboard(idx,       y_true_np, ann_pred, pro_pred, ann_metrics, pro_metrics, out_dir)

    print(f"\nAll plots saved to: {out_dir}/")
    print(f"  01_timeseries_full.png")
    print(f"  01b_timeseries_one_day.png")
    print(f"  01c_timeseries_window_smooth.png")
    print(f"  01c_timeseries_window_varying.png")
    print(f"  02_residuals_over_time.png")
    print(f"  03_scatter_predicted_vs_actual.png")
    print(f"  04_error_distribution.png")
    print(f"  05_metrics_bar.png")
    print(f"  06_cdf_errors.png")
    print(f"  07_dashboard.png  ← full summary")
    print(f"\nANN  RMSE={ann_metrics['rmse']:.3f}  R²={ann_metrics['r2']:.4f}")
    print(f"Pro  RMSE={pro_metrics['rmse']:.3f}  R²={pro_metrics['r2']:.4f}")


if __name__ == "__main__":
    main()