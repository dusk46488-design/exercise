#!/usr/bin/env python3
"""Data Visualisation Pipeline — RS3 Experimental Data.

Architecture: Backend → Model → Process
    Backend:   CSV/JSON I/O, statistical helpers (reused from 0204).
    Model:     VisualizationPlugin ABC and concrete plot implementations.
    Process:   IPO — load dataset versions, apply visualisation plugins, export figures.

Visualisation plugins:
    history      — Overlaid histograms comparing dataset versions side-by-side.
    aggregate    — Box / violin plots with mean, median, and IQR overlays.
    completeness — Missing-data heatmap, correlation matrix heatmap, entropy bar chart.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless / CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: BACKEND — I/O and statistical helpers (adapted from 0204)
# ═══════════════════════════════════════════════════════════════════════════════

class CSVBackend:
    """Reads CSV files into a pandas DataFrame."""

    @staticmethod
    def load(filepath: str) -> pd.DataFrame:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")
        df = pd.read_csv(filepath)
        print(f"  [load]      {filepath} → {len(df)} rows, {len(df.columns)} columns")
        return df


class JSONBackend:
    """Reads DataFrames and metadata dictionaries from JSON files."""

    @staticmethod
    def load_df(filepath: str) -> pd.DataFrame:
        """Load a JSON file (records or list format) into a DataFrame."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "data" in data:
                df = pd.DataFrame(data["data"], columns=data.get("columns"))
            else:
                df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {filepath}")
        df = df.where(pd.notna(df), None)
        print(f"  [load]      {filepath} → {len(df)} rows, {len(df.columns)} columns")
        return df

    @staticmethod
    def load_dict(filepath: str) -> Dict[str, Any]:
        """Load an arbitrary JSON dictionary (e.g., analysis metadata)."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


class StatsHelper:
    """Stateless statistical helpers for plot annotations."""

    @staticmethod
    def safe_mean(series: pd.Series) -> float:
        valid = series.dropna()
        valid = valid[np.isfinite(valid)]
        return float(valid.mean()) if len(valid) > 0 else 0.0

    @staticmethod
    def safe_median(series: pd.Series) -> float:
        valid = series.dropna()
        valid = valid[np.isfinite(valid)]
        return float(valid.median()) if len(valid) > 0 else 0.0

    @staticmethod
    def safe_std(series: pd.Series) -> float:
        valid = series.dropna()
        valid = valid[np.isfinite(valid)]
        return float(valid.std(ddof=1)) if len(valid) > 1 else 0.0

    @staticmethod
    def entropy(series: pd.Series, bins: int = 20) -> float:
        valid = series.dropna()
        valid = valid[np.isfinite(valid)]
        if len(valid) < 2:
            return 0.0
        counts, _ = np.histogram(valid, bins=bins, density=False)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs)))


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: MODEL — Visualization plugin interface and implementations
# ═══════════════════════════════════════════════════════════════════════════════

class VisualizationPlugin(ABC):
    """Abstract base for visualisation plugins.

    Each plugin receives one or more DataFrames (one per dataset version),
    optional metadata, and produces a matplotlib Figure.

    Plugins are stateless — configuration is set at construction time.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def plot(
        self,
        datasets: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """Produce a matplotlib Figure from one or more datasets.

        Args:
            datasets:  Mapping of version label → DataFrame.
            metadata:  Optional analysis metadata (from 0204 JSON reports).
            figsize:   Figure dimensions in inches.

        Returns:
            A matplotlib Figure ready for `savefig` or `show`.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.name}: {self.description}"


# ── Challenge 1: Dataset History ─────────────────────────────────────────────

class DatasetHistoryPlot(VisualizationPlugin):
    """Plot overlaid or side-by-side histograms comparing dataset versions.

    Visualises the evolution of a numeric column's distribution across the
    data-processing pipeline: generated → filtered → completed → imputed →
    normalised.  Overlaid semi-transparent histograms make distributional
    shifts (from outlier removal, imputation, or scaling) directly visible.

    Design rationale (RS3):
        Plotting retention_score and trial_rt_ms side-by-side across versions
        reveals (a) whether filtering removes the extreme RT tail,
        (b) whether imputation shifts central tendency, and
        (c) how normalisation compresses or stretches the distribution.
    """

    def __init__(
        self,
        column: str,
        bins: int = 30,
        alpha: float = 0.45,
        kde: bool = True,
    ):
        super().__init__(
            name="history",
            description="Overlaid histograms comparing dataset versions",
        )
        self.column = column
        self.bins = bins
        self.alpha = alpha
        self.kde = kde

    def plot(
        self,
        datasets: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        fig, (ax_hist, ax_kde) = plt.subplots(1, 2, figsize=figsize)

        colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(datasets)))

        for (label, df), colour in zip(datasets.items(), colours):
            if self.column not in df.columns:
                print(f"  [history]   skipping '{label}' — column '{self.column}' not found")
                continue
            series = df[self.column].dropna()
            is_num = pd.api.types.is_numeric_dtype(series)
            if is_num:
                series = series[np.isfinite(series)]
            if len(series) == 0:
                print(f"  [history]   skipping '{label}' — no valid data in '{self.column}'")
                continue

            # ── Histogram subplot ──
            ax_hist.hist(
                series, bins=self.bins, alpha=self.alpha,
                color=colour, label=f"{label} (n={len(series)})",
                edgecolor="white", linewidth=0.5,
            )
            if is_num:
                ax_hist.axvline(StatsHelper.safe_mean(series), color=colour,
                               linestyle="--", linewidth=1.2, alpha=0.7)
                ax_hist.axvline(StatsHelper.safe_median(series), color=colour,
                               linestyle=":", linewidth=1.2, alpha=0.7)

            # ── KDE subplot ──
            if self.kde and is_num:
                try:
                    series.plot.kde(ax=ax_kde, color=colour, linewidth=2.0,
                                    label=f"{label} (n={len(series)})")
                except Exception:
                    pass  # KDE fails on constant or degenerate columns

        ax_hist.set_title(f"Histogram: {self.column} across versions", fontsize=13, fontweight="bold")
        ax_hist.set_xlabel(self.column)
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend(fontsize=8, loc="upper right")
        ax_hist.grid(True, alpha=0.3)

        if self.kde:
            ax_kde.set_title(f"KDE: {self.column} across versions", fontsize=13, fontweight="bold")
            ax_kde.set_xlabel(self.column)
            ax_kde.set_ylabel("Density")
            ax_kde.legend(fontsize=8, loc="upper right")
            ax_kde.grid(True, alpha=0.3)
        else:
            ax_kde.set_visible(False)

        fig.suptitle(
            f"Dataset History — {self.column}",
            fontsize=15, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        return fig


class MultiColumnHistoryPlot(VisualizationPlugin):
    """Grid of histograms: one row per dataset version, one column per variable.

    This provides a compact overview of how every numeric variable's
    distribution evolves across the data-processing pipeline.
    """

    def __init__(self, columns: Optional[List[str]] = None, bins: int = 30):
        super().__init__(
            name="history_grid",
            description="Grid of histograms: versions × columns",
        )
        self.columns = columns
        self.bins = bins

    def plot(
        self,
        datasets: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        # Determine common numeric columns if not specified
        if self.columns is None:
            col_sets = []
            for df in datasets.values():
                num_cols = [c for c in df.columns
                            if pd.api.types.is_numeric_dtype(df[c])
                            and not pd.api.types.is_bool_dtype(df[c])]
                col_sets.append(set(num_cols))
            if col_sets:
                self.columns = sorted(col_sets[0].intersection(*col_sets))
            else:
                self.columns = []

        n_versions = len(datasets)
        n_cols = len(self.columns)
        if n_cols == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No common numeric columns found", ha="center", va="center")
            return fig

        if figsize is None:
            figsize = (3.5 * n_cols, 3 * n_versions)

        fig, axes = plt.subplots(
            n_versions, n_cols, figsize=figsize,
            squeeze=False,
        )
        version_labels = list(datasets.keys())

        for i, (label, df) in enumerate(datasets.items()):
            for j, col in enumerate(self.columns):
                ax = axes[i][j]
                if col not in df.columns:
                    ax.text(0.5, 0.5, f"Col '{col}'\nnot found", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="gray")
                    if i == 0:
                        ax.set_title(col, fontsize=10, fontweight="bold")
                    if j == 0:
                        ax.set_ylabel(f"{label}\nn={len(df)}", fontsize=8)
                    continue
                series = df[col].dropna()
                series = series[np.isfinite(series)]
                if len(series) > 0:
                    ax.hist(series, bins=self.bins, color="steelblue",
                            edgecolor="white", linewidth=0.3, alpha=0.85)
                    ax.axvline(series.mean(), color="crimson", linestyle="--",
                              linewidth=1.0, label=f"μ={series.mean():.2f}")
                    ax.axvline(series.median(), color="darkorange", linestyle=":",
                              linewidth=1.0, label=f"med={series.median():.2f}")
                    ax.legend(fontsize=6, loc="upper right")
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9)
                if i == 0:
                    ax.set_title(col, fontsize=10, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(f"{label}\nn={len(df)}", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.2)

        fig.suptitle("Dataset History Grid — versions × variables",
                     fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig


# ── Challenge 2: Aggregates and Data ─────────────────────────────────────────

class AggregatePlot(VisualizationPlugin):
    """Box-and-whisker or violin plot with mean, median, and IQR overlays.

    Uses aggregate metadata from the 0204 analysis pipeline (descriptive,
    outliers) to annotate the plot with statistical summaries extracted from
    the JSON metadata report.

    Design rationale (RS3):
        Overlaying means and medians on the raw-data distribution lets the
        analyst judge at a glance whether the distribution is symmetric
        (mean ≈ median) or skewed (mean ≠ median) — directly relevant to
        the ex-Gaussian shape diagnostics from 0204.
    """

    def __init__(
        self,
        column: str,
        groupby: Optional[str] = None,
        style: str = "box",  # "box" or "violin"
    ):
        super().__init__(
            name="aggregate",
            description="Box/violin plot with mean, median, and IQR overlays",
        )
        self.column = column
        self.groupby = groupby
        self.style = style

    def plot(
        self,
        datasets: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        n_sets = len(datasets)
        fig, axes = plt.subplots(1, n_sets, figsize=figsize, squeeze=False)
        colours = plt.cm.Set2(np.linspace(0, 1, 8))

        for idx, (label, df) in enumerate(datasets.items()):
            ax = axes[0][idx]
            if self.column not in df.columns:
                ax.text(0.5, 0.5, f"Column\n'{self.column}'\nnot found",
                       ha="center", va="center", transform=ax.transAxes,
                       fontsize=9, color="gray")
                ax.set_title(label, fontsize=11, fontweight="bold")
                continue
            series = df[self.column].dropna()
            is_num = pd.api.types.is_numeric_dtype(series)
            if is_num:
                series = series[np.isfinite(series)]

            if len(series) == 0:
                ax.text(0.5, 0.5, f"No valid data", ha="center", va="center")
                ax.set_title(label, fontsize=11, fontweight="bold")
                continue

            if not is_num:
                ax.text(0.5, 0.5, f"Non-numeric\ncolumn",
                       ha="center", va="center", transform=ax.transAxes,
                       fontsize=9, color="gray")
                ax.set_title(label, fontsize=11, fontweight="bold")
                continue

            if self.groupby and self.groupby in df.columns:
                # Grouped plot
                groups = sorted(df[self.groupby].dropna().unique())
                positions = np.arange(len(groups))
                plot_data = [df.loc[df[self.groupby] == g, self.column].dropna() for g in groups]

                if self.style == "violin":
                    parts = ax.violinplot(plot_data, positions=positions,
                                          showmeans=True, showmedians=True)
                else:
                    bp = ax.boxplot(plot_data, positions=positions,
                                    patch_artist=True, widths=0.6)
                    for patch, colour in zip(bp["boxes"], colours[:len(groups)]):
                        patch.set_facecolor(colour)
                        patch.set_alpha(0.6)

                # Overlay individual points with jitter
                for pos, data in zip(positions, plot_data):
                    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(data))
                    ax.scatter(np.full_like(data, pos) + jitter, data,
                              alpha=0.25, s=12, color="black", zorder=3)

                ax.set_xticks(positions)
                ax.set_xticklabels([str(g) for g in groups], rotation=30, ha="right", fontsize=8)
                ax.set_xlabel(self.groupby)

                # Annotate with means
                for pos, data in zip(positions, plot_data):
                    mu = StatsHelper.safe_mean(data)
                    ax.annotate(f"μ={mu:.1f}", (pos + 0.25, mu),
                               fontsize=7, color="crimson", fontweight="bold")

            else:
                # Single distribution
                if self.style == "violin":
                    ax.violinplot([series], positions=[0], showmeans=True, showmedians=True)
                else:
                    bp = ax.boxplot([series], positions=[0], patch_artist=True, widths=0.5)
                    bp["boxes"][0].set_facecolor("steelblue")
                    bp["boxes"][0].set_alpha(0.6)

                # Jittered strip
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(series))
                ax.scatter(np.zeros_like(series) + jitter, series,
                          alpha=0.3, s=14, color="black", zorder=3)

                # Annotation
                mu = StatsHelper.safe_mean(series)
                med = StatsHelper.safe_median(series)
                sigma = StatsHelper.safe_std(series)
                ax.axhline(mu, color="crimson", linestyle="--", linewidth=1.2,
                          label=f"μ = {mu:.2f}")
                ax.axhline(med, color="darkorange", linestyle=":", linewidth=1.2,
                          label=f"median = {med:.2f}")
                ax.legend(fontsize=8)

                # Annotate with metadata if available
                if metadata and "descriptive" in metadata:
                    desc = metadata["descriptive"]
                    if "columns" in desc and self.column in desc["columns"]:
                        info = desc["columns"][self.column]
                        text = (f"skew={info.get('skewness','?')}, "
                                f"kurt={info.get('kurtosis_excess','?')}, "
                                f"miss={info.get('missing_pct',0):.1f}%")
                        ax.set_title(f"{label}\nn={len(df)}  [{text}]",
                                    fontsize=10, fontweight="bold")
                        continue

                ax.set_title(f"{label}\nn={len(df)}", fontsize=10, fontweight="bold")
                ax.set_xticks([])

            ax.set_ylabel(self.column)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Aggregate View — {self.column}"
            + (f" by {self.groupby}" if self.groupby else ""),
            fontsize=14, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        return fig


# ── Challenge 3: Information and Population ──────────────────────────────────

class CompletenessPlot(VisualizationPlugin):
    """Visualise data completeness, correlation structure, and information content.

    Produces a multi-panel figure:
      1. Missing-data heatmap (rows × columns with NaN indicators).
      2. Correlation matrix heatmap (Pearson r for numeric column pairs).
      3. Per-column entropy bar chart.

    Design rationale (RS3):
        The three panels together answer the "coverage" question from
        exercise 0204 Challenge 3: (a) where are the gaps? (b) which
        variables encode redundant information? (c) how uniformly is the
        sample distributed across each variable's support?
    """

    def __init__(self, max_rows_heatmap: int = 200, entropy_bins: int = 20):
        super().__init__(
            name="completeness",
            description="Missing-data heatmap, correlation heatmap, entropy chart",
        )
        self.max_rows_heatmap = max_rows_heatmap
        self.entropy_bins = entropy_bins

    def plot(
        self,
        datasets: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (18, 6),
    ) -> plt.Figure:
        # Use the first dataset for completeness view (or the largest)
        label, df = list(datasets.items())[0]
        if len(datasets) > 1:
            # Find the rawest (generated) version — typically first or with most rows
            for lbl, d in datasets.items():
                if "generated" in lbl.lower() or "raw" in lbl.lower():
                    label, df = lbl, d
                    break

        numeric_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])
                        and not pd.api.types.is_bool_dtype(df[c])]

        fig, (ax_miss, ax_corr, ax_ent) = plt.subplots(1, 3, figsize=figsize)

        # ── Panel 1: Missing-data heatmap ──
        self._plot_missing_heatmap(df, ax_miss, label)

        # ── Panel 2: Correlation heatmap ──
        self._plot_correlation_heatmap(df, numeric_cols, ax_corr)

        # ── Panel 3: Entropy bar chart ──
        self._plot_entropy_bars(df, numeric_cols, ax_ent)

        fig.suptitle(
            f"Completeness & Information — {label}",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        return fig

    def _plot_missing_heatmap(self, df: pd.DataFrame, ax: plt.Axes, label: str):
        """Plot a binary heatmap: present = 0 (light), missing = 1 (dark)."""
        sample_df = df
        if len(df) > self.max_rows_heatmap:
            sample_df = df.sample(self.max_rows_heatmap, random_state=42)
            ax.set_title(f"Missing Data (sample {self.max_rows_heatmap}/{len(df)} rows)",
                        fontsize=11, fontweight="bold")
        else:
            ax.set_title(f"Missing Data ({len(df)} rows)",
                        fontsize=11, fontweight="bold")

        missing_mask = sample_df.isna().astype(int)
        # Only show columns with at least one missing value, or all if few columns
        display_cols = [c for c in missing_mask.columns if missing_mask[c].sum() > 0]
        if not display_cols:
            ax.text(0.5, 0.5, "No missing values in any column",
                   ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        im = ax.imshow(missing_mask[display_cols].T.values,
                       aspect="auto", cmap="Reds", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_xlabel("Row index")
        ax.set_yticks(range(len(display_cols)))
        ax.set_yticklabels(display_cols, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Missing (1 = yes)")

    def _plot_correlation_heatmap(
        self, df: pd.DataFrame, numeric_cols: List[str], ax: plt.Axes
    ):
        """Plot a Pearson correlation heatmap for numeric columns."""
        if len(numeric_cols) < 2:
            ax.text(0.5, 0.5, "Need ≥ 2 numeric columns for correlation",
                   ha="center", va="center", transform=ax.transAxes, fontsize=12)
            return

        corr = df[numeric_cols].corr(method="pearson")
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(numeric_cols, fontsize=7)
        ax.set_title("Pearson Correlation Matrix", fontsize=11, fontweight="bold")

        # Annotate cells with r values
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                val = corr.iloc[i, j]
                colour = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=6, color=colour)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    def _plot_entropy_bars(
        self, df: pd.DataFrame, numeric_cols: List[str], ax: plt.Axes
    ):
        """Plot per-column Shannon entropy as a horizontal bar chart."""
        if not numeric_cols:
            ax.text(0.5, 0.5, "No numeric columns for entropy calculation",
                   ha="center", va="center", transform=ax.transAxes, fontsize=12)
            return

        entropies = {}
        for col in numeric_cols:
            entropies[col] = StatsHelper.entropy(df[col], bins=self.entropy_bins)

        cols_sorted = sorted(entropies, key=entropies.get, reverse=True)
        values = [entropies[c] for c in cols_sorted]

        colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(cols_sorted)))
        bars = ax.barh(range(len(cols_sorted)), values, color=colours, edgecolor="white")
        ax.set_yticks(range(len(cols_sorted)))
        ax.set_yticklabels(cols_sorted, fontsize=8)
        ax.set_xlabel("Shannon Entropy (bits)")
        ax.set_title("Per-Column Shannon Entropy", fontsize=11, fontweight="bold")
        ax.invert_yaxis()

        # Annotate bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                   f"{val:.2f}", va="center", fontsize=8)

        # Reference line: log2(N) for uniform distribution
        max_entropy = np.log2(self.entropy_bins)
        ax.axvline(max_entropy, color="gray", linestyle="--", linewidth=0.8,
                  alpha=0.6, label=f"max ({self.entropy_bins} bins) = {max_entropy:.1f}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis="x")


class CorrelationScatterPlot(VisualizationPlugin):
    """Scatter-plot matrix for key numeric variable pairs with regression lines.

    Complements the correlation heatmap by showing the actual bivariate
    relationships, including condition-coloured points where applicable.
    """

    def __init__(self, columns: Optional[List[str]] = None, hue: Optional[str] = None):
        super().__init__(
            name="correlation_scatter",
            description="Scatter matrix with regression lines",
        )
        self.columns = columns
        self.hue = hue

    def plot(
        self,
        datasets: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        label, df = list(datasets.items())[0]

        numeric_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])
                        and not pd.api.types.is_bool_dtype(df[c])]
        if self.columns:
            numeric_cols = [c for c in self.columns if c in numeric_cols]
        numeric_cols = numeric_cols[:6]  # Limit to avoid overwhelming plot

        if len(numeric_cols) < 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "Need ≥ 2 numeric columns", ha="center", va="center")
            return fig

        n = len(numeric_cols)
        fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)

        for i, col_y in enumerate(numeric_cols):
            for j, col_x in enumerate(numeric_cols):
                ax = axes[i][j]
                if i == j:
                    # Diagonal: histogram
                    series = df[col_y].dropna()
                    ax.hist(series, bins=25, color="steelblue", edgecolor="white",
                           linewidth=0.3, alpha=0.8)
                    ax.set_xlabel(col_y, fontsize=8)
                    ax.set_ylabel("Freq", fontsize=8)
                else:
                    # Off-diagonal: scatter
                    valid = df[[col_x, col_y]].dropna()
                    if self.hue and self.hue in df.columns:
                        for grp_name, grp_df in valid.groupby(self.hue):
                            ax.scatter(grp_df[col_x], grp_df[col_y],
                                      alpha=0.5, s=10, label=str(grp_name))
                        ax.legend(fontsize=5, loc="best")
                    else:
                        ax.scatter(valid[col_x], valid[col_y],
                                  alpha=0.4, s=12, color="steelblue")
                    # Regression line
                    if len(valid) > 2:
                        try:
                            slope, intercept, r_val, _, _ = sp_stats.linregress(
                                valid[col_x], valid[col_y])
                            xs = np.linspace(valid[col_x].min(), valid[col_x].max(), 50)
                            ax.plot(xs, slope * xs + intercept, color="crimson",
                                   linewidth=1.2, linestyle="--")
                            ax.annotate(f"r={r_val:.2f}", xy=(0.05, 0.92),
                                       xycoords="axes fraction", fontsize=7,
                                       color="crimson", fontweight="bold")
                        except Exception:
                            pass
                    ax.set_xlabel(col_x, fontsize=8)
                    ax.set_ylabel(col_y, fontsize=8)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.2)

        fig.suptitle(
            f"Correlation Scatter Matrix — {label}"
            + (f" (hue={self.hue})" if self.hue else ""),
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        return fig


class CompletenessComparisonPlot(VisualizationPlugin):
    """Compare entropy or missing-data rates across dataset versions.

    Bar chart showing how Shannon entropy changes from raw → filtered →
    imputed → normalised for each numeric column.  This directly visualises
    the information-preservation (or information-loss) profile of the
    data-processing pipeline.
    """

    def __init__(self, metric: str = "entropy"):
        super().__init__(
            name="completeness_comparison",
            description="Compare entropy or missing rates across dataset versions",
        )
        self.metric = metric  # "entropy" or "missing_pct"

    def plot(
        self,
        datasets: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (14, 7),
    ) -> plt.Figure:
        # Find common numeric columns
        col_sets = []
        for df in datasets.values():
            num_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])
                        and not pd.api.types.is_bool_dtype(df[c])]
            col_sets.append(frozenset(num_cols))
        common_cols = sorted(
            col_sets[0].intersection(*col_sets) if col_sets else set()
        )
        if not common_cols:
            common_cols = sorted(col_sets[0]) if col_sets else []

        version_labels = list(datasets.keys())
        n_cols = len(common_cols)
        n_versions = len(version_labels)

        x = np.arange(n_cols)
        width = 0.8 / n_versions

        fig, ax = plt.subplots(figsize=figsize)
        colours = plt.cm.Set2(np.linspace(0, 1, max(n_versions, 3)))

        for i, (vlabel, df) in enumerate(datasets.items()):
            values = []
            for col in common_cols:
                if self.metric == "entropy":
                    values.append(StatsHelper.entropy(df[col]))
                else:  # missing_pct
                    n_miss = int(df[col].isna().sum())
                    values.append(100.0 * n_miss / max(len(df), 1))

            bars = ax.bar(x + i * width, values, width, label=vlabel,
                         color=colours[i % len(colours)], edgecolor="white", linewidth=0.5)

            if self.metric == "entropy":
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                           f"{val:.2f}", ha="center", fontsize=7, rotation=90)

        ax.set_xticks(x + width * (n_versions - 1) / 2)
        ax.set_xticklabels(common_cols, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(
            "Shannon Entropy (bits)" if self.metric == "entropy"
            else "Missing (%)"
        )
        ax.set_title(
            f"Per-Column {self.metric.title()} across Dataset Versions",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        return fig


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: PROCESS — VisualizationPipeline with IPO
# ═══════════════════════════════════════════════════════════════════════════════

class VisualizationPipeline:
    """Registry of visualisation plugins with IPO processing engine.

    Extends the 0204 AnalysisPipeline architecture with a VisualizationPlugin
    family.  The pipeline loads one or more dataset versions (representing
    stages of the data-processing history) and applies pluggable visualisation
    plugins to produce publication-quality figures.

    Usage::

        pipeline = VisualizationPipeline()
        pipeline.add_dataset("raw", "output/rs3_tests.csv")
        pipeline.add_dataset("clean", "output/rs3_tests_clean.json")
        fig = pipeline.plot("history", column="retention_score")
        pipeline.save_figure(fig, "output/history_retention.png")
    """

    def __init__(self):
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, Any] = {}
        self._plugins: Dict[str, VisualizationPlugin] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register built-in visualisation plugins."""
        self.register(DatasetHistoryPlot(column="retention_score"))
        self.register(MultiColumnHistoryPlot())
        self.register(AggregatePlot(column="retention_score", groupby="condition"))
        self.register(CompletenessPlot())
        self.register(CorrelationScatterPlot())
        # Note: completeness_comparison's metric is set via kwargs at plot() time
        self.register(CompletenessComparisonPlot(metric="entropy"))

    def register(self, plugin: VisualizationPlugin) -> None:
        self._plugins[plugin.name] = plugin
        print(f"  [register]  viz '{plugin.name}' -- {plugin.description}")

    def add_dataset(self, label: str, filepath: str) -> None:
        """Load a dataset from CSV or JSON and associate it with *label*.

        Labels typically represent stages in the data-processing history:
        'generated', 'filtered', 'completed', 'imputed', 'normalised'.
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".json":
            self._datasets[label] = JSONBackend.load_df(filepath)
        else:
            self._datasets[label] = CSVBackend.load(filepath)

    def add_dataframe(self, label: str, df: pd.DataFrame) -> None:
        """Register an already-loaded DataFrame under *label*."""
        self._datasets[label] = df.copy()

    def load_metadata(self, filepath: str) -> None:
        """Load analysis metadata from a 0204 JSON report."""
        self._metadata = JSONBackend.load_dict(filepath)
        print(f"  [metadata]  loaded from {filepath}")

    def list_datasets(self) -> List[str]:
        return list(self._datasets.keys())

    def list_plugins(self) -> List[str]:
        return list(self._plugins.keys())

    def get_plugin(self, name: str) -> VisualizationPlugin:
        if name not in self._plugins:
            available = ", ".join(self._plugins.keys())
            raise KeyError(f"Unknown plugin '{name}'. Available: {available}")
        return self._plugins[name]

    def plot(
        self,
        plugin_name: str,
        **kwargs,
    ) -> plt.Figure:
        """Apply a named visualisation plugin to the loaded datasets.

        Args:
            plugin_name:  Name of the registered VisualizationPlugin to use.
            **kwargs:     Passed through to the plugin's plot() method
                          (e.g., column, groupby, figsize).

        Returns:
            A matplotlib Figure.
        """
        if not self._datasets:
            raise RuntimeError("No datasets loaded. Call add_dataset() first.")

        plugin = self.get_plugin(plugin_name)

        # Override plugin attributes from kwargs
        for attr, val in kwargs.items():
            if hasattr(plugin, attr):
                setattr(plugin, attr, val)

        print(f"\n  [plot]      '{plugin_name}' on {len(self._datasets)} dataset(s): "
              f"{', '.join(self._datasets.keys())}")
        metadata = self._metadata if self._metadata else None
        return plugin.plot(self._datasets, metadata=metadata)

    @staticmethod
    def save_figure(fig: plt.Figure, filepath: str, dpi: int = 150) -> str:
        """Save a matplotlib Figure to disk.

        Args:
            fig:      The Figure to save.
            filepath: Destination path (extension determines format).
            dpi:      Resolution in dots per inch.

        Returns:
            Absolute path to the saved file.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  [save]      {os.path.abspath(filepath)}")
        plt.close(fig)
        return os.path.abspath(filepath)


# ═══════════════════════════════════════════════════════════════════════════════
# High-Level Convenience: Run a complete visualisation suite
# ═══════════════════════════════════════════════════════════════════════════════

def run_visualisation_suite(
    dataset_specs: List[Tuple[str, str]],
    output_dir: str = "output",
    metadata_path: Optional[str] = None,
    columns_of_interest: Optional[List[str]] = None,
) -> List[str]:
    """Run a comprehensive suite of visualisations on multiple dataset versions.

    This is the primary entry point for the exercise challenges.  It loads
    every dataset version, optionally loads analysis metadata, and produces:
      - History plots (histogram + KDE) for each column of interest
      - A multi-column history grid
      - Aggregate plots (box + violin) for each column
      - Completeness plots (missing heatmap, correlation, entropy)
      - Completeness comparison across versions
      - Correlation scatter matrix

    Args:
        dataset_specs:      List of (label, filepath) tuples.
        output_dir:         Directory to write figure files into.
        metadata_path:      Optional path to a 0204 analysis metadata JSON.
        columns_of_interest:Columns to focus history/aggregate plots on.

    Returns:
        List of absolute paths to the saved figure files.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    pipeline = VisualizationPipeline()

    # Load all dataset versions
    for label, path in dataset_specs:
        pipeline.add_dataset(label, path)

    if metadata_path:
        pipeline.load_metadata(metadata_path)

    # Auto-discover numeric columns if not specified
    if columns_of_interest is None:
        first_df = list(pipeline._datasets.values())[0]
        columns_of_interest = [
            c for c in first_df.columns
            if pd.api.types.is_numeric_dtype(first_df[c])
            and not pd.api.types.is_bool_dtype(first_df[c])
        ][:5]  # Limit to first 5 for manageability

    print(f"\n{'='*60}")
    print(f"VISUALISATION SUITE: {len(dataset_specs)} dataset(s)")
    print(f"Columns of interest: {columns_of_interest}")
    print(f"{'='*60}")

    # ── Challenge 1: Dataset History ──
    for col in columns_of_interest:
        try:
            fig = pipeline.plot("history", column=col)
            path = pipeline.save_figure(fig, f"{output_dir}/history_{col}.png")
            saved.append(path)
        except Exception as exc:
            print(f"  [skip]      history:{col} — {exc}")

    try:
        fig = pipeline.plot("history_grid")
        path = pipeline.save_figure(fig, f"{output_dir}/history_grid.png")
        saved.append(path)
    except Exception as exc:
        print(f"  [skip]      history_grid — {exc}")

    # ── Challenge 2: Aggregates ──
    for col in columns_of_interest:
        for style in ("box", "violin"):
            try:
                fig = pipeline.plot("aggregate", column=col, style=style)
                path = pipeline.save_figure(fig, f"{output_dir}/aggregate_{style}_{col}.png")
                saved.append(path)
            except Exception as exc:
                print(f"  [skip]      aggregate:{style}:{col} — {exc}")

    # ── Challenge 3: Completeness ──
    try:
        fig = pipeline.plot("completeness")
        path = pipeline.save_figure(fig, f"{output_dir}/completeness.png")
        saved.append(path)
    except Exception as exc:
        print(f"  [skip]      completeness — {exc}")

    try:
        fig = pipeline.plot("correlation_scatter")
        path = pipeline.save_figure(fig, f"{output_dir}/correlation_scatter.png")
        saved.append(path)
    except Exception as exc:
        print(f"  [skip]      correlation_scatter — {exc}")

    for metric in ("entropy", "missing_pct"):
        try:
            fig = pipeline.plot("completeness_comparison", metric=metric)
            path = pipeline.save_figure(fig, f"{output_dir}/completeness_comparison_{metric}.png")
            saved.append(path)
        except Exception as exc:
            print(f"  [skip]      completeness_comparison:{metric} — {exc}")

    print(f"\n{'='*60}")
    print(f"DONE — {len(saved)} figure(s) saved to {os.path.abspath(output_dir)}/")
    print(f"{'='*60}\n")
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Data Visualisation Pipeline — RS3 Experimental Data",
        epilog="Plugins: history, history_grid, aggregate, completeness, "
               "correlation_scatter, completeness_comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Dataset files to visualise (CSV or JSON).  "
             "Prefix with 'label=' to name the version, e.g. "
             "'raw=output/rs3_tests.csv' 'clean=output/rs3_tests_clean.json'.",
    )
    parser.add_argument(
        "-p", "--plugin",
        default="history",
        help="Visualisation plugin to apply (default: history).",
    )
    parser.add_argument(
        "-c", "--column",
        default=None,
        help="Column to plot (for history, aggregate plugins).",
    )
    parser.add_argument(
        "-g", "--groupby",
        default=None,
        help="Grouping column for aggregate plots.",
    )
    parser.add_argument(
        "--style",
        default="box",
        choices=["box", "violin"],
        help="Plot style for aggregate plugin (default: box).",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for the figure (default: output/<plugin>_<column>.png).",
    )
    parser.add_argument(
        "-m", "--metadata",
        default=None,
        help="Path to a 0204 analysis metadata JSON report.",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run the full visualisation suite (Challenges 1–3).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered visualisation plugins and exit.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure resolution in DPI (default: 150).",
    )

    args = parser.parse_args()

    pipeline = VisualizationPipeline()

    if args.list:
        print("\nRegistered visualisation plugins:")
        for name in pipeline.list_plugins():
            p = pipeline.get_plugin(name)
            print(f"  {name:26s} -- {p.description}")
        return

    # ── Parse labelled inputs ──
    if not args.inputs:
        parser.error("At least one input file is required.")

    dataset_specs: List[Tuple[str, str]] = []
    for token in args.inputs:
        if "=" in token:
            label, path = token.split("=", 1)
        else:
            path = token
            # Auto-generate label from filename
            stem = os.path.splitext(os.path.basename(token))[0]
            label = stem.replace("rs3_", "").replace("tests", "tests")
            if len(dataset_specs) == 0:
                label = "generated"
            elif len(dataset_specs) == 1:
                label = "filtered"
            elif len(dataset_specs) == 2:
                label = "completed"
            elif len(dataset_specs) == 3:
                label = "imputed"
            else:
                label = f"version_{len(dataset_specs)}"
        dataset_specs.append((label, path))

    # ── Suite mode: run all visualisations ──
    if args.suite:
        run_visualisation_suite(
            dataset_specs=dataset_specs,
            output_dir="output",
            metadata_path=args.metadata,
        )
        return

    # ── Single-plot mode ──
    for label, path in dataset_specs:
        pipeline.add_dataset(label, path)

    if args.metadata:
        pipeline.load_metadata(args.metadata)

    # Determine output path
    if args.output is None:
        col_suffix = f"_{args.column}" if args.column else ""
        args.output = f"output/{args.plugin}{col_suffix}.png"

    # Build kwargs for the plugin
    kwargs: Dict[str, Any] = {}
    if args.column:
        kwargs["column"] = args.column
    if args.groupby:
        kwargs["groupby"] = args.groupby
    if args.style and args.plugin == "aggregate":
        kwargs["style"] = args.style

    fig = pipeline.plot(args.plugin, **kwargs)
    pipeline.save_figure(fig, args.output, dpi=args.dpi)
    print(f"\nFigure saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
