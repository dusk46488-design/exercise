#!/usr/bin/env python3
"""Data Exploration & Analysis Pipeline -- RS3 Experimental Data.

Architecture: Backend → Model → Process
    Backend:   CSV/JSON I/O, extended statistical helpers (skew, kurtosis, outlier detection).
    Model:     FilterPlugin (transforms), AnalysisPlugin (produces summary dicts).
    Process:   IPO -- load data, apply filter chain, run analyses, export metadata JSON.

Filter plugins (transforms -- DataFrame → DataFrame):
    zscore      -- Z-score standardisation: (x - mu) / sigma.
    minmax      -- Min-max scaling to [0, 1].
    robust      -- Robust scaling: (x - median) / IQR.

Analysis plugins (aggregations -- DataFrame → dict):
    descriptive -- Descriptive statistics per numeric column.
    outliers    -- Outlier detection via IQR and Z-score methods.
    completeness-- Correlation matrix, missing-data patterns, entropy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: BACKEND -- I/O and statistical helpers
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
    """Reads and writes DataFrames to/from JSON files.

    Extended from 0203 with a JSON *reader* so the pipeline can ingest
    previously cleaned JSON output as well as raw CSV.
    """

    @staticmethod
    def save(df: pd.DataFrame, filepath: str, orient: str = "records") -> str:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        sanitised = df.where(pd.notna(df), None)
        data = json.loads(sanitised.to_json(orient="records", date_format="iso"))
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  [save]      {os.path.abspath(filepath)} ← {len(df)} rows, {len(df.columns)} columns")
        return os.path.abspath(filepath)

    @staticmethod
    def load(filepath: str) -> pd.DataFrame:
        """Load a JSON file (records or list format) into a DataFrame."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Handle 'split' or 'index' orientations
            if "data" in data:
                df = pd.DataFrame(data["data"], columns=data.get("columns"))
            else:
                df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {filepath}")
        # Convert null back to NaN for consistent handling
        df = df.where(pd.notna(df), None)
        print(f"  [load]      {filepath} → {len(df)} rows, {len(df.columns)} columns")
        return df

    @staticmethod
    def save_dict(data: Dict[str, Any], filepath: str) -> str:
        """Write an arbitrary dictionary as a pretty-printed JSON file.

        Used for aggregation metadata output.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        def _default(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return str(obj)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_default)
        print(f"  [save]      {os.path.abspath(filepath)} ← metadata report")
        return os.path.abspath(filepath)


class StatsHelper:
    """Stateless statistical helpers for aggregation and outlier detection."""

    @staticmethod
    def safe_median(series: pd.Series) -> float:
        valid = series.dropna()
        valid = valid[np.isfinite(valid)]
        return float(valid.median()) if len(valid) > 0 else 0.0

    @staticmethod
    def safe_mean(series: pd.Series) -> float:
        valid = series.dropna()
        valid = valid[np.isfinite(valid)]
        return float(valid.mean()) if len(valid) > 0 else 0.0

    @staticmethod
    def safe_std(series: pd.Series) -> float:
        valid = series.dropna()
        valid = valid[np.isfinite(valid)]
        return float(valid.std(ddof=1)) if len(valid) > 1 else 0.0

    @staticmethod
    def safe_skew(series: pd.Series) -> float:
        """Fisher-Pearson skewness coefficient."""
        valid = series.dropna()
        valid = valid[np.isfinite(valid)]
        if len(valid) < 3:
            return 0.0
        return float(sp_stats.skew(valid, bias=False))

    @staticmethod
    def safe_kurtosis(series: pd.Series) -> float:
        """Excess kurtosis (0 = normal)."""
        valid = series.dropna()
        valid = valid[np.isfinite(valid)]
        if len(valid) < 4:
            return 0.0
        return float(sp_stats.kurtosis(valid, bias=False))

    @staticmethod
    def entropy(series: pd.Series, bins: int = 20) -> float:
        """Shannon entropy of a numeric series (binned histogram estimate)."""
        valid = series.dropna()
        valid = valid[np.isfinite(valid)]
        if len(valid) < 2:
            return 0.0
        counts, _ = np.histogram(valid, bins=bins, density=False)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs)))


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: MODEL -- Plugin interfaces and implementations
# ═══════════════════════════════════════════════════════════════════════════════

class FilterPlugin(ABC):
    """Abstract base for DataFrame transform plugins.

    Each plugin receives a DataFrame and returns a transformed DataFrame.
    Plugins are stateless -- configuration is set at construction time.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def __repr__(self) -> str:
        return f"{self.name}: {self.description}"


class AnalysisPlugin(ABC):
    """Abstract base for aggregation / analysis plugins.

    Unlike FilterPlugin, these produce a *summary dictionary* (key → value),
    which is serialised to JSON as metadata.  They do not modify the DataFrame.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        ...

    def __repr__(self) -> str:
        return f"{self.name}: {self.description}"


# ── Filter Plugins: Normalisation ────────────────────────────────────────────

class ZScoreFilter(FilterPlugin):
    """Z-score standardisation: z = (x - mu) / sigma.

    After transformation each column has mean ≈ 0 and std ≈ 1.
    Sensitive to outliers -- extreme values inflate sigma and shrink all z-scores.

    Design rationale (RS3):
        Z-score is appropriate when the data are approximately symmetric and
        the analyst intends to use distance-based methods (PCA, k-means, SVM)
        that assume equal-variance features.  For the RS3 retention scores
        (roughly logistic-normal) and log-transformed RTs, z-score reduces
        the risk of high-variance features dominating the model.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        super().__init__(
            name="zscore",
            description="Z-score standardisation: (x - mean) / std",
        )
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        cols = self._numeric_columns(df_out)
        for col in cols:
            if self.columns is not None and col not in self.columns:
                continue
            mu = StatsHelper.safe_mean(df_out[col])
            sigma = StatsHelper.safe_std(df_out[col])
            if sigma > 1e-10:
                df_out[col] = (df_out[col] - mu) / sigma
            else:
                df_out[col] = 0.0  # constant column
        print(f"  [zscore]    standardised {len(cols)} numeric columns")
        return df_out

    @staticmethod
    def _numeric_columns(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])]


class MinMaxFilter(FilterPlugin):
    """Min-max scaling to [0, 1]: x' = (x - min) / (max - min).

    Preserves the shape of the distribution and the relative distances between
    observations.  Sensitive to outliers -- a single extreme value compresses
    all other observations into a narrow band.

    Design rationale (RS3):
        Min-max is appropriate when boundedness matters (e.g., neural-network
        inputs with sigmoid/tanh activations) or when the analyst needs to
        preserve zero entries (sparse features).  For JOL predictions (0–100)
        and retention scores (0–100), min-max maps naturally to the
        [0, 1] interval without altering rank order.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        super().__init__(
            name="minmax",
            description="Min-max scaling to [0, 1]: (x - min) / (max - min)",
        )
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        cols = ZScoreFilter._numeric_columns(df_out)
        for col in cols:
            if self.columns is not None and col not in self.columns:
                continue
            col_min = df_out[col].min(skipna=True)
            col_max = df_out[col].max(skipna=True)
            denom = col_max - col_min
            if denom > 1e-10:
                df_out[col] = (df_out[col] - col_min) / denom
            else:
                df_out[col] = 0.5
        print(f"  [minmax]    scaled {len(cols)} numeric columns to [0, 1]")
        return df_out


class RobustFilter(FilterPlugin):
    """Robust scaling: x' = (x - median) / IQR.

    Uses the median and inter-quartile range instead of mean and std, making
    the transformation resistant to outliers.  For normally-distributed data,
    robust scaling approximates z-score with ~15 % efficiency loss; for
    heavy-tailed data (ex-Gaussian RTs), it is substantially more stable.

    Design rationale (RS3):
        The RS3 trial RTs follow an ex-Gaussian distribution with a heavy
        right tail (τ = 100–400 ms).  Robust scaling prevents the few
        extreme RTs (> 10 s) from dominating the scale estimate, making it
        the recommended normalisation for distance-based models trained on
        raw (un-transformed) RT data.  After robust scaling, outliers remain
        visible as large (but finite) absolute values.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        super().__init__(
            name="robust",
            description="Robust scaling: (x - median) / IQR",
        )
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        cols = ZScoreFilter._numeric_columns(df_out)
        for col in cols:
            if self.columns is not None and col not in self.columns:
                continue
            med = StatsHelper.safe_median(df_out[col])
            q1 = float(df_out[col].quantile(0.25))
            q3 = float(df_out[col].quantile(0.75))
            iqr = q3 - q1
            if iqr > 1e-10:
                df_out[col] = (df_out[col] - med) / iqr
            else:
                df_out[col] = 0.0
        print(f"  [robust]    scaled {len(cols)} numeric columns (median/IQR)")
        return df_out


# ── Analysis Plugins: Aggregation ────────────────────────────────────────────

class DescriptiveAnalysis(AnalysisPlugin):
    """Compute descriptive statistics for every numeric column.

    Reports: count, missing (count + %), mean, median, std, min, max,
    skewness, excess kurtosis.

    These statistics characterise the central tendency, dispersion, and shape
    of each variable's distribution -- directly relevant to the experiment's
    distributional assumptions (ex-Gaussian RTs, logistic scores, truncated-
    normal JOLs).
    """

    def __init__(self, groupby: Optional[str] = None):
        super().__init__(
            name="descriptive",
            description="Descriptive statistics per numeric column",
        )
        self.groupby = groupby

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
        report: Dict[str, Any] = {
            "analysis": "descriptive",
            "n_rows": len(df),
            "n_columns_total": len(df.columns),
            "n_columns_numeric": len(numeric_cols),
            "groupby": self.groupby,
        }

        if self.groupby and self.groupby in df.columns:
            groups: Dict[str, Any] = {}
            for grp_name, grp_df in df.groupby(self.groupby):
                groups[str(grp_name)] = self._describe_frame(grp_df, numeric_cols)
            report["groups"] = groups
        else:
            report["columns"] = self._describe_frame(df, numeric_cols)

        print(f"  [descriptive] {len(numeric_cols)} numeric columns summarised"
              + (f" grouped by '{self.groupby}'" if self.groupby and self.groupby in df.columns else ""))
        return report

    def _describe_frame(self, df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
        result = {}
        for col in cols:
            series = df[col]
            n = len(series)
            n_miss = int(series.isna().sum())
            result[col] = {
                "count": n - n_miss,
                "missing": n_miss,
                "missing_pct": round(100.0 * n_miss / n, 2) if n > 0 else 0.0,
                "mean": round(StatsHelper.safe_mean(series), 4),
                "median": round(StatsHelper.safe_median(series), 4),
                "std": round(StatsHelper.safe_std(series), 4),
                "min": round(float(series.min(skipna=True)), 4) if n_miss < n else None,
                "max": round(float(series.max(skipna=True)), 4) if n_miss < n else None,
                "skewness": round(StatsHelper.safe_skew(series), 4),
                "kurtosis_excess": round(StatsHelper.safe_kurtosis(series), 4),
            }
        return result


class OutlierAnalysis(AnalysisPlugin):
    """Detect outliers using two complementary methods.

    1. **IQR method** (Tukey, 1977): values outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR].
       Non-parametric; ~0.7 % false-positive rate for normal data.
    2. **Z-score method**: values with |z| > 3.
       Parametric; assumes approximate normality.

    Reports: outlier counts, percentages, and extreme-value ranges per column.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        super().__init__(
            name="outliers",
            description="Outlier detection via IQR and Z-score methods",
        )
        self.columns = columns

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
        if self.columns is not None:
            numeric_cols = [c for c in numeric_cols if c in self.columns]

        report: Dict[str, Any] = {
            "analysis": "outliers",
            "n_rows": len(df),
        }
        columns_report: Dict[str, Any] = {}

        for col in numeric_cols:
            series = df[col].dropna()
            series = series[np.isfinite(series)]
            n = len(series)

            if n < 4:
                columns_report[col] = {"error": "insufficient data for outlier analysis"}
                continue

            # IQR method
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            iqr_outliers = int(((series < lower_fence) | (series > upper_fence)).sum())

            # Z-score method
            mu = StatsHelper.safe_mean(series)
            sigma = StatsHelper.safe_std(series)
            z_lower = -3.0
            z_upper = 3.0
            if sigma > 1e-10:
                z_scores = (series - mu) / sigma
                z_outliers = int(((z_scores < z_lower) | (z_scores > z_upper)).sum())
            else:
                z_outliers = 0

            columns_report[col] = {
                "n": n,
                "iqr_method": {
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "iqr": round(iqr, 4),
                    "lower_fence": round(lower_fence, 4),
                    "upper_fence": round(upper_fence, 4),
                    "outlier_count": iqr_outliers,
                    "outlier_pct": round(100.0 * iqr_outliers / n, 2),
                },
                "zscore_method": {
                    "mean": round(mu, 4),
                    "std": round(sigma, 4),
                    "threshold": 3.0,
                    "outlier_count": z_outliers,
                    "outlier_pct": round(100.0 * z_outliers / n, 2),
                },
            }

        report["columns"] = columns_report
        print(f"  [outliers]  {len(numeric_cols)} columns analysed for outliers")
        return report


class CompletenessAnalysis(AnalysisPlugin):
    """Explore data completeness through correlation, missing-data patterns, and
    information-theoretic measures.

    1. **Correlation matrix** -- Pearson r for all numeric column pairs.
       Identifies redundancy and potential multicollinearity.
    2. **Missing-data pattern** -- Fraction of rows with ≥1 missing value,
       and per-column missing rate.  Addresses the "coverage" question
       in Challenge 3.
    3. **Entropy** -- Shannon entropy (binned estimate) per numeric column.
       High entropy → more uniform / less informative; low entropy →
       more concentrated / potentially under-varying.  Normalisation
       changes entropy -- comparing pre/post normalisation entropy
       quantifies how much the transformation reshaped the distribution.
    """

    def __init__(self, entropy_bins: int = 20):
        super().__init__(
            name="completeness",
            description="Correlation matrix, missing-data patterns, and entropy",
        )
        self.entropy_bins = entropy_bins

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
        report: Dict[str, Any] = {
            "analysis": "completeness",
            "n_rows": len(df),
            "n_columns_total": len(df.columns),
            "n_columns_numeric": len(numeric_cols),
        }

        # ── Missing-data overview ──
        n_rows_with_missing = int(df.isna().any(axis=1).sum())
        report["missing_overview"] = {
            "rows_with_missing": n_rows_with_missing,
            "rows_complete": len(df) - n_rows_with_missing,
            "completeness_pct": round(100.0 * (len(df) - n_rows_with_missing) / max(len(df), 1), 2),
            "total_missing_cells": int(df.isna().sum().sum()),
        }

        # Per-column missing rates
        col_missing = {}
        for col in df.columns:
            n_miss = int(df[col].isna().sum())
            if n_miss > 0:
                col_missing[col] = {
                    "missing_count": n_miss,
                    "missing_pct": round(100.0 * n_miss / max(len(df), 1), 2),
                }
        report["missing_by_column"] = col_missing

        # ── Correlation matrix ──
        if len(numeric_cols) >= 2:
            corr_df = df[numeric_cols].corr(method="pearson")
            corr_dict: Dict[str, Dict[str, float]] = {}
            for col_a in numeric_cols:
                corr_dict[col_a] = {}
                for col_b in numeric_cols:
                    if col_a != col_b:
                        val = corr_df.loc[col_a, col_b]
                        corr_dict[col_a][col_b] = round(float(val), 4) if pd.notna(val) else None
            report["correlation_matrix"] = corr_dict
            print(f"  [completeness] correlation matrix: {len(numeric_cols)}×{len(numeric_cols)}")
        else:
            report["correlation_matrix"] = None

        # ── Entropy ──
        entropy_report = {}
        for col in numeric_cols:
            series = df[col]
            entropy_report[col] = round(StatsHelper.entropy(series, bins=self.entropy_bins), 4)
        report["entropy"] = entropy_report

        print(f"  [completeness] {len(col_missing)} columns with missing data, "
              f"entropy computed for {len(numeric_cols)} columns")
        return report


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: PROCESS -- AnalysisPipeline with IPO
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisPipeline:
    """Registry of filter and analysis plugins, with IPO processing engine.

    Extends the 0203 FilterPipeline with:
      - JSON input backend (reads previously cleaned JSON)
      - Analysis plugins (aggregation, outliers, completeness)
      - Normalisation filter plugins (zscore, minmax, robust)
      - Metadata report output in JSON
    """

    def __init__(self):
        self._filters: Dict[str, FilterPlugin] = {}
        self._analyses: Dict[str, AnalysisPlugin] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register built-in normalisation filters and analysis plugins."""
        # Normalisation filters (FilterPlugin)
        self.register_filter(ZScoreFilter())
        self.register_filter(MinMaxFilter())
        self.register_filter(RobustFilter())
        # Analysis plugins (AnalysisPlugin)
        self.register_analysis(DescriptiveAnalysis())
        self.register_analysis(OutlierAnalysis())
        self.register_analysis(CompletenessAnalysis())

    def register_filter(self, plugin: FilterPlugin) -> None:
        self._filters[plugin.name] = plugin
        print(f"  [register]  filter '{plugin.name}' -- {plugin.description}")

    def register_analysis(self, plugin: AnalysisPlugin) -> None:
        self._analyses[plugin.name] = plugin
        print(f"  [register]  analysis '{plugin.name}' -- {plugin.description}")

    def list_filters(self) -> List[str]:
        return list(self._filters.keys())

    def list_analyses(self) -> List[str]:
        return list(self._analyses.keys())

    def get_filter(self, name: str) -> FilterPlugin:
        if name not in self._filters:
            available = ", ".join(self._filters.keys())
            raise KeyError(f"Unknown filter '{name}'. Available: {available}")
        return self._filters[name]

    def get_analysis(self, name: str) -> AnalysisPlugin:
        if name not in self._analyses:
            available = ", ".join(self._analyses.keys())
            raise KeyError(f"Unknown analysis '{name}'. Available: {available}")
        return self._analyses[name]

    # ── IPO Process Function ────────────────────────────────────────────────

    def process(
        self,
        input_path: str,
        output_json: str,
        filters: Optional[List[str]] = None,
        analyses: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        groupby: Optional[str] = None,
        report_json: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run the IPO pipeline: load → filter → analyse → report.

        Args:
            input_path:   Path to input CSV or JSON file.
            output_json:  Path for the (optionally normalised) output JSON.
            filters:      Ordered list of normalisation filter names to apply.
            analyses:     Ordered list of analysis plugin names to run.
            columns:      Subset of columns to operate on.
            groupby:      Column name for grouped descriptive statistics.
            report_json:  Path for the analysis metadata report JSON.

        Returns:
            (processed DataFrame, aggregated report dict).
        """
        # ── Input ──
        print(f"\n=== IPO Pipeline: {input_path} → {output_json} ===")
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".json":
            df = JSONBackend.load(input_path)
        else:
            df = CSVBackend.load(input_path)
        print(f"  [input]     columns: {list(df.columns)}")

        # ── Process: normalisation filters ──
        report: Dict[str, Any] = {"pipeline": "analysis_0204", "input": input_path}
        filters = filters or []
        for filter_name in filters:
            plugin = self.get_filter(filter_name)
            if columns is not None and hasattr(plugin, "columns"):
                plugin.columns = columns
            print(f"  [process]   applying filter: {plugin.name}")
            df = plugin.apply(df)
            report[f"filter_{filter_name}_applied"] = True

        # ── Save processed data ──
        JSONBackend.save(df, output_json)

        # ── Process: analysis plugins ──
        analyses = analyses or []
        for analysis_name in analyses:
            plugin = self.get_analysis(analysis_name)
            if isinstance(plugin, DescriptiveAnalysis) and groupby is not None:
                plugin.groupby = groupby
            if columns is not None and hasattr(plugin, "columns"):
                plugin.columns = columns
            print(f"  [process]   running analysis: {plugin.name}")
            result = plugin.analyze(df)
            report[analysis_name] = result

        # ── Output: metadata report ──
        if report_json and analyses:
            JSONBackend.save_dict(report, report_json)

        print(f"=== Done: {len(df)} rows written to {output_json} ===\n")
        return df, report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Data Exploration & Analysis Pipeline -- RS3 Experimental Data",
        epilog="Filters: zscore, minmax, robust.  "
               "Analyses: descriptive, outliers, completeness.  "
               "Output: processed JSON + optional metadata report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to the input file (CSV or JSON).",
    )
    parser.add_argument(
        "-o", "--output",
        help="Path for the output JSON file (default: <input_stem>_explored.json).",
    )
    parser.add_argument(
        "-f", "--filter",
        dest="filters",
        action="append",
        choices=["zscore", "minmax", "robust"],
        help="Normalisation filter to apply.  May be repeated to chain "
             "(e.g., -f robust).  Available: zscore, minmax, robust.",
    )
    parser.add_argument(
        "-a", "--analysis",
        dest="analyses",
        action="append",
        choices=["descriptive", "outliers", "completeness"],
        help="Analysis plugin to run.  May be repeated (e.g., -a descriptive -a outliers).",
    )
    parser.add_argument(
        "-c", "--columns",
        nargs="*",
        default=None,
        help="Subset of columns to operate on (default: all numeric columns).",
    )
    parser.add_argument(
        "-g", "--groupby",
        default=None,
        help="Column name for grouped descriptive statistics (e.g., 'condition').",
    )
    parser.add_argument(
        "-r", "--report",
        default=None,
        help="Path for the analysis metadata report JSON (default: <output_stem>_report.json).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered filter and analysis plugins and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility (default: 2026).",
    )

    args = parser.parse_args()

    # ── Build pipeline ──
    pipeline = AnalysisPipeline()

    if args.list:
        print("\nRegistered filter plugins:")
        for name in pipeline.list_filters():
            p = pipeline.get_filter(name)
            print(f"  {name:14s} -- {p.description}")
        print("\nRegistered analysis plugins:")
        for name in pipeline.list_analyses():
            p = pipeline.get_analysis(name)
            print(f"  {name:14s} -- {p.description}")
        return

    # ── Validate ──
    if args.input is None:
        parser.error("Input file is required (unless using --list).")

    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{stem}_explored.json"

    if args.report is None and args.analyses:
        stem = os.path.splitext(os.path.basename(args.output))[0]
        args.report = f"{stem}_report.json"

    # ── Run IPO pipeline ──
    df, report = pipeline.process(
        input_path=args.input,
        output_json=args.output,
        filters=args.filters or [],
        analyses=args.analyses or [],
        columns=args.columns,
        groupby=args.groupby,
        report_json=args.report if args.analyses else None,
    )

    # ── Print summary ──
    if args.analyses:
        print("=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        for analysis_name in args.analyses:
            if analysis_name in report:
                result = report[analysis_name]
                print(f"\n--- {analysis_name} ---")
                print(json.dumps(result, indent=2, default=str))
    else:
        print("No analyses requested.  Use -a/--analysis to add: descriptive, outliers, completeness.")


if __name__ == "__main__":
    main()
