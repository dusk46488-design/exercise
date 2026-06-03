#!/usr/bin/env python3
"""Data Filtering & Cleaning Pipeline — RS3 Experimental Data.

Architecture: Backend → Model → Process
    Backend:   CSV loader, JSON writer, statistical helpers for imputation.
    Model:     Filter plugin interface and concrete filter implementations.
    Process:   IPO function — load CSV, apply filter chain, export JSON.

Filter plugins:
    drop       — Remove rows with malformed or missing values.
    heuristic  — Complete missing data using rule-of-thumb (median/mode).
    imputation — Infer missing values from the complete subset of the data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: BACKEND — I/O and statistical helpers
# ═══════════════════════════════════════════════════════════════════════════════

class CSVBackend:
    """Reads CSV files produced by the RS3 data generator into a pandas DataFrame."""

    @staticmethod
    def load(filepath: str) -> pd.DataFrame:
        """Load a CSV file and return a DataFrame.

        Args:
            filepath: Path to the CSV file.

        Returns:
            DataFrame with the loaded data.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")
        df = pd.read_csv(filepath)
        print(f"  [load]      {filepath} → {len(df)} rows, {len(df.columns)} columns")
        return df


class JSONBackend:
    """Writes pandas DataFrames to JSON files.

    This extends the pipeline architecture with a JSON output backend,
    complementing the CSV backend used in exercises 0201–02.
    """

    @staticmethod
    def save(df: pd.DataFrame, filepath: str, orient: str = "records") -> str:
        """Write a DataFrame to a JSON file.

        Args:
            df: DataFrame to export.
            filepath: Destination path for the JSON file.
            orient: JSON orientation — 'records' (list of objects),
                    'split' (data/columns/index), or 'index' (dict of rows).

        Returns:
            The absolute path to the written file.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = JSONBackend._sanitise(df)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  [save]      {os.path.abspath(filepath)} ← {len(df)} rows, {len(df.columns)} columns")
        return os.path.abspath(filepath)

    @staticmethod
    def _sanitise(df: pd.DataFrame) -> Any:
        """Convert a DataFrame to a JSON-safe structure, replacing NaN with None."""
        # Replace NaN sentinels with None (→ null in JSON)
        sanitised = df.where(pd.notna(df), None)
        return json.loads(sanitised.to_json(orient="records", date_format="iso"))


class StatsHelper:
    """Stateless statistical helpers used by imputation filters."""

    @staticmethod
    def safe_median(series: pd.Series) -> float:
        """Return the median of non-null, finite values, or 0.0 if empty."""
        valid = series.dropna()
        valid = valid[np.isfinite(valid)]
        return float(valid.median()) if len(valid) > 0 else 0.0

    @staticmethod
    def safe_mode(series: pd.Series) -> Any:
        """Return the mode of non-null values, or None if empty."""
        valid = series.dropna()
        if len(valid) == 0:
            return None
        mode_vals = valid.mode()
        return mode_vals.iloc[0] if len(mode_vals) > 0 else None

    @staticmethod
    def safe_mean(series: pd.Series) -> float:
        """Return the mean of non-null, finite values, or 0.0 if empty."""
        valid = series.dropna()
        valid = valid[np.isfinite(valid)]
        return float(valid.mean()) if len(valid) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: MODEL — Filter plugin interface and implementations
# ═══════════════════════════════════════════════════════════════════════════════

class FilterPlugin(ABC):
    """Abstract base class for data-filtering plugins.

    Each plugin is a callable that receives a DataFrame and returns a filtered
    (or completed) DataFrame.  Plugins are stateless by design — all
    configuration is passed at construction time.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the filter to *df* and return the result."""
        ...

    def __repr__(self) -> str:
        return f"{self.name}: {self.description}"


class DropFilter(FilterPlugin):
    """Drop rows that contain malformed or missing data.

    This is the simplest filtering strategy: any row with at least one NaN,
    infinite, or otherwise invalid value in the specified columns is removed
    from the dataset.  Under Rubin's (1976) taxonomy this corresponds to
    **complete-case analysis** — feasible when the missingness is MCAR and
    the proportion of incomplete cases is low (typically < 5 %).

    Design rationale for the RS3 experiment:
        Complete-case analysis is applicable as a *baseline* filter.  When
        missingness is MCAR, listwise deletion yields unbiased parameter
        estimates (Little & Rubin, 2019, Ch. 3).  For the RS3 generated data,
        MCAR-injected rows constitute ~1–3 % of trials and tests — well below
        the 5 % threshold where listwise deletion is generally considered
        acceptable (Schafer, 1999).  The filter provides a clean comparison
        point against more sophisticated completion methods.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        super().__init__(
            name="drop",
            description="Remove rows with missing or malformed values",
        )
        self.columns = columns  # None → check all columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        cols = self.columns if self.columns is not None else df.columns.tolist()

        # Detect malformed values: NaN, inf, or empty strings
        mask = pd.Series(True, index=df.index)
        for col in cols:
            if col not in df.columns:
                continue
            col_mask = df[col].notna()
            if pd.api.types.is_numeric_dtype(df[col]):
                # Also catch infinities in numeric columns
                col_mask = col_mask & np.isfinite(df[col].astype(float))
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                # Catch empty strings in string/object columns
                col_mask = col_mask & (df[col].astype(str).str.strip() != "")
            mask = mask & col_mask

        df_clean = df.loc[mask].copy()
        n_after = len(df_clean)
        print(f"  [drop]      {n_before - n_after} rows removed ({n_after} retained)")
        return df_clean


class HeuristicFilter(FilterPlugin):
    """Complete missing data using rule-of-thumb heuristics.

    Missing values in each column are replaced with a simple central-tendency
    statistic computed from the *non-missing* observations in that same column:

        - **Numeric columns** → median (robust to outliers and skew; see below).
        - **Categorical / object columns** → mode (most frequent category).

    Design rationale for the RS3 experiment:
        Median substitution is a fast, transparent, and deterministic
        heuristic.  For symmetric or mildly skewed distributions — such as the
        ex-Gaussian RTs and logistic test scores in the RS3 dataset — the
        median approximates the central location without the sensitivity to
        extreme values that the mean exhibits.  It preserves the ordinal
        structure of the data and introduces no variance inflation beyond the
        attenuation of the column variance (which is a known limitation of
        single-imputation methods; Little & Rubin, 2019, Ch. 4).  This filter
        is intended as an *intermediate* option between complete-case deletion
        and model-based multiple imputation — useful when the analyst needs a
        complete rectangular dataset quickly but is willing to accept the
        downward bias in variance estimates that all single-imputation methods
        carry.
    """

    def __init__(self, columns: Optional[List[str]] = None, stats: Optional[StatsHelper] = None):
        super().__init__(
            name="heuristic",
            description="Fill missing values with median (numeric) or mode (categorical)",
        )
        self.columns = columns
        self._stats = stats or StatsHelper()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.copy()
        cols = self.columns if self.columns is not None else df.columns.tolist()
        filled_count = 0

        for col in cols:
            if col not in df_filled.columns:
                continue
            n_missing = df_filled[col].isna().sum()
            if n_missing == 0:
                continue

            if pd.api.types.is_numeric_dtype(df_filled[col]):
                replacement = self._stats.safe_median(df_filled[col])
            else:
                replacement = self._stats.safe_mode(df_filled[col])

            df_filled[col] = df_filled[col].fillna(replacement)
            filled_count += n_missing

        n_before_missing = df.isna().sum().sum()
        print(f"  [heuristic] {filled_count} missing values filled "
              f"(median for numeric cols, mode for categorical)")
        return df_filled


class ImputationFilter(FilterPlugin):
    """Impute missing data by inferring from the complete subset.

    Unlike the heuristic filter (which uses a single column-wise statistic),
    this filter uses the **complete-case subset** — rows where *all* values
    are observed — as a reference distribution from which missing values are
    drawn via **hot-deck imputation** (random draw with replacement from
    observed values in the same column).

    Hot-deck imputation preserves the empirical distribution of each variable,
    including its shape, spread, and any floor/ceiling effects, without
    imposing parametric assumptions (Andridge & Little, 2010).  For the RS3
    experiment, where the ex-Gaussian RTs exhibit heavy right tails and the
    retention scores are bounded [0, 100], hot-deck avoids the artefacts that
    parametric draws (e.g., regression imputation) can introduce when the
    distributional assumptions are violated.

    Design rationale for the RS3 experiment:
        Hot-deck imputation is applicable when (a) the complete cases are a
        reasonably large and representative subset of the full dataset, and
        (b) the analyst prefers a non-parametric method that does not require
        specifying an imputation model.  In the RS3 data, MCAR-injected
        missingness ensures that complete cases are a random sample of the
        full data, satisfying condition (a).  The filter is registered as a
        plugin alongside the drop and heuristic filters, allowing the analyst
        to compare all three approaches on the same input.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        seed: int = 2026,
        stats: Optional[StatsHelper] = None,
    ):
        super().__init__(
            name="imputation",
            description="Hot-deck imputation — draw from observed values in complete cases",
        )
        self.columns = columns
        self._rng = np.random.default_rng(seed)
        self._stats = stats or StatsHelper()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_imputed = df.copy()
        cols = self.columns if self.columns is not None else df.columns.tolist()

        # Build the complete-case subset (rows with no missing values at all)
        complete_mask = df_imputed[cols].notna().all(axis=1)
        df_complete = df_imputed.loc[complete_mask, cols]

        if len(df_complete) == 0:
            print("  [imputation] WARNING — no complete cases; falling back to heuristic")
            fallback = HeuristicFilter(columns=cols, stats=self._stats)
            return fallback.apply(df_imputed)

        imputed_count = 0
        for col in cols:
            if col not in df_imputed.columns:
                continue
            missing_mask = df_imputed[col].isna()
            n_missing = missing_mask.sum()
            if n_missing == 0:
                continue

            observed = df_complete[col].dropna().values
            if len(observed) == 0:
                # No observed values for this column in complete cases —
                # fall back to median/mode for this column only
                if pd.api.types.is_numeric_dtype(df_imputed[col]):
                    fill_val = self._stats.safe_median(df_imputed[col])
                else:
                    fill_val = self._stats.safe_mode(df_imputed[col])
                df_imputed.loc[missing_mask, col] = fill_val
            else:
                draws = self._rng.choice(observed, size=n_missing, replace=True)
                df_imputed.loc[missing_mask, col] = draws

            imputed_count += n_missing

        print(f"  [imputation] {imputed_count} missing values imputed "
              f"(hot-deck from {len(df_complete)} complete cases)")
        return df_imputed


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: PROCESS — IPO function and plugin registry
# ═══════════════════════════════════════════════════════════════════════════════

class FilterPipeline:
    """Registry of filter plugins and IPO processing engine.

    The pipeline follows the IPO (Input → Process → Output) pattern:

        Input:   CSV file loaded via ``CSVBackend``.
        Process: One or more ``FilterPlugin`` instances applied in sequence.
        Output:  Cleaned DataFrame written to JSON via ``JSONBackend``.
    """

    def __init__(self):
        self._plugins: Dict[str, FilterPlugin] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register the three built-in filter plugins."""
        self.register(DropFilter())
        self.register(HeuristicFilter())
        self.register(ImputationFilter())

    def register(self, plugin: FilterPlugin) -> None:
        """Register a filter plugin by name."""
        self._plugins[plugin.name] = plugin
        print(f"  [register]  plugin '{plugin.name}' — {plugin.description}")

    def list_plugins(self) -> List[str]:
        """Return the names of all registered plugins."""
        return list(self._plugins.keys())

    def get_plugin(self, name: str) -> FilterPlugin:
        """Retrieve a plugin by name.

        Raises:
            KeyError: If the plugin is not registered.
        """
        if name not in self._plugins:
            available = ", ".join(self._plugins.keys())
            raise KeyError(f"Unknown filter '{name}'. Available: {available}")
        return self._plugins[name]

    # ── IPO Process Function ────────────────────────────────────────────────

    def process(
        self,
        input_csv: str,
        output_json: str,
        filters: List[str],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run the IPO pipeline: load → filter → save.

        Args:
            input_csv:   Path to the input CSV file.
            output_json: Path for the output JSON file.
            filters:     Ordered list of filter plugin names to apply.
            columns:     Optional subset of columns to filter on (None = all).

        Returns:
            The processed DataFrame.

        This is the **primary process function** and implements the IPO pattern
        required by exercises 0201–03.
        """
        # ── Input ──
        print(f"\n=== IPO Pipeline: {input_csv} → {output_json} ===")
        df = CSVBackend.load(input_csv)
        print(f"  [input]     columns: {list(df.columns)}")

        # ── Process ──
        for filter_name in filters:
            plugin = self.get_plugin(filter_name)
            print(f"  [process]   applying filter: {plugin.name}")
            # Pass column subset if the plugin supports it
            if columns is not None and hasattr(plugin, "columns"):
                plugin.columns = columns
            df = plugin.apply(df)
            print(f"  [process]   after '{plugin.name}': "
                  f"{len(df)} rows, {df.isna().sum().sum()} missing cells")

        # ── Output ──
        JSONBackend.save(df, output_json)
        print(f"=== Done: {len(df)} rows written to {output_json} ===\n")
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Data Filtering & Cleaning Pipeline — RS3 Experimental Data",
        epilog="Filter plugins: drop, heuristic, imputation.  "
               "Output: cleaned JSON with the selected filter(s) applied.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to the input CSV file (e.g., rs3_tests.csv).",
    )
    parser.add_argument(
        "-o", "--output",
        help="Path for the output JSON file (default: <input_stem>_clean.json).",
    )
    parser.add_argument(
        "-f", "--filter",
        dest="filters",
        action="append",
        choices=["drop", "heuristic", "imputation"],
        help="Filter plugin to apply.  May be specified multiple times to chain "
             "filters in order (e.g., -f drop -f imputation).  "
             "Available: drop, heuristic, imputation.",
    )
    parser.add_argument(
        "-c", "--columns",
        nargs="*",
        default=None,
        help="Subset of columns to filter on (default: all columns).",
    )
    parser.add_argument(
        "--list-filters",
        action="store_true",
        help="List all registered filter plugins and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for imputation reproducibility (default: 2026).",
    )

    args = parser.parse_args()

    # ── Build pipeline ──
    pipeline = FilterPipeline()

    if args.list_filters:
        print("\nRegistered filter plugins:")
        for name in pipeline.list_plugins():
            p = pipeline.get_plugin(name)
            print(f"  {name:14s} — {p.description}")
        return

    # ── Validate ──
    if args.input is None:
        parser.error("Input CSV file is required (unless using --list-filters).")

    if not args.filters:
        parser.error("At least one --filter is required (drop, heuristic, imputation).")

    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{stem}_clean.json"

    # ── Re-seed imputation plugin if present ──
    if "imputation" in args.filters:
        imp = pipeline.get_plugin("imputation")
        if isinstance(imp, ImputationFilter):
            imp._rng = np.random.default_rng(args.seed)

    # ── Run IPO pipeline ──
    result = pipeline.process(
        input_csv=args.input,
        output_json=args.output,
        filters=args.filters,
        columns=args.columns,
    )

    # ── Print summary ──
    print("Summary statistics after filtering:")
    print(result.describe(include="all").to_string())
    print(f"\nMissing cells remaining: {result.isna().sum().sum()}")


if __name__ == "__main__":
    main()
