import argparse
import json
import math
from typing import Any, Dict, List, Optional

import pandas as pd


COLUMN_ALIASES = {
    "imdb_id": ["imdb_id", "tconst"],
    "name": ["Name", "primaryTitle", "title"],
    "genre": ["Genre", "genres"],
    "rating": ["Rating", "averageRating"],
    "popularity": ["Popularity", "numVotes"],
    "year": ["Year", "startYear"],
    "description": ["Description", "originalTitle"],
}


def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset from CSV."""
    return pd.read_csv(path)


def resolve_column(df: pd.DataFrame, logical_name: str) -> Optional[str]:
    """Resolve logical column names against known aliases in a dataset."""
    for candidate in COLUMN_ALIASES.get(logical_name, []):
        if candidate in df.columns:
            return candidate
    return None


def validate_required_columns(df: pd.DataFrame, required: Optional[List[str]] = None) -> Dict[str, Any]:
    """Check whether required logical columns exist using schema aliases."""
    required = required or ["imdb_id", "name", "genre", "rating", "popularity", "year"]
    missing = []
    resolved = {}

    for logical_col in required:
        physical_col = resolve_column(df, logical_col)
        resolved[logical_col] = physical_col
        if physical_col is None:
            missing.append(logical_col)

    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "resolved": resolved,
    }


def missing_value_report(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Report missing-value counts and percentages for selected logical columns."""
    columns = columns or ["imdb_id", "name", "genre", "rating", "popularity", "year"]
    row_count = len(df)
    per_column: Dict[str, Dict[str, Any]] = {}

    for logical_col in columns:
        physical_col = resolve_column(df, logical_col)
        if physical_col is None:
            per_column[logical_col] = {
                "column": None,
                "missing_count": row_count,
                "missing_pct": 100.0,
            }
            continue

        missing_count = int(df[physical_col].isna().sum())
        missing_pct = (missing_count / row_count * 100.0) if row_count else 0.0
        per_column[logical_col] = {
            "column": physical_col,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        }

    total_missing = sum(item["missing_count"] for item in per_column.values())
    return {
        "rows": row_count,
        "total_missing": total_missing,
        "per_column": per_column,
    }


def duplicate_record_report(df: pd.DataFrame, key_logical_col: str = "imdb_id") -> Dict[str, Any]:
    """Detect duplicate records based on a logical key column."""
    key_col = resolve_column(df, key_logical_col)
    if key_col is None:
        return {
            "ok": False,
            "key_column": None,
            "duplicate_count": 0,
            "duplicate_values": [],
        }

    duplicate_mask = df.duplicated(subset=[key_col], keep=False)
    duplicates = df.loc[duplicate_mask, key_col].dropna().astype(str).tolist()
    duplicate_values = sorted(set(duplicates))

    return {
        "ok": len(duplicate_values) == 0,
        "key_column": key_col,
        "duplicate_count": len(duplicate_values),
        "duplicate_values": duplicate_values,
    }


def extract_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract deterministic, model-ready features from movie metadata."""
    out = pd.DataFrame(index=df.index)

    id_col = resolve_column(df, "imdb_id")
    genre_col = resolve_column(df, "genre")
    rating_col = resolve_column(df, "rating")
    popularity_col = resolve_column(df, "popularity")
    year_col = resolve_column(df, "year")

    out["movie_id"] = df[id_col].astype(str) if id_col else ""

    if genre_col:
        normalized_genres = (
            df[genre_col]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(" ", "", regex=False)
        )
        out["genre_tokens"] = normalized_genres
        out["genre_count"] = normalized_genres.apply(
            lambda x: 0 if not x or x == "\\n" else len([g for g in x.split(",") if g])
        )
    else:
        out["genre_tokens"] = ""
        out["genre_count"] = 0

    if rating_col:
        ratings = pd.to_numeric(df[rating_col], errors="coerce").fillna(0.0)
        out["rating_norm_0_10"] = (ratings.clip(lower=0.0, upper=10.0) / 10.0).round(6)
    else:
        out["rating_norm_0_10"] = 0.0

    if popularity_col:
        popularity = pd.to_numeric(df[popularity_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        out["popularity_log1p"] = popularity.map(lambda x: float(math.log1p(x))).round(6)
    else:
        out["popularity_log1p"] = 0.0

    if year_col:
        years = pd.to_numeric(df[year_col], errors="coerce")
        out["year_decade"] = years.map(
            lambda y: int(y // 10 * 10) if pd.notna(y) and y > 0 else -1
        )
    else:
        out["year_decade"] = -1

    # Enforce deterministic row and column ordering.
    out = out.sort_values(by=["movie_id"]).reset_index(drop=True)
    out = out[[
        "movie_id",
        "genre_tokens",
        "genre_count",
        "rating_norm_0_10",
        "popularity_log1p",
        "year_decade",
    ]]

    return out


def feature_fingerprint(features: pd.DataFrame) -> str:
    """Generate a deterministic fingerprint for extracted features."""
    row_hashes = pd.util.hash_pandas_object(features, index=True)
    return str(int(row_hashes.sum()))


def verify_reproducible_feature_extraction(df: pd.DataFrame) -> Dict[str, Any]:
    """Verify that feature extraction produces identical output across repeated runs."""
    features_a = extract_structured_features(df)
    features_b = extract_structured_features(df)
    fp_a = feature_fingerprint(features_a)
    fp_b = feature_fingerprint(features_b)

    return {
        "ok": fp_a == fp_b,
        "fingerprint_first": fp_a,
        "fingerprint_second": fp_b,
        "row_count": len(features_a),
        "feature_columns": list(features_a.columns),
    }


def audit_preprocessing_integrity(path: str) -> Dict[str, Any]:
    """Run a full preprocessing integrity audit for one dataset."""
    df = load_dataset(path)

    required_result = validate_required_columns(df)
    missing_result = missing_value_report(df)
    duplicate_result = duplicate_record_report(df)
    reproducibility_result = verify_reproducible_feature_extraction(df)

    passed = all([
        required_result["ok"],
        duplicate_result["ok"],
        reproducibility_result["ok"],
    ])

    return {
        "dataset": path,
        "rows": len(df),
        "columns": len(df.columns),
        "passed": passed,
        "required_columns": required_result,
        "missing_values": missing_result,
        "duplicates": duplicate_result,
        "feature_reproducibility": reproducibility_result,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit preprocessing integrity for one or more movie datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["imdb_filtered.csv", "imdb_full_metadata.csv"],
        help="CSV datasets to audit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print output as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = [audit_preprocessing_integrity(path) for path in args.datasets]

    if args.json:
        print(json.dumps(results, indent=2))
        return

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} | {result['dataset']}")
        print(f"  rows={result['rows']} columns={result['columns']}")
        print(f"  missing-total={result['missing_values']['total_missing']}")
        print(f"  duplicates={result['duplicates']['duplicate_count']}")
        print(f"  feature-fingerprint={result['feature_reproducibility']['fingerprint_first']}")


if __name__ == "__main__":
    main()
