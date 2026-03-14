import unittest
import tempfile
from pathlib import Path

import pandas as pd

from preprocessing_validation import (
    validate_required_columns,
    missing_value_report,
    duplicate_record_report,
    extract_structured_features,
    verify_reproducible_feature_extraction,
    audit_preprocessing_integrity,
)


class TestPreprocessingValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.filtered_like_df = pd.DataFrame(
            [
                {
                    "imdb_id": "tt0001",
                    "Name": "Movie A",
                    "Type": "movie",
                    "Genre": "Action,Sci-Fi",
                    "Rating": 8.5,
                    "Popularity": 150000,
                    "Year": 1999,
                    "Description": "A",
                },
                {
                    "imdb_id": "tt0002",
                    "Name": "Movie B",
                    "Type": "movie",
                    "Genre": "Drama",
                    "Rating": 7.2,
                    "Popularity": 90000,
                    "Year": 2001,
                    "Description": "B",
                },
            ]
        )

        self.full_metadata_like_df = pd.DataFrame(
            [
                {
                    "tconst": "tt1001",
                    "primaryTitle": "Movie X",
                    "genres": "Comedy,Drama",
                    "averageRating": 7.8,
                    "numVotes": 51000,
                    "startYear": 2010,
                },
                {
                    "tconst": "tt1002",
                    "primaryTitle": "Movie Y",
                    "genres": "Action",
                    "averageRating": 8.1,
                    "numVotes": 60000,
                    "startYear": 2012,
                },
            ]
        )

    def test_required_columns_supported_for_filtered_schema(self) -> None:
        result = validate_required_columns(self.filtered_like_df)
        self.assertTrue(result["ok"])
        self.assertEqual(result["missing"], [])

    def test_required_columns_supported_for_full_metadata_schema(self) -> None:
        result = validate_required_columns(self.full_metadata_like_df)
        self.assertTrue(result["ok"])
        self.assertEqual(result["missing"], [])

    def test_missing_value_report_detects_nulls(self) -> None:
        df = self.filtered_like_df.copy()
        df.loc[1, "Genre"] = None
        report = missing_value_report(df)

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["per_column"]["genre"]["missing_count"], 1)

    def test_duplicate_record_report_detects_duplicate_ids(self) -> None:
        df = pd.concat([self.filtered_like_df, self.filtered_like_df.iloc[[0]]], ignore_index=True)
        report = duplicate_record_report(df)

        self.assertFalse(report["ok"])
        self.assertEqual(report["duplicate_count"], 1)
        self.assertIn("tt0001", report["duplicate_values"])

    def test_feature_extraction_is_deterministic(self) -> None:
        report = verify_reproducible_feature_extraction(self.filtered_like_df)

        self.assertTrue(report["ok"])
        self.assertEqual(report["fingerprint_first"], report["fingerprint_second"])
        self.assertEqual(report["row_count"], 2)

    def test_feature_extraction_schema_aliases(self) -> None:
        features = extract_structured_features(self.full_metadata_like_df)

        self.assertEqual(len(features), 2)
        self.assertIn("movie_id", features.columns)
        self.assertIn("genre_tokens", features.columns)
        self.assertIn("rating_norm_0_10", features.columns)

    def test_full_audit_passes_clean_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clean_dataset.csv"
            self.filtered_like_df.to_csv(path, index=False)
            result = audit_preprocessing_integrity(str(path))

            self.assertTrue(result["required_columns"]["ok"])
            self.assertTrue(result["duplicates"]["ok"])
            self.assertTrue(result["feature_reproducibility"]["ok"])


if __name__ == "__main__":
    unittest.main()
