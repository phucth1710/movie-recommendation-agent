# movie-recommendation-agent
A stateless AI-powered movie recommendation and ranking system with natural language query support, built using Python, SQLite, and structured metadata.

## Preprocessing Integrity Validation

Use the validation script to audit dataset consistency for missing values, duplicate IDs, and deterministic feature extraction:

python preprocessing_validation.py

You can target specific datasets and print machine-readable output:

python preprocessing_validation.py imdb_filtered.csv imdb_full_metadata.csv --json

## Automated Tests

Run preprocessing validation tests:

python -m unittest discover -s tests -p "test_*.py"
