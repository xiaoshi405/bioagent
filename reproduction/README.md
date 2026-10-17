# Reproduction files

`plotting/results.py` draws both result figures from the archived plot CSVs:

```bash
python reproduction/plotting/results.py --output outputs/figures
```

For table reconstruction, token checks and plots together:

```bash
python scripts/reproduce_paper_results.py --output outputs/reproduced
```

`frozen_source.zip` preserves the 93 original source/document/input entries named by the external experiment freeze, including their original comments and documentation. It is a historical snapshot, not an installation package. Original SHA256 values are checked against this snapshot. Working Python files are compared with the originals after excluding comments and docstrings; executable syntax and other string literals must match.

Third-party core files and scientific inputs keep their original byte checks. The current file manifest separately covers the cleaned working tree. Newly run experiments require their own preparation and freeze.
