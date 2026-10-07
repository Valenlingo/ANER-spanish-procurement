# Notebook validation

The notebook was checked locally with Python 3.11 and spaCy 3.8.16.

- Notebook schema and every code cell passed validation/compilation.
- Preparation ran on all 1,165 records. Source substring checks, original sample
  checksums, strict token alignment and DocBin text/span round trips passed.
- The new grouped split contains 582 training, 291 evaluation and 292 test records.
  All detected duplicate-family links remain in one split.
- The training function completed a two-epoch runtime check on training/evaluation
  data, saved its selected checkpoint, and reloaded it successfully.
- Scoring, prediction, error reporting and the findings block were exercised using
  evaluation examples only, including undefined metrics when no gold entities
  were present. The actual held-out test partition was not scored.
- Catalog-link parsing and namespaced XML extraction were checked with fixtures.

These initial runtime checks are not portfolio performance results. The final
notebook now preserves the user's completed Colab training/testing outputs.

## Web acquisition revision — 7 October 2026

- The official catalog was fetched and its January 2025 archive link resolved.
- All four original Atom documents were downloaded from the public publisher.
  The complete monthly ZIP was not required.
- Extraction reproduces the original universal-newline handling before selecting
  the first 25,000 characters. All four sample SHA-256 checksums matched the
  approved annotation metadata exactly.
- The revised notebook's setup was exercised with only the two annotation JSON
  inputs and no raw text directory. Exploration and source validation used the
  downloaded documents. The already downloaded public Atom bytes were reused
  for this integration check to avoid transferring them again.
- All 1,165 retained record substrings and 605 strict entity boundaries passed
  validation. A deliberately mismatched expected checksum stopped annotation reuse.
- Notebook schema and Python syntax passed. The split and model code, saved model
  outputs and written findings were preserved. Models were not retrained during
  this acquisition revision.
- Web downloading is now the default. Frozen snapshots are retained only for the
  explicitly selected offline mode.
