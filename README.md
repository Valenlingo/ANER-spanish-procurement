# ANER: Spanish public procurement NER

**Author:** Valentín Barbosa

A coursework project rebuilt as two documented experiments in named entity
recognition for Spanish public procurement texts. It covers data gathering,
annotation review, a statistical spaCy baseline and a hybrid system combining
the same trained checkpoint with regular expressions and name rules.

## Experiments

| Notebook | Purpose |
| --- | --- |
| [ANER_Colab.ipynb](ANER_Colab.ipynb) | Gather and explore the original public data, validate annotations, split the corpus, train and test the statistical baseline |
| [ANER_Experiment_2.ipynb](ANER_Experiment_2.ipynb) | Compare that checkpoint, rules alone and ML plus rules on evaluation and test data, without retraining |

### Recorded results

All scores below are **micro F1** using exact entity spans: character start,
end and label must all match.

| Partition / scoring scope | Records | ML baseline | ML + rules |
| --- | ---: | ---: | ---: |
| Evaluation, all records | 291 | 55.43% | 70.88% |
| Evaluation, normalized unique texts | 166 | 54.03% | 71.68% |
| Test, all records | 292 | 56.70% | **66.91%** |
| Test, normalized unique texts | 169 | 48.48% | **62.57%** |

On evaluation, rules recovered 27 missed entities and removed 9 false positives,
without introducing new false positives or losing correct predictions there.

On test, the hybrid improved F1 by **10.20 percentage points**. Precision increased
from 67.89% to 73.81%, and recall from 48.68% to 61.18%. Correct entity detections
increased from 74 to 93, while false positives decreased from 35 to 33.
Rules alone achieved 21.97% test F1, reflecting their limited coverage.

The hybrid recognized 13 of 29 test ORG entities and 4 of 5 dates, plus the
single email address and both schedules. MISC scores remained unchanged.
LOC and LAW had no correct test predictions. PER and EVENT had no test support.

The rules were developed using evaluation errors, and some test examples had
previously been inspected. **The test comparison is exploratory**, rather than
a fully independent estimate of generalization.

## Run in Google Colab

1. Open `ANER_Colab.ipynb` in Colab and run its installation and setup cells.
   If package installation requires a session restart, restart before importing
   the libraries, then resume at the imports.
2. Run the download and exploration stages first. Raw texts are gathered from
   the official public website; no raw Atom or text files from your PC are required.
3. In the AI-assisted annotation section, upload
   `ANER_AI_Assisted_Annotations.json` and `grouping_review.json` from
   `data/annotations/` when prompted. These are the previously reviewed labels,
   not annotations generated during this notebook run. The validation cell checks
   them against the newly downloaded source texts before training.
4. Continue through validation, splitting, training and baseline testing.
   `RUN_TRAINING` and `RUN_FINAL_TEST` control those stages and are enabled in
   the recorded run. Run `export_run()` to download `ANER_Run_Outputs.zip`,
   which contains the trained checkpoint and split files.
5. Open `ANER_Experiment_2.ipynb` in Colab and upload that run ZIP together
   with `ANER_Experiment_2.py`. Run the setup, evaluation and rule-review cells.
   This experiment reuses the baseline checkpoint and performs no retraining.
6. Run the added test-comparison cells. They score the baseline, rules alone
   and hybrid on the same test records, show per-label results and repeat the
   comparison on normalized unique texts. These cells score test data directly;
   the earlier evaluation-only cell can keep `RUN_HYBRID_TEST = False`.
7. Run the ZIP download cell after test scoring so the export includes the
   test reports and CSV summaries.

Use the notebooks for the current workflow. `ANER_Experiment_2.py` supplies the
hybrid rules and evaluation runner. `ANER_Colab.py` is an earlier cell-marked
baseline export; it does not include all later notebook setup edits.
`requirements.txt` lists the project dependencies, including spaCy 3.8.16.

## Data and annotations

Source: [Ministerio de Hacienda procurement catalog](https://www.hacienda.gob.es/es-ES/GobiernoAbierto/Datos%20Abiertos/Paginas/LicitacionesContratante.aspx).

The first notebook resolves the January 2025 catalog link and downloads four
original Atom documents directly from the publisher. It extracts
`cbc:Description`, reproduces the original newline handling and saves the
first **25,000 Unicode characters** of each document.

Download and exploration run before annotation loading. The later validation
stage verifies the sample checksums and source substrings against the approved
annotations. A mismatch stops label reuse. An explicit
`DOWNLOAD_SOURCE = False` uses the frozen `data/raw/` samples instead.

The revised corpus contains **1,165 records and 605 spans**, with a grouped split
of **582 training / 291 evaluation / 292 test records**. The nine labels are
ORG, PER, LOC, DATE, LAW, EMAIL, SCHEDULE, EVENT and MISC.
Guidelines version 2.1 is in
[docs/ANER_Annotation_Guidelines.docx](docs/ANER_Annotation_Guidelines.docx).

The annotations were AI-assisted and revised through human case review; they
are not independently double-annotated. Source lines, offsets, exclusions and
grouping decisions remain traceable. Identified duplicate families stay within
one partition. Normalized unique-text scoring retains the first original
passage per case/whitespace-normalized text without changing its gold offsets.

## Files and outputs

- `data/annotations/`: approved annotations and duplicate-family grouping review.
- `data/raw/`: frozen reference samples for explicit offline reproduction.
- `data/processed/`: readable split files and the split manifest. DocBin files are
  generated by the notebook and do not need to be uploaded to GitHub.
- `docs/`: annotation guidelines and validation notes.
- [reports/findings.md](reports/findings.md): completed findings for both experiments,
  including the hybrid test comparison.
- Other bundled `reports/` JSON files document initial preparation/runtime checks.
  Newly executed notebook runs write fresh reports and metadata.
- `checksums.json`: a file checksum manifest that needs refreshing when its
  listed files change.

The hybrid test cells save `exploratory_test_comparison.json`,
`test_summary.csv`, `test_normalized_unique_summary.csv` and
`test_per_label.csv` in the experiment output folder.

Downloaded Atom files, model weights and run ZIPs stay outside Git. The executed
notebook outputs contain the recorded results. To reproduce the comparison,
train the baseline and transfer its exported checkpoint into experiment 2.
Results can vary with the execution environment.

## Limitations

The corpus is small and contains repeated boilerplate. Grouping is heuristic
and does not guarantee full document-level independence. PER has no examples,
EVENT has one independent training group, and several other labels have very
little evaluation/test support. Perfect scores on sparse labels do not establish
broad recognition ability.

The hybrid helps with supported names and structured patterns, but has limited
coverage of unfamiliar organization names, places and legal references. Gains
remain when repeated texts are counted once, although both unique-text and
all-record scores come from the same small corpus.

Evaluation errors informed rule design, and the existing test set was partly
inspected before the hybrid comparison. Fresh, untouched data would provide a
stronger assessment of generalization. Full findings are in
[reports/findings.md](reports/findings.md).

