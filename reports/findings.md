# ANER — completed experiment findings

These findings summarize the executed Colab notebooks. Evaluation and test results refer to different partitions and must be compared separately. All reported scores require an exact match of entity start, end and label.

## Experiment 1 — statistical baseline

The first experiment trained a spaCy NER model from scratch on **1,165 Spanish public procurement records containing 605 entity annotations**, revised according to guidelines version 2.1. The corpus was divided into 50% training, 25% evaluation and 25% test data, keeping identified duplicate families together. In the reported baseline run, epoch 22 achieved the highest evaluation F1: **55.4%**.

On the test partition, the model achieved **67.9% precision, 48.7% recall and 56.7% F1**. F1 fell to **48.5%** when normalized duplicate test texts were removed. Performance was concentrated in MISC, which accounted for 73 of the 74 correctly recognized test entities. ORG training examples lacked diversity: 52 of its 59 spans referred to Seguridad Social or ROLECE. PER had no examples, while EVENT had no held-out support.

The subsequent evaluation-error review identified missed entities, incorrect labels and overextended boundaries, particularly in dates, legal citations and schedules. These findings establish a baseline with limited recognition across categories. These findings motivated experiment 2, which combined the same statistical checkpoint with bounded patterns and name rules.

## Experiment 2 — ML plus rules

Adding regular expressions and name-based rules improved the system without retraining the ML model. The improvement appeared on both evaluation and test data.

| Scoring scope | ML baseline F1 | ML + rules F1 |
|---|---:|---:|
| Evaluation, all records | 55.43% | 70.88% |
| Evaluation, normalized unique texts | 54.03% | 71.68% |
| Test, all records | 56.70% | 66.91% |
| Test, normalized unique texts | 48.48% | 62.57% |

On evaluation, the rules recovered **27 missed entities** and removed **9 incorrect predictions**, without introducing new errors or losing correct predictions.

On the **292 test records**, the hybrid system improved F1 by **10.20 percentage points**. Correct entity detections increased from **74 to 93**, while incorrect predictions decreased from **35 to 33**. Precision increased from **67.89% to 73.81%**, and recall from **48.68% to 61.18%**.

The improvement remained when repeated test texts were counted only once. This suggests that repeated examples alone do not explain the gain. Rules alone achieved **21.97% test F1**, showing that their limited coverage works best alongside the learned model.

The clearest test improvements were in **ORG and DATE**. The hybrid recognized **13 of 29 organization entities**, compared with none for the baseline, and **4 of 5 dates**, compared with one. It also recognized the single email address and both schedules. These small counts do not establish reliable performance across those categories.

Important weaknesses remain: **LOC and LAW had no correct test predictions**, and MISC performance remained unchanged. ORG and MISC still accounted for most missed entities. PER and EVENT had no test examples, so their recognition could not be assessed.

These results show that targeted rules can complement an ML model, although the test improvement was smaller than the evaluation improvement. The rules were developed using evaluation errors, and some test examples had previously been inspected. The test comparison is therefore **exploratory**; fresh, untouched data would provide a stronger assessment of generalization.

## Recorded run provenance

- Baseline checkpoint run: `20261007T201530228446Z` (best epoch: 22).
- Hybrid experiment run: `20261007T202905741462Z`, reusing that baseline checkpoint without retraining.
- Approved annotation guidelines: version 2.1; annotated dataset: revision 3.
- Grouped split: 582 training, 291 evaluation and 292 test records.
- Secondary scoring: 166 normalized unique evaluation texts and 169 normalized unique test texts.
- Hybrid test outputs: `exploratory_test_comparison.json`, `test_summary.csv`, `test_normalized_unique_summary.csv` and `test_per_label.csv`.
