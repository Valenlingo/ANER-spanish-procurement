# ANER — completed experiment findings

The following findings are taken from the user-executed Colab notebooks. Evaluation and test results refer to different partitions and must be compared separately.

## Experiment 1 — statistical baseline

The first experiment trained a spaCy NER model from scratch on **1,165 Spanish public procurement records containing 605 entity annotations**, revised according to guidelines version 2.1. The corpus was divided into 50% training, 25% evaluation and 25% test data, keeping identified duplicate families together. In the reported baseline run, epoch 22 achieved the highest evaluation F1: **55.4%**.

On the test partition, the model achieved **67.9% precision, 48.7% recall and 56.7% F1**. F1 fell to **48.5%** when normalized duplicate test texts were removed. Performance was concentrated in MISC, which accounted for 73 of the 74 correctly recognized test entities. ORG training examples lacked diversity: 52 of its 59 spans referred to Seguridad Social or ROLECE. PER had no examples, while EVENT had no held-out support.

The subsequent evaluation-error review identified missed entities, incorrect labels and overextended boundaries, particularly in dates, legal citations and schedules. These findings establish a baseline with limited recognition across categories. The next experiment will combine the statistical model with carefully bounded patterns and dictionaries, measuring their effect separately against this baseline.

## Experiment 2 — ML plus rules

Adding regular expressions and name-based rules improved the system without retraining the ML model. Evaluation F1 increased from **55.43% to 70.88%**. The rules recovered **27 missed entities** and removed **9 incorrect predictions**, without introducing new errors or removing correct predictions on this evaluation set.

The improvement remained when repeated texts were counted only once: F1 increased from **54.03% to 71.68%**. This suggests that repetition alone does not explain the gain.

The rules helped recognize organization names, dates, legal references, email addresses and schedules, while correcting some entity boundaries. Most remaining errors concern ORG and MISC. Several labels achieved perfect evaluation scores, but they had very few examples, so these results should be interpreted cautiously.

This experiment shows that targeted rules can complement an ML model. However, the rules were developed using evaluation errors, so these scores describe progress on development data rather than independently verified performance on new texts.
