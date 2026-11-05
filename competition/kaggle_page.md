# ConvoSense competition page

Text below the horizontal rules is what to paste into Kaggle. The host checklist at the end stays off the public page.

Files to upload on the **Data** tab (participants download these):

- `competition/kaggle_upload/train.csv` — 302 conversations, with labels
- `competition/kaggle_upload/validation.csv` — 103 conversations, with labels
- `competition/kaggle_upload/test.csv` — 101 conversations, inputs only
- `competition/kaggle_upload/sample_submission.csv` — submission template
- `competition/kaggle_upload/label_map.csv` — which tone each label number means

Kaggle rejected tone names as labels, so every file in `competition/kaggle_upload/` uses the numbers 1 to 5. The files in `competition/participant/` are the same data with tone names, for our own scripts. Do not upload those.

Answer key, uploaded only in Kaggle's solution slot:

- `competition/private/kaggle_upload/solution.csv`

Regenerate everything from the repository root with:

```bash
python src/competition/prepare_competition_files.py
python src/baselines/train_baselines.py
python src/competition/make_kaggle_numeric_files.py
```

Labels are the reliability-weighted vote in `outputs/ground_truth_weighted_vote.csv`. That vote assigns a tone to all 506 annotated conversations. The split is about 60% train, 20% validation, 20% test, by tone, and frozen in `competition/split_assignments.csv`: conversations already split keep their part, and only newly labeled ones are split. The test rows are split about evenly into the public and private leaderboards, stratified by tone.

---

## Title

ConvoSense: Customer Tone on AmazonHelp

## Subtitle

Classify the customer's tone in the last message of an Amazon support conversation.

## Overview

Predict the customer's tone in the last customer message of a Twitter support conversation with AmazonHelp. Earlier turns are context. The agent's tone is not part of the label.

Each row is one conversation. Turns are in time order, separated by ` | `. A turn is `[customer] - ...` or `[agent] - ...`. @mentions are removed, links are `[link]`, and images are `[image]`. `last_customer_message` is the last customer turn, or the last run of customer turns. That message is what the label describes.

Every conversation has one of these five tones. The `label` column holds the number:

1. Appreciative or positive — gratitude, satisfaction, or praise.
2. Neutral or informational — facts or answers, with no visible emotion, question, or demand.
3. Help-seeking or asking a question — a calm request for an answer or for help.
4. Frustrated or complaining — displeasure or criticism, without a demand for immediate action.
5. Escalatory or requesting immediate resolution — a demand, a threat, or a request to escalate.

Six annotators labeled the first batch, and a seventh annotator was added later: 682 labels on 506 conversations. 476 conversations have one annotator, and that label is kept. 30 conversations were labeled by six or seven annotators. On that shared set, mean pairwise Cohen's kappa is 0.437, Fleiss' kappa is 0.443, Krippendorff's alpha is 0.433, and pairs chose the same label 57.2% of the time. The label in these files is a reliability-weighted vote. Each annotator's weight is their mean pairwise Cohen's kappa with the others, and the tone with the greatest total weight is the label. That vote assigns one tone to every conversation.

The labeled data has 506 conversations. `train.csv` has 302, `validation.csv` has 103, and `test.csv` has 101. The split is made tone by tone, so each tone appears in the training file, the validation file, and the test set. Use the validation file to compare models. You can also train on train and validation together. `test.csv` has no labels.

The 101 test conversations are divided about evenly between a public leaderboard and a private leaderboard. The public score is visible during the competition. The private score is the final ranking, and it is revealed at the deadline. Every tone appears on both leaderboards.

### Files

| File | Columns | Rows |
| --- | --- | --- |
| `train.csv` | `id`, `conversation`, `last_customer_message`, `label` | 302 |
| `validation.csv` | `id`, `conversation`, `last_customer_message`, `label` | 103 |
| `test.csv` | `id`, `conversation`, `last_customer_message` | 101 |
| `sample_submission.csv` | `id`, `label` | 101 |

`id` is the tweet id of the first tweet in the thread. `label` is a number from 1 to 5, listed above and in `label_map.csv`. In `sample_submission.csv` the `label` column is filled with the most common tone in `train.csv`, 4 (Frustrated or complaining). Replace those values with your model's predictions.

## Evaluation

Submissions are scored with **macro-averaged F1**.

For each tone, precision is the share of predicted rows of that tone that are correct, and recall is the share of true rows of that tone that the submission found. The F1 score of a tone is the harmonic mean of its precision and recall:

F1 = 2 · precision · recall / (precision + recall)

Macro F1 is the unweighted average of the five tone F1 scores. Each tone counts the same.

This matches the label distribution from the annotation analysis. The five tones are imbalanced: Frustrated or complaining is 42.9% of the 506 labeled conversations, and Appreciative or positive and Escalatory or requesting immediate resolution are 11.3% each. A chi-square test against an equal share per tone gives p < 0.001. Accuracy would follow the largest tone. Macro F1 gives the four smaller tones the same weight as Frustrated or complaining.

The `label` values in the submission have to be whole numbers from 1 to 5.

## Submission file

`sample_submission.csv` shows the format. One row per id in `test.csv`, with a header:

```text
id,label
123,4
456,3
```

## Rules

This competition is for ARI 410/510 only. Predictions must come from a model. Labeling the test set by hand is not allowed.

You may train on `train.csv` and `validation.csv`, and you may use pretrained models and other publicly available text. Training on the test conversations, or using their labels, is outside these rules. So is using any tone label for a test conversation from outside this competition's files, such as an annotation from this course.

When you join, set your Kaggle team name so it includes your project team's name. Kaggle does not show hosts the email address on a submission.

## License

The tweet text comes from the Customer Support on Twitter dataset (Kaggle user thoughtvector), under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0). These processed conversations and tone labels are shared under the same license. Use is non-commercial. This competition adds a course limit: the files are for ARI 410/510 only.

---

## Host checklist

Keep this section off the public page, Discord, and the participant Data tab. The full record of choices and outputs is `docs/steps_2_to_6.md`.

Competition type: Machine Learning competition (Prediction). Visibility: Private, invitation link only. Set the end to the Phase 2b deadline on Canvas, then confirm the clock. Kaggle displays times in UTC.

Evaluation metric: macro F1. If the metric list has a macro F1 / mean F-score, choose it, set beta to 1, and point the target column at `label`. If the page asks for a metric file, paste `src/competition/macro_f1.py`, which accepts the numbers 1 to 5. Kaggle calls `score(solution, submission, row_id_column_name)`. The id column in every file is `id`. The answer column is `label`. The leaderboard column in `solution.csv` is `Usage`, with values `Public` and `Private`.

Upload map:

| Kaggle slot | Local file |
| --- | --- |
| Data tab | `competition/kaggle_upload/train.csv` |
| Data tab | `competition/kaggle_upload/validation.csv` |
| Data tab | `competition/kaggle_upload/test.csv` |
| Data tab | `competition/kaggle_upload/label_map.csv` |
| Data tab, and the sample-submission slot | `competition/kaggle_upload/sample_submission.csv` |
| Solution / answer key only | `competition/private/kaggle_upload/solution.csv` |

Leave `competition/private/` off the Data tab. That folder is gitignored, and so are the raw annotations and the per-conversation ground-truth CSVs in `outputs/`, because they contain the test labels too.

Before launching, submit `competition/private/kaggle_upload/perfect_submission.csv` and expect a score of 1 on both leaderboards. Then submit `competition/kaggle_upload/sample_submission.csv` and expect 0.118 public and 0.121 private (also in `competition/private/score_check.txt`). Showing three decimal places on the leaderboard is enough.

Per-class counts, including both leaderboard halves, are in `competition/private/split_summary.csv`. Every tone has at least 5 conversations on each leaderboard.

Majority vote leaves two ties. `233771` is in `validation.csv`. `1100995` is a private test row, labeled Escalatory by the weighted vote (1.318 against 1.268 for Frustrated), so that one row of `solution.csv` depends on the tie-break. The weighted labels are the ones in the files. The split is frozen in `competition/split_assignments.csv`, so rerunning the script never moves a conversation. Still do not change labels after launch, because the answer key would change.

Before launch, make the GitHub repository private. Commit `a58481e`, on the public `origin/phase2a`, contains `outputs/ground_truth_weighted_vote.csv` and the raw annotations, which hold the labels of 88 of the 101 test conversations.

After the perfect-score check, submit these two files and mark each as a Benchmark:

| Benchmark | File | Public macro F1 | Private macro F1 |
| --- | --- | ---: | ---: |
| Random dummy | `competition/kaggle_upload/benchmarks/dummy_baseline.csv` | 0.266 | 0.148 |
| Logistic regression | `competition/kaggle_upload/benchmarks/logistic_regression.csv` | 0.398 | 0.463 |

The random file is the dummy that won on validation (seed 3). `majority_baseline.csv` is there for comparison and is the same as `sample_submission.csv`. The GPT-5 baseline (macro F1 0.447) covers 50 of the 101 test conversations, so it is not uploaded as a Benchmark.
