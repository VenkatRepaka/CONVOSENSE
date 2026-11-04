# ConvoSense competition page

Text below the horizontal rules is what to paste into Kaggle. The host checklist at the end stays off the public page.

Files to upload on the **Data** tab (participants download these):

- `competition/participant/train.csv` — 261 conversations, with labels
- `competition/participant/validation.csv` — 87 conversations, with labels
- `competition/participant/test.csv` — 88 conversations, inputs only
- `competition/participant/sample_submission.csv` — submission template

Answer key, uploaded only in Kaggle's solution slot:

- `competition/private/solution.csv`

Regenerate everything from the repository root with:

```bash
python src/competition/prepare_competition_files.py
```

Labels are the reliability-weighted vote in `outputs/ground_truth_weighted_vote.csv`. That vote assigns a tone to all 436 annotated conversations. The split is stratified by tone, seed 42: about 60% train, 20% validation, 20% test. The test rows are split evenly into the public and private leaderboards.

---

## Title

ConvoSense: Customer Tone on AmazonHelp

## Subtitle

Classify the customer's tone in the last message of an Amazon support conversation.

## Overview

Predict the customer's tone in the last customer message of a Twitter support conversation with AmazonHelp. Earlier turns are context. The agent's tone is not part of the label.

Each row is one conversation. Turns are in time order, separated by ` | `. A turn is `[customer] - ...` or `[agent] - ...`. @mentions are removed, links are `[link]`, and images are `[image]`. `last_customer_message` is the last customer turn, or the last run of customer turns. That message is what the label describes.

Every conversation has one of these five tones:

1. Appreciative or positive — gratitude, satisfaction, or praise.
2. Neutral or informational — facts or answers, with no visible emotion, question, or demand.
3. Help-seeking or asking a question — a calm request for an answer or for help.
4. Frustrated or complaining — displeasure or criticism, without a demand for immediate action.
5. Escalatory or requesting immediate resolution — a demand, a threat, or a request to escalate.

Six annotators labeled these conversations: 582 labels on 436 conversations. 406 conversations have one annotator, and that label is kept. 30 conversations were labeled by five or six annotators. On that shared set, mean pairwise Cohen's kappa is 0.434, Fleiss' kappa is 0.447, Krippendorff's alpha is 0.427, and pairs chose the same label 56.6% of the time. The label in these files is a reliability-weighted vote. Each annotator's weight is their mean pairwise Cohen's kappa with the others, and the tone with the greatest total weight is the label. That vote assigns one tone to every conversation.

The labeled data has 436 conversations. `train.csv` has 261, `validation.csv` has 87, and `test.csv` has 88. The split is stratified by tone, so each tone appears in the training file, the validation file, and the test set. Use the validation file to compare models. You can also train on train and validation together. `test.csv` has no labels.

The 88 test conversations are divided evenly between a public leaderboard and a private leaderboard. The public score is visible during the competition. The private score is the final ranking, and it is revealed at the deadline. Every tone appears on both leaderboards.

### Files

| File | Columns | Rows |
| --- | --- | --- |
| `train.csv` | `id`, `conversation`, `last_customer_message`, `label` | 261 |
| `validation.csv` | `id`, `conversation`, `last_customer_message`, `label` | 87 |
| `test.csv` | `id`, `conversation`, `last_customer_message` | 88 |
| `sample_submission.csv` | `id`, `label` | 88 |

`id` is the tweet id of the first tweet in the thread. In `sample_submission.csv` the `label` column is filled with the most common tone in `train.csv`, `Frustrated or complaining`. Replace those values with your model's predictions.

## Evaluation

Submissions are scored with **macro-averaged F1**.

For each tone, precision is the share of predicted rows of that tone that are correct, and recall is the share of true rows of that tone that the submission found. The F1 score of a tone is the harmonic mean of its precision and recall:

F1 = 2 · precision · recall / (precision + recall)

Macro F1 is the unweighted average of the five tone F1 scores. Each tone counts the same.

This matches the label distribution from the annotation analysis. The five tones are imbalanced: Frustrated or complaining is 43.8% of the 436 labeled conversations, and Appreciative or positive is 11.5%. A chi-square test against an equal share per tone gives p < 0.001. Accuracy would follow the largest tone. Macro F1 gives the four smaller tones the same weight as Frustrated or complaining.

The `label` values in the submission have to match the five tone names exactly, including capitalization.

## Submission file

`sample_submission.csv` shows the format. One row per id in `test.csv`, with a header:

```text
id,label
123,Frustrated or complaining
456,Help-seeking or asking a question
```

## Rules

This competition is for ARI 410/510 only. Predictions must come from a model. Labeling the test set by hand is not allowed.

You may train on `train.csv` and `validation.csv`, and you may use pretrained models and other publicly available text. Training on the test conversations, or using their labels, is outside these rules.

When you join, set your Kaggle team name so it includes your project team's name. Kaggle does not show hosts the email address on a submission.

## License

The tweet text comes from the Customer Support on Twitter dataset (Kaggle user thoughtvector), under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0). These processed conversations and tone labels are shared under the same license. Use is non-commercial. This competition adds a course limit: the files are for ARI 410/510 only.

---

## Host checklist

Keep this section off the public page, Discord, and the participant Data tab. The full record of choices and outputs is `docs/steps_2_to_6.md`.

Competition type: Machine Learning competition (Prediction). Visibility: Private, invitation link only. Set the end to the Phase 2b deadline on Canvas, then confirm the clock. Kaggle displays times in UTC.

Evaluation metric: macro F1. If the metric list has a macro F1 / mean F-score, choose it, set beta to 1, and point the target column at `label`. If the page asks for a metric file, paste `src/competition/macro_f1.py`. Kaggle calls `score(solution, submission, row_id_column_name)`. The id column in every file is `id`. The answer column is `label`. The leaderboard column in `solution.csv` is `Usage`, with values `Public` and `Private`.

Upload map:

| Kaggle slot | Local file |
| --- | --- |
| Data tab | `competition/participant/train.csv` |
| Data tab | `competition/participant/validation.csv` |
| Data tab | `competition/participant/test.csv` |
| Data tab, and the sample-submission slot | `competition/participant/sample_submission.csv` |
| Solution / answer key only | `competition/private/solution.csv` |

Leave `competition/private/` off the Data tab. That folder is gitignored.

Before launching, submit `competition/private/perfect_submission.csv` and expect a score of 1 on both leaderboards. Then submit `sample_submission.csv`. The scores to expect are written in `competition/private/score_check.txt` when the script runs. Showing three decimal places on the leaderboard is enough.

Per-class counts, including both leaderboard halves, are in `competition/private/split_summary.csv`. The smallest tone has 5 conversations on each leaderboard.

The two conversations majority vote leaves tied, `1003675` and `1257160`, both landed in `validation.csv`. Train and test labels are the same under majority vote and the weighted vote. The weighted labels are the ones in the files.

After the perfect-score check, submit these two files and mark each as a Benchmark:

| Benchmark | File | Public macro F1 | Private macro F1 |
| --- | --- | ---: | ---: |
| Random dummy | `competition/submissions/dummy_baseline.csv` | 0.181 | 0.139 |
| Logistic regression | `competition/submissions/logistic_regression.csv` | 0.339 | 0.377 |

The random file is the dummy that won on validation. `majority_baseline.csv` is there for comparison and is the same as `sample_submission.csv`. The LLM baseline covers 50 test items, so it stays off the leaderboard. Its score is in `baselines/llm_scores.csv`.
