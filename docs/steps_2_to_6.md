# Steps 2 through 6: agreement, ground truth, distribution, competition, baselines

Record of the choices, why each one was made, and the numbers the scripts produced. Regenerate the files from the repository root with `/opt/anaconda3/bin/python`. The project `.venv` is Python 3.15 and does not have the scientific packages installed. Anaconda 3.13 does.

Paste text for the public Kaggle page is in `competition/kaggle_page.md`. This file is the team record. The answer key in `competition/private/` stays off the participant Data tab, git, Drive, and Discord.

Every file with a label per conversation is gitignored, because each one covers the test conversations too: `annotation_submissions/`, `annotator_packages_overlap/shared_items.jsonl` (its heuristic `estimated_label`), `outputs/ground_truth_majority_vote.csv`, `outputs/ground_truth_weighted_vote.csv`, and `outputs/adjudication_disagreements.csv`. They exist only on the team's machines. Share them inside the team privately, never through the repository. The other files in `outputs/` hold counts and scores only and stay in git.

The task label is the primary Potato field `conversation_sense`. Optional `secondary_sense` is not a vote. The annotator id is the submission folder name. Annotators 01–06 use `annotator_01` through `annotator_06`. Annotator 07 submitted only Potato's export, `annotator_07/exports/jsonl/annotations.jsonl`, which the loader reads when there is no top-level file. For annotators 01–06 the loader reads the top-level `annotations.jsonl` and skips Potato's duplicate export.

Seven annotators submitted 682 labels on 506 conversations. 476 conversations have one annotator. 30 were shared: 26 by all seven annotators, and 4 without `annotator_05` (82 of 100 items from that annotator). Annotator 07 labeled all 30 shared conversations and 70 conversations of their own. On the shared set, 3 conversations are unanimous and 27 are not.

## Step 2. Agreement

**Choice.** All four metrics named in the assignment, each from a library, and only on conversations with at least two labels.

| Metric | Library | What was scored |
| --- | --- | --- |
| Cohen's kappa | `sklearn.metrics.cohen_kappa_score` | Each of the 21 annotator pairs, on the conversations both labeled |
| Percentage agreement | `sklearn.metrics.accuracy_score` | The same pairs. The score is the fraction of items where the two labels match |
| Fleiss' kappa | `statsmodels.stats.inter_rater.fleiss_kappa` | Conversations labeled by every annotator. That implementation requires the same rater count on every row |
| Krippendorff's alpha | `simpledorff`, nominal metric | All 30 shared conversations. Missing labels stay missing |

**Main metric.** Krippendorff's alpha (0.433) is the agreement figure the team reports. It works for more than two annotators and nominal labels, corrects for chance, and keeps the 4 shared conversations `annotator_05` skipped. Fleiss' kappa drops those conversations, Cohen's kappa scores only one pair at a time, and percentage agreement does not correct for chance. The other two chance-corrected scores fall within 0.011 of alpha.

**Why.** A conversation with one label cannot show agreement, so the 476 singly labeled rows are left out. Fleiss drops the 4 shared conversations `annotator_05` skipped. Alpha keeps them, which is the missing-data case that metric is for. Cohen's kappa and percentage agreement in the summary are the unweighted mean of the 21 pairs. Pairs have 26 or 30 conversations.

The script is `src/annotation_analysis/agreement.py`, run through `src/annotation_analysis/determine_ground_truth.py`.

| Metric | Score |
| --- | ---: |
| Cohen's kappa | 0.437 |
| Fleiss' kappa | 0.443 |
| Krippendorff's alpha | 0.433 |
| Percentage agreement | 0.572 |

Chance-corrected agreement is moderate (0.41–0.60 on the Landis–Koch scale), about the same as the six-annotator alpha of 0.427. Raw agreement is higher, as expected, because percentage agreement does not subtract chance.

The highest pair is annotator 05 and annotator 06: Cohen's kappa 0.755 on 26 conversations, percentage agreement 0.808. The lowest pair is annotator 01 and annotator 05: Cohen's kappa 0.264 on 26 conversations, percentage agreement 0.423. Annotator 07's kappa with each of the others ranges from 0.327 to 0.525.

`outputs/annotator_agreement_summary.csv` holds the four scores. `outputs/pairwise_annotator_agreement.csv` holds every pair. `outputs/annotator_agreement.png` is the summary bars plus the pairwise Cohen heatmap.

## Step 3. Ground truth

A step-by-step explanation, with worked examples, is in `docs/ground_truth_method.md`.

**Choice.** Two labelings, both written out. Neither breaks a tie by alphabetical order.

| Method | How | Library |
| --- | --- | --- |
| Majority vote | The unique modal label. A tie stays unlabeled for adjudication | `pandas.DataFrame.mode` |
| Reliability-weighted vote | The label with the greatest total annotator weight. A tie stays unlabeled | `DataFrame.groupby` sum of weights. Ties use `numpy.isclose` |

A conversation with one annotator keeps that label under both methods.

**Weights.** Each annotator's weight is their mean pairwise Cohen's kappa with the other six (`sklearn.metrics.cohen_kappa_score`). There is no separate experience score in the submissions. A non-positive mean would be clipped to zero. All seven weights are positive.

| Annotator | Reliability weight |
| --- | ---: |
| annotator_01 | 0.345 |
| annotator_02 | 0.457 |
| annotator_03 | 0.482 |
| annotator_04 | 0.391 |
| annotator_05 | 0.475 |
| annotator_06 | 0.466 |
| annotator_07 | 0.445 |

**Why both methods.** The assignment lists majority vote, weighted vote, and adjudication. Majority vote is the usual rule and leaves the two ties for a person to resolve. The weighted vote is what the competition files use, because a competition row needs one label and this vote assigns one to all 506 conversations.

| Resolution | Majority vote | Weighted vote |
| --- | ---: | ---: |
| Single annotator | 476 | 476 |
| Decided by the vote | 28 | 30 |
| Unresolved tie | 2 | 0 |

Annotator 07 broke both six-annotator ties: `1003675` is now Appreciative 4, Help-seeking 3, and `1257160` is now Frustrated 4, Escalatory 3. That changes the published validation label of `1257160` from Escalatory to Frustrated. Two new majority ties appeared:

| Conversation | Votes | Weighted totals | Weighted-vote label | Split |
| --- | --- | --- | --- | --- |
| 1100995 | Frustrated 3, Escalatory 3 (annotator 05 skipped it) | Escalatory 1.318, Frustrated 1.268 | Escalatory or requesting immediate resolution | Test, Private |
| 233771 | Help-seeking 3, Neutral 3, Frustrated 1 | Neutral 1.423, Help-seeking 1.294, Frustrated 0.345 | Neutral or informational | Validation |

`1100995` is in the test set, so its answer-key label depends on the weighted vote's tie-break. That is a single row of 51 on the private leaderboard.

`outputs/adjudication_disagreements.csv` lists all 27 shared conversations where the annotators did not all choose the same label. The `vote_summary` column gives the counts.

`outputs/ground_truth_majority_vote.csv`, `outputs/ground_truth_weighted_vote.csv`, and `outputs/annotator_reliability_weights.csv` are the tables. `outputs/ground_truth_label_distribution.png` compares the two label counts. They differ only on the two ties.

## Step 4. Label distribution

**Choice.** Count each tone with `Series.value_counts`, and test balance with `scipy.stats.chisquare` against an equal share of each class. The classes are imbalanced when p < 0.05. The chart uses the reliability-weighted labels so every conversation is included.

**Why.** Step 4 asks for the amount of data in each class and a report on balance. The chi-square test is the comparison to a balanced dataset. Majority vote, with the two ties dropped, is tested as well.

Weighted vote, 506 conversations. An equal share would be 101.2 conversations per tone.

| Tone | Conversations | Share |
| --- | ---: | ---: |
| Appreciative or positive | 57 | 11.3% |
| Neutral or informational | 81 | 16.0% |
| Help-seeking or asking a question | 94 | 18.6% |
| Frustrated or complaining | 217 | 42.9% |
| Escalatory or requesting immediate resolution | 57 | 11.3% |

| Method | Conversations | Chi-square | p-value | Balanced |
| --- | ---: | ---: | ---: | --- |
| Majority vote | 504 | 177.6 | < 0.001 | No |
| Reliability-weighted vote | 506 | 175.7 | < 0.001 | No |

Frustrated or complaining is about 3.8 times the smallest classes. The classes are imbalanced under both votes. That is why the competition metric is macro F1.

`outputs/label_distribution.csv`, `outputs/label_balance.csv`, and `outputs/label_distribution.png`. The dashed line on the chart is the equal share.

## Step 5. Competition files

### Ground-truth labels

**Choice.** Reliability-weighted vote, from `outputs/ground_truth_weighted_vote.csv`.

**Why.** A competition row needs one label. The weighted vote assigns a tone to all 506 annotated conversations. Majority vote leaves `1100995` and `233771` unlabeled. `1100995` is a test conversation, so one private answer-key row depends on the tie-break. The files use the weighted labels.

### Split

**Choice.** A frozen split, about 60% / 20% / 20% by tone, seed 42. `competition/split_assignments.csv` lists every conversation's part (ids only, no labels). The 436 six-annotator conversations keep the part they had in the files already pushed to GitHub. The 70 conversations only annotator 07 labeled were split tone by tone with a seeded shuffle and added to that file. The test set is then stratified into Public and Private with `sklearn.model_selection.train_test_split`.

**Why.** The assignment asks for a training set, a validation set, and a test set, with every class in the test set. The first seven-annotator run re-stratified everything, and 72 of its 102 test conversations had already been published, with the same labels, in the six-annotator `train.csv` and `validation.csv`. Freezing keeps every published training or validation conversation out of test. Rerunning the script with new labels never moves a conversation that is already in the file. Annotator 07's 70 new conversations have only 2 Escalatory labels, too few for `train_test_split` to stratify, so they are shuffled per tone instead. The smallest test tone has 11 conversations, 5 and 6 per leaderboard half.

The previous six-annotator test set is all still in test (88 conversations), plus 13 new ones. Those 88 labels, though, were in commit `a58481e` (`outputs/ground_truth_weighted_vote.csv` and `annotation_submissions/`) before those files were untracked, and that commit is on the public `origin/phase2a`. The test labels stay exposed until the repository is private and that history is removed.

The script is `src/competition/prepare_competition_files.py`.

| Tone | Train | Validation | Test | Public | Private |
| --- | ---: | ---: | ---: | ---: | ---: |
| Appreciative or positive | 34 | 12 | 11 | 6 | 5 |
| Neutral or informational | 48 | 17 | 16 | 8 | 8 |
| Help-seeking or asking a question | 56 | 19 | 19 | 9 | 10 |
| Frustrated or complaining | 130 | 44 | 43 | 21 | 22 |
| Escalatory or requesting immediate resolution | 34 | 11 | 12 | 6 | 6 |
| **Total** | **302** | **103** | **101** | **50** | **51** |

### Files

**Choice.** Participants get four files. The answer key is a fifth file, stored only under `competition/private/`.

| Who downloads it | File | Columns |
| --- | --- | --- |
| Participants | `competition/participant/train.csv` | `id`, `conversation`, `last_customer_message`, `label` |
| Participants | `competition/participant/validation.csv` | same as train |
| Participants | `competition/participant/test.csv` | `id`, `conversation`, `last_customer_message` |
| Participants | `competition/participant/sample_submission.csv` | `id`, `label` |
| Host only | `competition/private/solution.csv` | `id`, `label`, `Usage` |

**Why.** Kaggle asks for `train.csv`, `test.csv`, `solution.csv`, and `sample_submission.csv`, with the id column first. Validation is a separate file so a model can be compared without touching the test set. `Usage` is `Public` or `Private`. `sample_submission.csv` fills every row with the most common training label, Frustrated or complaining.

`last_customer_message` is the span the label describes. `conversation` is the earlier context. Text is the plain-text AmazonHelp JSONL, with turns separated by ` | `.

`competition/private/` is gitignored. It also holds `perfect_submission.csv` (the answer key without `Usage`, for the host's perfect-score check), `split_summary.csv`, and `split_ids.csv`.

### Evaluation metric

**Choice.** Macro-averaged F1. The scorer is `src/competition/macro_f1.py`, which calls `sklearn.metrics.f1_score` with `average="macro"`.

**Why.** Step 4 found the tones are imbalanced. Frustrated or complaining is 42.9% of the 506 labeled conversations, and Appreciative or positive and Escalatory or requesting immediate resolution are 11.3% each. A chi-square test against an equal share per tone gives p < 0.001. Macro F1 is the unweighted mean of the five per-class F1 scores, so each tone counts the same. Accuracy would follow the largest tone.

A label in a submission has to match one of the five tone strings exactly. On Kaggle, map the id column to `id` and the answer column to `label`. If the metric list has macro F1, select it with beta 1. If the page asks for a metric file, paste `src/competition/macro_f1.py`.

**Check.** Submitting `perfect_submission.csv` scores 1 on the public and private leaderboards. Submitting `sample_submission.csv` scores 0.118 public and 0.121 private (`competition/private/score_check.txt` records 0.1183 and 0.1205).

## Step 6. Baselines

A step-by-step explanation of the dummy and trained baselines is in `docs/baselines_method.md`.

All scores below are macro F1 from the competition scorer. Full-test numbers are in `baselines/baseline_scores.csv`. Per-class F1 is in `baselines/per_class_f1.csv`.

### Dummy baseline

**Choice.** Uniform random labels, `sklearn.dummy.DummyClassifier(strategy="uniform")`, seed 3. Upload `competition/submissions/dummy_baseline.csv` and mark it as a Benchmark.

**Why.** The assignment says to keep whichever of the random baseline and the majority baseline scores higher. Compared on the validation set:

| Dummy | How it was scored | Validation macro F1 |
| --- | --- | ---: |
| Majority | Always predicts Frustrated or complaining | 0.120 |
| Random | Mean of seeds 0–19 | 0.196 |

The majority baseline has an F1 near 0.60 on Frustrated or complaining and 0 on the other four tones, so the macro average stays near 0.12. A uniform draw scores a little on every tone, so its macro F1 is higher.

One random draw moves around on 103 validation rows. Seed 3 is the draw whose validation score (0.191) is closest to the 20-seed mean (0.196). The seed was chosen from validation scores only. Its test score is 0.208, next to the 20-seed test mean of 0.193. On the two leaderboard halves it is 0.266 public and 0.148 private: a uniform draw on 50 rows is noisy. `majority_baseline.csv` is the same file as `sample_submission.csv`.

### Trained baseline

**Choice.** Word and bigram TF-IDF on `last_customer_message` and on `conversation`, then `sklearn.linear_model.LogisticRegression`. Upload `competition/submissions/logistic_regression.csv` and mark it as a Benchmark.

**Why.** The assignment asks for one simple trained model. Logistic regression on TF-IDF features runs locally on the 302 training conversations. The grid below was fit on train and scored on validation. The test labels were not used to pick the settings.

| C | class_weight | Validation macro F1 |
| ---: | --- | ---: |
| 0.25 | balanced | 0.330 |
| 4.0 | balanced | 0.328 |
| 1.0 | balanced | 0.315 |
| 4.0 | none | 0.217 |
| 1.0 | none | 0.150 |
| 0.25 | none | 0.120 |

`class_weight="balanced"` raises the rare tones, which is what macro F1 measures. `C=0.25` won, narrowly ahead of `C=4.0`. The validation score 0.330 is that model fit on train only. The submission file is the same settings refit on train and validation together, so the leaderboard model sees every labeled conversation the participants receive.

### Full-test scores

| Model | Validation | Public | Private | Full test (101) |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.120 | 0.118 | 0.121 | 0.119 |
| Random, seed 3 | 0.191 | 0.266 | 0.148 | 0.208 |
| Random, mean of 20 seeds | 0.196 |  |  | 0.193 |
| Logistic regression | 0.330 | 0.398 | 0.463 | 0.428 |

On Kaggle, the random benchmark should show about **0.266** public and **0.148** private. The logistic regression benchmark should show about **0.398** public and **0.463** private. The test scores are well above the validation score of 0.330. Validation and test are small (103 and 101 rows), and the submitted model is refit on train plus validation, so a gap this size is noise plus the extra training data, not a sign that test was used.

Per-class F1 for logistic regression on the full test set:

| Tone | Test F1 |
| --- | ---: |
| Appreciative or positive | 0.696 |
| Neutral or informational | 0.167 |
| Help-seeking or asking a question | 0.313 |
| Frustrated or complaining | 0.553 |
| Escalatory or requesting immediate resolution | 0.414 |

Neutral or informational is the weak class: test F1 0.167, with 0 on the public half and 0.308 on the private half.

### LLM baseline

**Choice.** GPT-5 in the ChatGPT web interface, zero-shot, on 50 test conversations (`baselines/llm/items.jsonl`, seed 42, drawn from the six-annotator test set). The frozen split keeps all 50 in the current test set, so the same reply was rescored against the seven-annotator labels: macro F1 **0.441**, unchanged because none of the 50 labels changed. The reply is `baselines/llm/raw_responses.jsonl`, converted to `baselines/llm/predictions.csv`. Scores are in `baselines/llm_scores.csv`.

On the same 50 conversations (`baselines/llm_sample_comparison.csv`):

| Model | Macro F1 on the 50 |
| --- | ---: |
| Majority | 0.118 |
| Random, seed 3 | 0.260 |
| Logistic regression | 0.303 |
| GPT-5 zero-shot | 0.441 |

**Why it stays off the leaderboard.** The assignment says an LLM baseline stays off the leaderboard unless every test item is included. 50 of 101 is not.

## Commands

```bash
MPLCONFIGDIR=/tmp/mplconfig /opt/anaconda3/bin/python src/annotation_analysis/determine_ground_truth.py
/opt/anaconda3/bin/python src/competition/prepare_competition_files.py
/opt/anaconda3/bin/python src/baselines/train_baselines.py
/opt/anaconda3/bin/python src/baselines/llm_baseline.py score baselines/llm/predictions.csv
```

The first command rewrites `outputs/`: agreement, both ground truths, the disagreement list, and the label-distribution chart. The second keeps the frozen split in `competition/split_assignments.csv`, splits any newly labeled conversations, adds them to that file, and rewrites the competition files. The third refits the dummy and logistic regression baselines and rewrites `baselines/baseline_scores.csv`. Do not rerun `llm_baseline.py write`: it would draw a new 50 from the current test file, and the GPT-5 reply would no longer match. The last command rescores the saved reply against the current answer key.
