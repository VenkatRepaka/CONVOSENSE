# Steps 2 through 6: agreement, ground truth, distribution, competition, baselines

Record of the choices, why each one was made, and the numbers the scripts produced. Regenerate the files from the repository root with `/opt/anaconda3/bin/python`. The project `.venv` is Python 3.15 and does not have the scientific packages installed. Anaconda 3.13 does.

Paste text for the public Kaggle page is in `competition/kaggle_page.md`. This file is the team record. The answer key in `competition/private/` stays off the participant Data tab, git, Drive, and Discord.

Every file with a label per conversation is gitignored, because each one covers the test conversations too: `annotation_submissions/`, `annotator_packages_overlap/shared_items.jsonl` (its heuristic `estimated_label`), `outputs/ground_truth_majority_vote.csv`, `outputs/ground_truth_weighted_vote.csv`, and `outputs/adjudication_disagreements.csv`. They exist only on the team's machines. Share them inside the team privately, never through the repository. The other files in `outputs/` hold counts and scores only and stay in git.

The task label is the primary Potato field `conversation_sense`. Optional `secondary_sense` is not a vote. The annotator id is the submission folder name (`annotator_01` through `annotator_06`), because `user_id` in the JSONL is a display name for some classmates. Each folder contains the same JSONL twice. The loader reads `annotation_submissions/annotator_*/annotations.jsonl` only.

Six annotators submitted 582 labels on 436 conversations. 406 conversations have one annotator. 30 were shared: 26 by all six annotators, and 4 without `annotator_05` (82 of 100 items). On the shared set, 5 conversations are unanimous and 25 are not.

## Step 2. Agreement

**Choice.** All four metrics named in the assignment, each from a library, and only on conversations with at least two labels.

| Metric | Library | What was scored |
| --- | --- | --- |
| Cohen's kappa | `sklearn.metrics.cohen_kappa_score` | Each of the 15 annotator pairs, on the conversations both labeled |
| Percentage agreement | `sklearn.metrics.accuracy_score` | The same pairs. The score is the fraction of items where the two labels match |
| Fleiss' kappa | `statsmodels.stats.inter_rater.fleiss_kappa` | Conversations labeled by every annotator. That implementation requires the same rater count on every row |
| Krippendorff's alpha | `simpledorff`, nominal metric | All 30 shared conversations. Missing labels stay missing |

**Main metric.** Krippendorff's alpha (0.427) is the agreement figure the team reports. It works for more than two annotators and nominal labels, corrects for chance, and keeps the 4 shared conversations `annotator_05` skipped. Fleiss' kappa drops those conversations, Cohen's kappa scores only one pair at a time, and percentage agreement does not correct for chance. The other two chance-corrected scores fall within 0.02 of alpha.

**Why.** A conversation with one label cannot show agreement, so the 406 singly labeled rows are left out. Fleiss drops the 4 shared conversations `annotator_05` skipped. Alpha keeps them, which is the missing-data case that metric is for. Cohen's kappa and percentage agreement in the summary are the unweighted mean of the 15 pairs. Pairs have 26 or 30 conversations.

The script is `src/annotation_analysis/agreement.py`, run through `src/annotation_analysis/determine_ground_truth.py`.

| Metric | Score |
| --- | ---: |
| Cohen's kappa | 0.434 |
| Fleiss' kappa | 0.447 |
| Krippendorff's alpha | 0.427 |
| Percentage agreement | 0.566 |

Chance-corrected agreement is moderate. Raw agreement is higher, as expected, because percentage agreement does not subtract chance.

The highest pair is annotator 05 and annotator 06: Cohen's kappa 0.755 on 26 conversations, percentage agreement 0.808. The lowest pair is annotator 01 and annotator 05: Cohen's kappa 0.264 on 26 conversations, percentage agreement 0.423.

`outputs/annotator_agreement_summary.csv` holds the four scores. `outputs/pairwise_annotator_agreement.csv` holds every pair. `outputs/annotator_agreement.png` is the summary bars plus the pairwise Cohen heatmap.

## Step 3. Ground truth

A step-by-step explanation, with worked examples, is in `docs/ground_truth_method.md`.

**Choice.** Two labelings, both written out. Neither breaks a tie by alphabetical order.

| Method | How | Library |
| --- | --- | --- |
| Majority vote | The unique modal label. A tie stays unlabeled for adjudication | `pandas.DataFrame.mode` |
| Reliability-weighted vote | The label with the greatest total annotator weight. A tie stays unlabeled | `DataFrame.groupby` sum of weights. Ties use `numpy.isclose` |

A conversation with one annotator keeps that label under both methods.

**Weights.** Each annotator's weight is their mean pairwise Cohen's kappa with the others (`sklearn.metrics.cohen_kappa_score`). There is no separate experience score in the submissions. A non-positive mean would be clipped to zero. All six weights are positive.

| Annotator | Reliability weight |
| --- | ---: |
| annotator_01 | 0.348 |
| annotator_02 | 0.444 |
| annotator_03 | 0.483 |
| annotator_04 | 0.394 |
| annotator_05 | 0.479 |
| annotator_06 | 0.457 |

**Why both methods.** The assignment lists majority vote, weighted vote, and adjudication. Majority vote is the usual rule and leaves the two real ties for a person to resolve. The weighted vote is what the competition files use, because a competition row needs one label and this vote assigns one to all 436 conversations.

| Resolution | Majority vote | Weighted vote |
| --- | ---: | ---: |
| Single annotator | 406 | 406 |
| Decided by the vote | 28 | 30 |
| Unresolved tie | 2 | 0 |

The two majority ties, both with six annotators:

| Conversation | Split | Weighted-vote label |
| --- | --- | --- |
| 1003675 | Appreciative 3, Help-seeking 3 | Appreciative or positive |
| 1257160 | Frustrated 3, Escalatory 3 | Escalatory or requesting immediate resolution |

Both still appear in `outputs/adjudication_disagreements.csv`, which lists all 25 shared conversations where the annotators did not all choose the same label. The `vote_summary` column gives the counts.

`outputs/ground_truth_majority_vote.csv`, `outputs/ground_truth_weighted_vote.csv`, and `outputs/annotator_reliability_weights.csv` are the tables. `outputs/ground_truth_label_distribution.png` compares the two label counts. They differ only on those two ties.

## Step 4. Label distribution

**Choice.** Count each tone with `Series.value_counts`, and test balance with `scipy.stats.chisquare` against an equal share of each class. The classes are imbalanced when p < 0.05. The chart uses the reliability-weighted labels so every conversation is included.

**Why.** Step 4 asks for the amount of data in each class and a report on balance. The chi-square test is the comparison to a balanced dataset. Majority vote, with the two ties dropped, is tested as well.

Weighted vote, 436 conversations. An equal share would be 87.2 conversations per tone.

| Tone | Conversations | Share |
| --- | ---: | ---: |
| Appreciative or positive | 50 | 11.5% |
| Neutral or informational | 54 | 12.4% |
| Help-seeking or asking a question | 86 | 19.7% |
| Frustrated or complaining | 191 | 43.8% |
| Escalatory or requesting immediate resolution | 55 | 12.6% |

| Method | Conversations | Chi-square | p-value | Balanced |
| --- | ---: | ---: | ---: | --- |
| Majority vote | 434 | 166.3 | < 0.001 | No |
| Reliability-weighted vote | 436 | 164.0 | < 0.001 | No |

Frustrated or complaining is almost four times the smallest class. The classes are imbalanced under both votes. That is why the competition metric is macro F1.

`outputs/label_distribution.csv`, `outputs/label_balance.csv`, and `outputs/label_distribution.png`. The dashed line on the chart is the equal share.

## Step 5. Competition files

### Ground-truth labels

**Choice.** Reliability-weighted vote, from `outputs/ground_truth_weighted_vote.csv`.

**Why.** A competition row needs one label. The weighted vote assigns a tone to all 436 annotated conversations. Majority vote leaves two ties unlabeled: `1003675` and `1257160`. With the split built from the weighted labels, both of those conversations fell in validation, so the test answer key does not depend on how the two ties are resolved. Validation is still used to tune and refit the baselines, so the tie labels can shift baseline scores slightly, and a split built from majority-vote labels would be a different split. `docs/ground_truth_method.md` has the numbers. The files use the weighted labels.

### Split

**Choice.** Stratified 60% / 20% / 20% by tone, seed 42, via `sklearn.model_selection.train_test_split`. The test half is stratified again into Public and Private.

**Why.** The assignment asks for a training set, a validation set, and a test set, with every class in the test set. The smallest tone has 50 conversations. A 20% test set leaves 10 of those for testing, and the public/private cut leaves 5 on each leaderboard. Seed 42 matches the seed already used to build the annotator packages.

The script is `src/competition/prepare_competition_files.py`.

| Tone | Train | Validation | Test | Public | Private |
| --- | ---: | ---: | ---: | ---: | ---: |
| Appreciative or positive | 30 | 10 | 10 | 5 | 5 |
| Neutral or informational | 32 | 11 | 11 | 6 | 5 |
| Help-seeking or asking a question | 52 | 17 | 17 | 8 | 9 |
| Frustrated or complaining | 114 | 38 | 39 | 20 | 19 |
| Escalatory or requesting immediate resolution | 33 | 11 | 11 | 5 | 6 |
| **Total** | **261** | **87** | **88** | **44** | **44** |

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

**Why.** Step 4 found the tones are imbalanced. Frustrated or complaining is 43.8% of the 436 labeled conversations, and Appreciative or positive is 11.5%. A chi-square test against an equal share per tone gives p < 0.001. Macro F1 is the unweighted mean of the five per-class F1 scores, so each tone counts the same. Accuracy would follow the largest tone.

A label in a submission has to match one of the five tone strings exactly. On Kaggle, map the id column to `id` and the answer column to `label`. If the metric list has macro F1, select it with beta 1. If the page asks for a metric file, paste `src/competition/macro_f1.py`.

**Check.** Submitting `perfect_submission.csv` scores 1 on the public and private leaderboards. Submitting `sample_submission.csv` scores 0.125 public and 0.121 private (`competition/private/score_check.txt` records the private score as 0.1206).

## Step 6. Baselines

A step-by-step explanation of the dummy and trained baselines is in `docs/baselines_method.md`.

All scores below are macro F1 from the competition scorer. Full-test numbers are in `baselines/baseline_scores.csv`. Per-class F1 is in `baselines/per_class_f1.csv`.

### Dummy baseline

**Choice.** Uniform random labels, `sklearn.dummy.DummyClassifier(strategy="uniform")`, seed 16. Upload `competition/submissions/dummy_baseline.csv` and mark it as a Benchmark.

**Why.** The assignment says to keep whichever of the random baseline and the majority baseline scores higher. Compared on the validation set:

| Dummy | How it was scored | Validation macro F1 |
| --- | --- | ---: |
| Majority | Always predicts Frustrated or complaining | 0.122 |
| Random | Mean of seeds 0–19 | 0.194 |

The majority baseline has an F1 near 0.61 on Frustrated or complaining and 0 on the other four tones, so the macro average stays near 0.12. A uniform draw scores a little on every tone, so its macro F1 is higher.

One random draw moves around on 87 validation rows. Seed 16 is the draw whose validation score (0.193) is closest to the 20-seed mean (0.194). The seed was chosen from validation scores only. Its test score is 0.160, next to the 20-seed test mean of 0.167. `majority_baseline.csv` is the same file as `sample_submission.csv`.

### Trained baseline

**Choice.** Word and bigram TF-IDF on `last_customer_message` and on `conversation`, then `sklearn.linear_model.LogisticRegression`. Upload `competition/submissions/logistic_regression.csv` and mark it as a Benchmark.

**Why.** The assignment asks for one simple trained model. Logistic regression on TF-IDF features runs locally on the 261 training conversations. The grid below was fit on train and scored on validation. The test labels were not used to pick the settings.

| C | class_weight | Validation macro F1 |
| ---: | --- | ---: |
| 0.25 | balanced | 0.396 |
| 4.0 | balanced | 0.328 |
| 1.0 | balanced | 0.311 |
| 4.0 | none | 0.209 |
| 1.0 | none | 0.159 |
| 0.25 | none | 0.122 |

`class_weight="balanced"` raises the rare tones, which is what macro F1 measures. `C=0.25` won. The validation score 0.396 is that model fit on train only. The submission file is the same settings refit on train and validation together, so the leaderboard model sees every labeled conversation the participants receive.

### Full-test scores

| Model | Validation | Public | Private | Full test (88) |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.122 | 0.125 | 0.121 | 0.123 |
| Random, seed 16 | 0.193 | 0.181 | 0.139 | 0.160 |
| Random, mean of 20 seeds | 0.194 |  |  | 0.167 |
| Logistic regression | 0.396 | 0.339 | 0.377 | 0.378 |

On Kaggle, the random benchmark should show about **0.181** public and **0.139** private. The logistic regression benchmark should show about **0.339** public and **0.377** private.

Per-class F1 for logistic regression on the full test set:

| Tone | Test F1 |
| --- | ---: |
| Appreciative or positive | 0.500 |
| Neutral or informational | 0.133 |
| Help-seeking or asking a question | 0.323 |
| Frustrated or complaining | 0.519 |
| Escalatory or requesting immediate resolution | 0.414 |

Neutral or informational is the weak class. Its private-leaderboard F1 is 0.

### LLM baseline

**Choice.** GPT-5 in the ChatGPT web interface, zero-shot, on 50 of the 88 test conversations. This submission stays off the leaderboard.

**Why.** The assignment asks for a frontier model on at least 50 test items, with no labeled examples in the prompt, and says this run stays off the leaderboard unless every test item is included. There was no API key in the environment, so the run was done by hand in a new ChatGPT conversation. An earlier run with Grok 4.7 read the files inside a coding session; it could not be verified and was replaced (kept as `baselines/llm/predictions_grok.csv` and `run_grok.json`, macro F1 0.404).

**How it was sent.** `src/baselines/llm_baseline.py write` drew 50 test ids with seed 42 and wrote them to `baselines/llm/items.jsonl` with no labels. The instruction is `baselines/llm/prompt.txt`: the five tone definitions, the precedence order (Escalatory, Frustrated, Help-seeking, Appreciative, Neutral), and the output format. Worked examples from the annotation guidelines were excluded from the draw. None of those example ids are in the test set. The prompt text and all 50 items went in one message, with no labeled examples. The reply is saved unedited in `baselines/llm/raw_responses.jsonl` and converted to `baselines/llm/predictions.csv`. The run record is `baselines/llm/run.json`.

**Output.** Macro F1 on those 50 items is **0.441**.

| Tone | F1 on the 50 |
| --- | ---: |
| Appreciative or positive | 0.667 |
| Neutral or informational | 0.235 |
| Help-seeking or asking a question | 0.308 |
| Frustrated or complaining | 0.526 |
| Escalatory or requesting immediate resolution | 0.471 |

Same 50 conversations, so the comparison uses one item set (`baselines/llm_sample_comparison.csv`):

| Model | Macro F1 on the 50 |
| --- | ---: |
| Majority | 0.118 |
| Random, seed 16 | 0.143 |
| Logistic regression | 0.300 |
| GPT-5, zero-shot | 0.441 |

The zero-shot model is ahead of logistic regression on this sample and well below a perfect score. Appreciative or positive is the easy tone. Help-seeking is low partly because the model used that label for only 3 of the 50 conversations. Neutral or informational is hard for both the logistic regression and the LLM, which matches the imbalance found in step 4: the flat, factual tone is easy to miss when Frustrated or complaining is almost half the data.

## Commands

```bash
MPLCONFIGDIR=/tmp/mplconfig /opt/anaconda3/bin/python src/annotation_analysis/determine_ground_truth.py
/opt/anaconda3/bin/python src/competition/prepare_competition_files.py
/opt/anaconda3/bin/python src/baselines/train_baselines.py
/opt/anaconda3/bin/python src/baselines/llm_baseline.py score baselines/llm/predictions.csv
```

The first command rewrites `outputs/`: agreement, both ground truths, the disagreement list, and the label-distribution chart. The second rewrites the split. The third refits the dummy and logistic regression baselines and rewrites `baselines/baseline_scores.csv`. The fourth scores the saved LLM predictions.
