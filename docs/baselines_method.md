# How the baselines were derived

This is a reference for step 6 of `docs/ARI410-510_Project_Leaderboard.pdf`, "Create Baselines". It covers the dummy baseline and the trained model, which both go on the Kaggle leaderboard as Benchmarks. The LLM baseline is not covered here.

Code: `src/baselines/train_baselines.py`. Run it from the repository root after `src/competition/prepare_competition_files.py`:

```bash
/opt/anaconda3/bin/python src/baselines/train_baselines.py
```

## 1. What the assignment asks for

| Requirement | What we did |
| --- | --- |
| A dummy baseline: random or majority, "whichever of these two gives the best performance" | Built both, compared them on validation, and kept random |
| At least one simple trained model | TF-IDF features with logistic regression |
| Put the dummy and trained baselines on the leaderboard as Benchmarks | `competition/submissions/dummy_baseline.csv` and `competition/submissions/logistic_regression.csv` |

## 2. The data each step uses

| File | Rows | Used for |
| --- | ---: | --- |
| `competition/participant/train.csv` | 302 | Fitting every model |
| `competition/participant/validation.csv` | 103 | Choosing the dummy, the random seed and the logistic regression settings, then refitting the final model |
| `competition/participant/test.csv` | 101 | Making the submission files (no labels) |
| `competition/private/solution.csv` | 101 | Reporting test, public and private scores only |

**Rule:** every choice is made on validation. The test labels are read only to report scores, never to pick a setting. They are not written into any output, and the submission files contain only `id` and `label`.

## 3. The metric

Every score uses the competition scorer, `src/competition/macro_f1.py`. It calls `sklearn.metrics.f1_score(average="macro")` over the tones present in the answer key. Macro F1 is the F1 of each tone, averaged with equal weight, so each of the five tones counts the same no matter how common it is.

Public and private scores use the rows of `solution.csv` marked `Usage = Public` or `Private`, exactly as Kaggle will.

## 4. Majority baseline

**How:** find the most common tone in `train.csv` and predict it for every row. That tone is Frustrated or complaining, with 130 of 302 training rows. If two tones were tied, the code would pick the one that comes first in the guideline order. That didn't happen.

The code writes the constant label directly. This gives the same output as `sklearn.dummy.DummyClassifier(strategy="most_frequent")`, which is the name used in the notes column of `baselines/baseline_scores.csv`.

**Why it scores so low:** it gets every Frustrated row right and every other row wrong.

- Frustrated: recall 1.0 and precision 43/101 = 0.426 on the full test set, so F1 = 0.597.
- The other four tones: F1 = 0.
- Macro F1 = 0.597 ÷ 5 = **0.119**.

`competition/submissions/majority_baseline.csv` is identical to `competition/participant/sample_submission.csv`. The script checks this.

## 5. Random baseline

**How:** `sklearn.dummy.DummyClassifier(strategy="uniform", random_state=seed)` picks one of the five tones with equal probability for every row. It ignores the text. It is fitted on the training labels only so it knows the five class names.

**Which seed:** a single random draw is noisy on 103 validation rows. Across seeds 0–19, validation macro F1 ranges from 0.142 to 0.281. To avoid reporting a lucky or unlucky draw:

1. Score seeds 0 to 19 on validation. The mean is 0.196.
2. Pick the seed whose validation score is closest to that mean. If two were equally close, the lower seed would win.
3. The result is **seed 3**, with a validation score of 0.191.

The seed is picked from validation scores only. The script also records each seed's test score (mean 0.193, range 0.103 to 0.303), but those scores play no part in the choice. Seed 3's test score is 0.208, close to the test mean.

All 20 seeds are in `baselines/random_baseline_seeds.csv`.

## 6. Choosing the dummy

The assignment says to keep whichever dummy does better. The comparison uses validation, with the 20-seed mean standing in for random, so that one draw can't decide it:

| Dummy | Validation macro F1 |
| --- | ---: |
| Majority | 0.120 |
| Random, mean of seeds 0–19 | 0.196 |

**Random wins.** Under macro F1, majority gets zero on four of the five tones, while uniform guessing gets a little right on every tone. The leaderboard dummy, `dummy_baseline.csv`, contains seed 3's predictions.

## 7. Trained baseline: TF-IDF + logistic regression

### Features

There are two TF-IDF blocks side by side, built with `ColumnTransformer`:

| Column | Vectorizer settings |
| --- | --- |
| `last_customer_message` | Words and two-word phrases (`ngram_range=(1, 2)`), `min_df=1`, `sublinear_tf=True` |
| `conversation` | Words and two-word phrases, `min_df=2`, `sublinear_tf=True` |

- `last_customer_message` is the span the label describes, so it gets its own features. Every word in it counts, even if it appears only once.
- `conversation` adds the context from earlier turns. It also contains the last message, so those words appear in both blocks. `min_df=2` drops phrases seen in only one conversation, which keeps the larger vocabulary in check.
- `sublinear_tf` dampens repeated words. For example, "!!!" repeated five times doesn't count five times as much.

### Classifier

`sklearn.linear_model.LogisticRegression` with `solver="lbfgs"`, `max_iter=2000` and `random_state=42`. The script stops with an error if any fit reaches the iteration limit without converging. None did.

### Choosing the settings on validation

The script fits on `train.csv` and scores on `validation.csv` for six combinations:

| C | class_weight | Validation macro F1 |
| ---: | --- | ---: |
| **0.25** | **balanced** | **0.330** |
| 4.0 | balanced | 0.328 |
| 1.0 | balanced | 0.315 |
| 4.0 | none | 0.217 |
| 1.0 | none | 0.150 |
| 0.25 | none | 0.120 |

- **C** controls regularization. A smaller C means a simpler model, which helps with only 302 training rows.
- **`class_weight="balanced"`** weights each tone inversely to how common it is. Without it, the model leans toward Frustrated. At C = 0.25 without balancing, it predicts Frustrated for every row and scores exactly the majority baseline's 0.120. Balanced weighting lifts the rare tones, which is what macro F1 rewards.
- If two settings scored the same, the rule would prefer balanced, then the smaller C. That didn't happen, but C = 0.25 beat C = 4.0 by only 0.003, so the choice is close.

The full grid is in `baselines/logistic_regression_search.csv`.

### The submitted model

The chosen settings (C = 0.25, balanced) are **refit on train plus validation (405 rows)**, and that model predicts `test.csv`. This gives the leaderboard model every labeled conversation that participants also receive.

So the reported validation score (0.330) is from the train-only fit, and the test scores are from the refit model.

## 8. Results

| Model | Validation | Public | Private | Full test (101) |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.120 | 0.118 | 0.121 | 0.119 |
| Random, seed 3 (leaderboard dummy) | 0.191 | 0.266 | 0.148 | 0.208 |
| Random, mean of 20 seeds | 0.196 | | | 0.193 |
| Logistic regression | 0.330 | 0.398 | 0.463 | 0.428 |

On Kaggle, the two Benchmarks should show about **0.266 public / 0.148 private** (random) and **0.398 public / 0.463 private** (logistic regression).

The test scores are higher than the validation score. Validation and test are small, and the submitted model is refit on 405 rows instead of 302. The settings were chosen on validation only, so the gap is not from looking at test.

Logistic regression F1 per tone:

| Tone | Validation | Public | Private | Full test |
| --- | ---: | ---: | ---: | ---: |
| Appreciative or positive | 0.348 | 0.615 | 0.800 | 0.696 |
| Neutral or informational | 0.182 | 0.000 | 0.308 | 0.167 |
| Help-seeking or asking a question | 0.286 | 0.250 | 0.375 | 0.313 |
| Frustrated or complaining | 0.654 | 0.625 | 0.478 | 0.553 |
| Escalatory or requesting immediate resolution | 0.182 | 0.500 | 0.353 | 0.414 |

Neutral or informational is the weakest tone; its public F1 is 0. It has no strong cue words, and flat statements about a late order look a lot like complaints. Annotators confuse Neutral with Help-seeking and Frustrated too (see `docs/ground_truth_method.md`).

## 9. Is the logistic regression benchmark good enough?

On a 0-to-1 scale, the logistic regression scores look low. For this task they are a reasonable reference point, and fine to put on the leaderboard.

### Which number to quote

Logistic regression has four scores, and they are used for different things:

| Score | Value | Who sees it |
| --- | ---: | --- |
| Validation (train-only fit) | 0.330 | The team; used to choose the settings |
| **Public leaderboard** | **0.398** | **Classmates, during the competition. This is the benchmark they compare against.** |
| Private leaderboard | 0.463 | Everyone, at the Phase 2b deadline. It decides the final ranking. |
| Full test, all 101 rows | 0.428 | Only the team. Kaggle never shows a full-test score. |

The full-test score is not a number anyone sees on Kaggle. When talking to classmates, quote **0.398** as the number to beat, and mention 0.463 as the private benchmark for the final ranking.

### Reference points

| Reference | Macro F1 |
| --- | ---: |
| Always predict Frustrated (majority) | 0.12 |
| Random guessing (leaderboard dummy), public / private | 0.266 / 0.148 |
| **Logistic regression, public leaderboard** | **0.398** |
| Logistic regression, private leaderboard | 0.463 |
| Logistic regression, full test | 0.428 |
| LLM zero-shot, GPT-5 (50 test items; see `docs/steps_2_to_6.md`) | 0.441 |

All of these are scored on test or validation conversations. The annotator result below is on a different set, so it is kept out of this table.

### How annotators score against each other

The labels themselves are noisy (Krippendorff's alpha 0.433 on the shared conversations), so annotators don't agree perfectly with each other either. To see how consistent individual annotators are, each one was scored against the other annotators on the 30 shared conversations:

1. Hold out one annotator.
2. Label each shared conversation with the reliability-weighted vote of the remaining annotators (weights from `outputs/annotator_reliability_weights.csv`).
3. Score the held-out annotator's labels against that vote with macro F1 and accuracy.
4. Repeat for all seven annotators.

| Annotator | Items | Macro F1 vs. the others | Accuracy vs. the others |
| --- | ---: | ---: | ---: |
| A1 | 30 | 0.493 | 0.633 |
| A2 | 30 | 0.472 | 0.700 |
| A3 | 30 | 0.530 | 0.700 |
| A4 | 30 | 0.457 | 0.600 |
| A5 | 26 | 0.487 | 0.654 |
| A6 | 30 | 0.482 | 0.667 |
| A7 | 30 | 0.563 | 0.733 |
| **Mean** | | **0.498** | **0.670** |

On the 30 shared conversations, annotators averaged 0.498 macro F1 against the weighted consensus of the other annotators. Logistic regression was evaluated on a different set of 101 test conversations, so these scores aren't a direct human–model comparison.

This result shows the labels are subjective. It is not an upper limit for models. A model is scored against a consensus label, which is less noisy than any single annotator, so a model could score higher than an individual annotator does.

Caveats:

- The result rests on only 30 conversations, and the per-annotator scores range from 0.457 to 0.563, so treat 0.498 as rough.
- The 30 shared conversations were not a random sample. `src/pipeline/make_overlap_annotator_packages.py` picked cases a rule-based checker rated as clear-cut, six per estimated tone. Annotators may agree less on the rest of the data.
- The weights come from all seven annotators, including the one held out. A strict version would recompute the weights without them. Weights only matter on ties, so the effect is small.
- This number is not produced by any script yet. To regenerate it, run the snippet below from the repository root. It needs the local `annotation_submissions/` folder, which is not in git.

```python
import glob, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

rows = []
for folder in sorted(glob.glob("annotation_submissions/annotator_*")):
    path = folder + "/annotations.jsonl"
    if not os.path.isfile(path):  # annotator_07 has only Potato's export
        path = folder + "/exports/jsonl/annotations.jsonl"
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        label = next(iter(record["labels"]["conversation_sense"]))
        rows.append((str(record["instance_id"]), os.path.basename(folder), label))
matrix = pd.DataFrame(rows, columns=["id", "annotator", "label"]).pivot(
    index="id", columns="annotator", values="label"
)
shared = matrix[matrix.notna().sum(axis=1) >= 5]
weights = pd.read_csv("outputs/annotator_reliability_weights.csv").set_index(
    "annotator_id"
)["reliability_weight"]

for held_out in shared.columns:
    truth, predicted = [], []
    for _, row in shared.iterrows():
        if pd.isna(row[held_out]):
            continue
        others = row.drop(held_out).dropna()
        totals = others.groupby(others).apply(lambda votes: weights[votes.index].sum())
        truth.append(totals.idxmax())
        predicted.append(row[held_out])
    print(
        held_out,
        len(truth),
        round(f1_score(truth, predicted, average="macro", zero_division=0), 3),
        round(float(np.mean(np.array(truth) == np.array(predicted))), 3),
    )
```

### Why the task scores low

- **Small training data:** 302 training conversations, or 405 with validation.
- **Five subtle tones.** The boundaries that confuse models (Frustrated vs. Escalatory, Neutral vs. Help-seeking) confuse annotators too.
- **Macro F1 is strict.** Neutral has 16 test rows, and its F1 of 0.17 pulls the average down as much as any common tone would.
- **Bag-of-words features can't read sarcasm.** "Thanks a lot" can be praise or a complaint.

### Why it's the right kind of baseline

The assignment says the trained model "does not need to be a strong model; it is a reference point for everyone else." A useful benchmark:

- clearly beats the dummy, which shows the task is learnable (0.398 vs. 0.266 on the public leaderboard; 0.463 vs. 0.148 on the private one), and
- leaves visible room for other teams. On the same 50 test conversations, GPT-5 zero-shot scores 0.44 and logistic regression scores 0.30, so a stronger model already does better.

A very strong baseline would discourage submissions. The slide 12 pitch, "Beat the 0.398 Benchmark", relies on that headroom.

Suggested wording for the talk:

> Our logistic regression benchmark scores 0.398 macro F1 on the public leaderboard, against 0.266 for the random dummy. The final ranking uses the private leaderboard, where it scores 0.463. The labels are subjective: on our 30 shared conversations, annotators averaged 0.498 against each other's consensus. On the same 50 test conversations, GPT-5 zero-shot already beats logistic regression, 0.44 to 0.30, so there's room for better models.

## 10. Checks the script runs

- The ids in `solution.csv` match `test.csv`, in the same order.
- Every submission file has exactly the columns `id,label` and the same ids as `test.csv`.
- `majority_baseline.csv` has the same ids and labels as `sample_submission.csv`.
- Every prediction is one of the five tone names.
- Logistic regression converged on every fit.

## 11. Limitations

- **Validation is small (103 rows).** One or two rows can change which setting wins, and validation macro F1 is noisy. The seed range of 0.142 to 0.281 shows how much luck alone can move a score.
- **Public and private halves are tiny (50 and 51 rows).** A rare tone has 5 or 6 rows per half, so one prediction can move that tone's F1 a lot. That is why logistic regression scores 0.398 public and 0.463 private, and the random dummy 0.266 and 0.148.
- **The grid is deliberately small.** It has six settings, because this is a reference point, not a tuned model.
- **The scores depend on the validation labels.** Validation chooses the seed and the settings and is part of the refit. Giving the two tied conversations from the ground-truth step their other label moved logistic regression to 0.318 public / 0.486 private. Details are in `docs/ground_truth_method.md`.

## 12. Files

| File | Contents | In git |
| --- | --- | --- |
| `competition/submissions/dummy_baseline.csv` | Leaderboard dummy (random, seed 3) | Yes |
| `competition/submissions/logistic_regression.csv` | Leaderboard trained model | Yes |
| `competition/submissions/majority_baseline.csv` | Majority predictions, same as the sample submission | Yes |
| `competition/submissions/random_baseline.csv` | Random, seed 3 (same predictions as `dummy_baseline.csv`) | Yes |
| `baselines/baseline_scores.csv` | Macro F1 by model and split | Yes |
| `baselines/per_class_f1.csv` | F1 by model, split and tone | Yes |
| `baselines/logistic_regression_search.csv` | The six-setting grid | Yes |
| `baselines/random_baseline_seeds.csv` | Validation and test score for seeds 0–19 | Yes |

Code locations in `src/baselines/train_baselines.py`:

| Function | Line | Role |
| --- | ---: | --- |
| `_majority_label` | 211 | Most common training tone |
| `_random_labels` | 222 | Uniform `DummyClassifier` for one seed |
| `_random_scores_by_seed` | 230 | Scores seeds 0–19 |
| `_representative_random_seed` | 255 | Seed closest to the validation mean |
| `_choose_dummy` | 267 | Majority vs. random on validation |
| `_search_logistic_regression` | 283 | Six-setting grid on validation |
| `_fit_logistic_regression` | 318 | TF-IDF + logistic regression pipeline |
| `_macro_f1` | 405 | Calls the competition scorer |

See also `docs/steps_2_to_6.md` (step 6) and the slide 10 presenter notes in `docs/phase2a_presenter_notes/`.
