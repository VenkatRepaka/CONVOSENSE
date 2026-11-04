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
| `competition/participant/train.csv` | 261 | Fitting every model |
| `competition/participant/validation.csv` | 87 | Choosing the dummy, the random seed and the logistic regression settings, then refitting the final model |
| `competition/participant/test.csv` | 88 | Making the submission files (no labels) |
| `competition/private/solution.csv` | 88 | Reporting test, public and private scores only |

**Rule:** every choice is made on validation. The test labels are read only to report scores, never to pick a setting. They are not written into any output, and the submission files contain only `id` and `label`.

## 3. The metric

Every score uses the competition scorer, `src/competition/macro_f1.py`. It calls `sklearn.metrics.f1_score(average="macro")` over the tones present in the answer key. Macro F1 is the F1 of each tone, averaged with equal weight, so each of the five tones counts the same no matter how common it is.

Public and private scores use the rows of `solution.csv` marked `Usage = Public` or `Private`, exactly as Kaggle will.

## 4. Majority baseline

**How:** find the most common tone in `train.csv` and predict it for every row. That tone is Frustrated or complaining, with 114 of 261 training rows. If two tones were tied, the code would pick the one that comes first in the guideline order. That didn't happen.

The code writes the constant label directly. This gives the same output as `sklearn.dummy.DummyClassifier(strategy="most_frequent")`, which is the name used in the notes column of `baselines/baseline_scores.csv`.

**Why it scores so low:** it gets every Frustrated row right and every other row wrong.

- Frustrated: recall 1.0 and precision 39/88 = 0.443 on the full test set, so F1 = 0.614.
- The other four tones: F1 = 0.
- Macro F1 = 0.614 ÷ 5 = **0.123**.

`competition/submissions/majority_baseline.csv` is identical to `competition/participant/sample_submission.csv`. The script checks this.

## 5. Random baseline

**How:** `sklearn.dummy.DummyClassifier(strategy="uniform", random_state=seed)` picks one of the five tones with equal probability for every row. It ignores the text. It is fitted on the training labels only so it knows the five class names.

**Which seed:** a single random draw is noisy on 87 validation rows. Across seeds 0–19, validation macro F1 ranges from 0.137 to 0.274. To avoid reporting a lucky or unlucky draw:

1. Score seeds 0 to 19 on validation. The mean is 0.194.
2. Pick the seed whose validation score is closest to that mean. If two were equally close, the lower seed would win.
3. The result is **seed 16**, with a validation score of 0.193.

The seed is picked from validation scores only. The script also records each seed's test score (mean 0.167, range 0.116 to 0.266), but those scores play no part in the choice. Seed 16's test score is 0.160, close to the test mean.

All 20 seeds are in `baselines/random_baseline_seeds.csv`.

## 6. Choosing the dummy

The assignment says to keep whichever dummy does better. The comparison uses validation, with the 20-seed mean standing in for random, so that one draw can't decide it:

| Dummy | Validation macro F1 |
| --- | ---: |
| Majority | 0.122 |
| Random, mean of seeds 0–19 | 0.194 |

**Random wins.** Under macro F1, majority gets zero on four of the five tones, while uniform guessing gets a little right on every tone. The leaderboard dummy, `dummy_baseline.csv`, contains seed 16's predictions.

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
| **0.25** | **balanced** | **0.396** |
| 4.0 | balanced | 0.328 |
| 1.0 | balanced | 0.311 |
| 4.0 | none | 0.209 |
| 1.0 | none | 0.159 |
| 0.25 | none | 0.122 |

- **C** controls regularization. A smaller C means a simpler model, which helps with only 261 training rows.
- **`class_weight="balanced"`** weights each tone inversely to how common it is. Without it, the model leans toward Frustrated. At C = 0.25 without balancing, it predicts Frustrated for every row and scores exactly the majority baseline's 0.122. Balanced weighting lifts the rare tones, which is what macro F1 rewards.
- If two settings scored the same, the rule would prefer balanced, then the smaller C. That didn't happen.

The full grid is in `baselines/logistic_regression_search.csv`.

### The submitted model

The chosen settings (C = 0.25, balanced) are **refit on train plus validation (348 rows)**, and that model predicts `test.csv`. This gives the leaderboard model every labeled conversation that participants also receive.

So the reported validation score (0.396) is from the train-only fit, and the test scores are from the refit model.

## 8. Results

| Model | Validation | Public | Private | Full test (88) |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.122 | 0.125 | 0.121 | 0.123 |
| Random, seed 16 (leaderboard dummy) | 0.193 | 0.181 | 0.139 | 0.160 |
| Random, mean of 20 seeds | 0.194 | | | 0.167 |
| Logistic regression | 0.396 | 0.339 | 0.377 | 0.378 |

On Kaggle, the two Benchmarks should show about **0.181 public / 0.139 private** (random) and **0.339 public / 0.377 private** (logistic regression).

Logistic regression F1 per tone:

| Tone | Validation | Public | Private | Full test |
| --- | ---: | ---: | ---: | ---: |
| Appreciative or positive | 0.444 | 0.286 | 0.615 | 0.500 |
| Neutral or informational | 0.143 | 0.200 | 0.000 | 0.133 |
| Help-seeking or asking a question | 0.343 | 0.353 | 0.286 | 0.323 |
| Frustrated or complaining | 0.667 | 0.524 | 0.513 | 0.519 |
| Escalatory or requesting immediate resolution | 0.385 | 0.333 | 0.471 | 0.414 |

Neutral or informational is the weakest tone; its private F1 is 0. It has no strong cue words, and flat statements about a late order look a lot like complaints. Annotators confuse Neutral with Help-seeking and Frustrated too (see `docs/ground_truth_method.md`).

## 9. Is 0.38 a good baseline?

On a 0-to-1 scale, 0.396 on validation and 0.378 on test look low. For this task they are a reasonable reference point, and fine to put on the leaderboard.

### Reference points

| Reference | Macro F1 |
| --- | ---: |
| Always predict Frustrated (majority) | 0.12 |
| Random guessing (leaderboard dummy) | 0.19 |
| **Logistic regression, validation** | **0.40** |
| **Logistic regression, full test** | **0.38** |
| LLM zero-shot, GPT-5 (50 test items; see `docs/steps_2_to_6.md`) | 0.44 |
| **One annotator vs. the other annotators** (estimate below) | **~0.54** |

### Human-level estimate

The labels themselves are noisy (Krippendorff's alpha 0.427), so even a person does not score near 1.0. To estimate how well a single human does, each annotator was scored against the other annotators on the 30 shared conversations:

1. Hold out one annotator.
2. Label each shared conversation with the reliability-weighted vote of the remaining annotators (weights from `outputs/annotator_reliability_weights.csv`).
3. Score the held-out annotator's labels against that vote with macro F1 and accuracy.
4. Repeat for all six annotators.

| Annotator | Items | Macro F1 vs. the others | Accuracy vs. the others |
| --- | ---: | ---: | ---: |
| A1 | 30 | 0.467 | 0.600 |
| A2 | 30 | 0.472 | 0.700 |
| A3 | 30 | 0.711 | 0.700 |
| A4 | 30 | 0.437 | 0.567 |
| A5 | 26 | 0.624 | 0.692 |
| A6 | 30 | 0.511 | 0.633 |
| **Mean** | | **0.537** | **0.649** |

A careful human reaches about 0.54 macro F1. No model can reliably beat the consistency of the labels it is scored against, so a realistic ceiling is roughly 0.5–0.6, not 1.0. Logistic regression gets about 70% of the way there (0.378 ÷ 0.537).

Caveats:

- The estimate rests on only 30 conversations, and the per-annotator scores range from 0.44 to 0.71, so treat ~0.54 as rough.
- The weights come from all six annotators, including the one held out. A strict version would recompute the weights without them. Weights only matter on ties, so the effect is small.
- It compares humans on the 30 shared conversations, while the models are scored on 88 test conversations. The two numbers are a guide, not a like-for-like comparison.
- This number is not produced by any script yet. To regenerate it, run the snippet below from the repository root. It needs the local `annotation_submissions/` folder, which is not in git.

```python
import glob, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

rows = []
for folder in sorted(glob.glob("annotation_submissions/annotator_*")):
    for line in open(folder + "/annotations.jsonl", encoding="utf-8"):
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

- **Small training data:** 261 training conversations, or 348 with validation.
- **Five subtle tones.** The boundaries that confuse models (Frustrated vs. Escalatory, Neutral vs. Help-seeking) confuse annotators too.
- **Macro F1 is strict.** Neutral has 11 test rows, and its F1 of 0.13 pulls the average down as much as any common tone would.
- **Bag-of-words features can't read sarcasm.** "Thanks a lot" can be praise or a complaint.

### Why it's the right kind of baseline

The assignment says the trained model "does not need to be a strong model; it is a reference point for everyone else." A useful benchmark:

- clearly beats the dummy, which shows the task is learnable (0.38 vs. 0.19), and
- leaves visible room for other teams (0.38 vs. about 0.54 for humans).

A very strong baseline would discourage submissions. The slide 12 pitch, "Beat 0.38 Macro F1", relies on that headroom.

Suggested wording for the talk:

> Our logistic regression reaches 0.38 macro F1, twice the random dummy. For reference, a single annotator scores about 0.54 against the others, so labels this subjective cap what any model can reach. There's real room between 0.38 and human level for better models.

## 10. Checks the script runs

- The ids in `solution.csv` match `test.csv`, in the same order.
- Every submission file has exactly the columns `id,label` and the same ids as `test.csv`.
- `majority_baseline.csv` has the same ids and labels as `sample_submission.csv`.
- Every prediction is one of the five tone names.
- Logistic regression converged on every fit.

## 11. Limitations

- **Validation is small (87 rows).** One or two rows can change which setting wins, and validation macro F1 is noisy. The seed range of 0.137 to 0.274 shows how much luck alone can move a score.
- **Public and private halves are tiny (44 rows each).** A rare tone has 5 or 6 rows per half, so one prediction can move that tone's F1 a lot. That is why logistic regression scores 0.339 public and 0.377 private.
- **The grid is deliberately small.** It has six settings, because this is a reference point, not a tuned model.
- **The scores depend on the validation labels.** Validation chooses the seed and the settings and is part of the refit. Changing the two tied validation labels from the ground-truth step moved logistic regression to 0.357 public / 0.348 private. Details are in `docs/ground_truth_method.md`.

## 12. Files

| File | Contents | In git |
| --- | --- | --- |
| `competition/submissions/dummy_baseline.csv` | Leaderboard dummy (random, seed 16) | Yes |
| `competition/submissions/logistic_regression.csv` | Leaderboard trained model | Yes |
| `competition/submissions/majority_baseline.csv` | Majority predictions, same as the sample submission | Yes |
| `competition/submissions/random_baseline.csv` | Random, seed 16 (same predictions as `dummy_baseline.csv`) | Yes |
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
