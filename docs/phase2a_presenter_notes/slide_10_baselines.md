# Slide 10: Leaderboard Baselines

**Time:** 0:30

**Goal:** Present the dummy and trained baselines that are on the leaderboard as Benchmarks.

## On the slide

- Macro F1 on validation, public and private for Majority, Random (dummy) and Logistic regression
- Dummy: uniform random, low by design (no-learning floor)
- Trained: TF-IDF + logistic regression
- Hardest tone: Neutral, test F1 0.17

## What to say

> These dummy scores are low on purpose. Neither one reads the text, so they show what you get with no learning at all, the floor any real model has to beat.
>
> We compared two dummy baselines. Always predicting the majority tone scores 0.12, because it gets zero on four of the five tones. Random guessing scores 0.19 on validation, so the random one is our dummy benchmark.
>
> The trained baseline is TF-IDF on words and word pairs with logistic regression, using balanced class weights. It scores 0.398 public and 0.463 private, compared with 0.266 public and 0.148 private for the random dummy. Neutral is the hardest tone: on the full test set its F1 is only 0.17.

## Key facts

| Model | Validation | Public | Private | Full test (101) |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.120 | 0.118 | 0.121 | 0.119 |
| Random, seed 3 (dummy) | 0.191 | 0.266 | 0.148 | 0.208 |
| Logistic regression | 0.330 | 0.398 | 0.463 | 0.428 |

Logistic regression per-tone F1 on the full test set:

| Tone | F1 |
| --- | ---: |
| Appreciative or positive | 0.696 |
| Neutral or informational | 0.167 |
| Help-seeking or asking a question | 0.313 |
| Frustrated or complaining | 0.553 |
| Escalatory or requesting immediate resolution | 0.414 |

- **Why the dummies are low:** macro F1 averages the five tones equally. Majority gets F1 = 0.597 on Frustrated and 0 on the other four, so 0.597 / 5 = 0.119. Random gets a little of each tone right, so it lands near 1/5 = 0.2. With 5 tones, a constant prediction can never score above 0.2.
- Settings were chosen on validation only: C = 0.25, `class_weight="balanced"`. The submitted model was then refit on train plus validation.
- The random seed (3) was the one whose validation score was closest to the 20-seed mean (0.196). It was not picked using test scores.

## If asked

- **Why random over majority?** The assignment says to keep whichever dummy does better, and with macro F1 random wins.
- **Why is the random dummy 0.27 public but 0.15 private?** Each half has about 50 rows, and a uniform draw on that few rows is noisy. Its full-test score, 0.21, is close to the 20-seed mean of 0.19.
- **Why is the test score higher than validation (0.43 vs. 0.33)?** Both sets are small, and the submitted model is refit on train plus validation. The settings were picked on validation only.
- **Why is Neutral so hard?** It has no strong cue words, and flat statements about a late order look a lot like complaints.

Full method reference: [docs/baselines_method.md](../baselines_method.md)

[Previous](slide_09_how_it_works.md) | [Next](slide_11_llm_baseline.md)
