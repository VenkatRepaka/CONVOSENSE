# Slide 10: Leaderboard Baselines

**Time:** 0:30

**Goal:** Present the dummy and trained baselines that are on the leaderboard as Benchmarks.

## On the slide

- Macro F1 on validation, public and private for Majority, Random (dummy) and Logistic regression
- Dummy: uniform random
- Trained: TF-IDF + logistic regression
- Hardest tone: Neutral

## What to say

> We compared two dummy baselines. Always predicting the majority tone scores 0.12, because it gets zero on four of the five tones. Random guessing scores 0.19 on validation, so the random one is our dummy benchmark.
>
> The trained baseline is TF-IDF on words and word pairs with logistic regression, using balanced class weights. It scores 0.339 public and 0.377 private, compared with 0.181 public and 0.139 private for the random dummy. Neutral is the hardest tone, with an F1 of only 0.13.

## Key facts

| Model | Validation | Public | Private | Full test (88) |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.122 | 0.125 | 0.121 | 0.123 |
| Random, seed 16 (dummy) | 0.193 | 0.181 | 0.139 | 0.160 |
| Logistic regression | 0.396 | 0.339 | 0.377 | 0.378 |

Logistic regression per-tone F1 on the full test set:

| Tone | F1 |
| --- | ---: |
| Appreciative or positive | 0.500 |
| Neutral or informational | 0.133 |
| Help-seeking or asking a question | 0.323 |
| Frustrated or complaining | 0.519 |
| Escalatory or requesting immediate resolution | 0.414 |

- Settings were chosen on validation only: C = 0.25, `class_weight="balanced"`. The submitted model was then refit on train plus validation.
- The random seed (16) was the one whose validation score was closest to the 20-seed mean. It was not picked using test scores.

## If asked

- **Why random over majority?** The assignment says to keep whichever dummy does better, and with macro F1 random wins.
- **Why is Neutral so hard?** It has no strong cue words, and flat statements about a late order look a lot like complaints.

Full method reference: [docs/baselines_method.md](../baselines_method.md)

[Previous](slide_09_how_it_works.md) | [Next](slide_11_llm_baseline.md)
