# Slide 8: Competition Data: Stratified Split

**Time:** 0:20

**Goal:** Explain how the data was split and which files participants get.

## On the slide

- Table of tones by split: Train 302, Validation 103, Test 101, Public 50, Private 51
- Files: `train.csv`, `validation.csv`, `test.csv`, `sample_submission.csv`
- `solution.csv` is only the hidden answer key

## What to say

> We split the data about 60/20/20 by tone, so every tone appears everywhere. The test set is split again in half into public and private leaderboards, and even the smallest tone has at least five examples on each one. You get a training file and a separate validation file to compare models, and the test file has no labels. The answer key is uploaded only to Kaggle.

## Key facts

| Tone | Train | Validation | Test | Public | Private |
| --- | ---: | ---: | ---: | ---: | ---: |
| Appreciative | 34 | 12 | 11 | 6 | 5 |
| Neutral | 48 | 17 | 16 | 8 | 8 |
| Help-seeking | 56 | 19 | 19 | 9 | 10 |
| Frustrated | 130 | 44 | 43 | 21 | 22 |
| Escalatory | 34 | 11 | 12 | 6 | 6 |
| **Total** | **302** | **103** | **101** | **50** | **51** |

- The split is frozen in `competition/split_assignments.csv` (ids only). The 436 six-annotator conversations kept their part, and annotator 7's 70 new conversations were split tone by tone, seed 42. So no conversation whose label was already published for training moved into test.
- Public/Private: scikit-learn `train_test_split`, stratified, seed 42
- Columns in `train.csv` and `validation.csv`: `id`, `conversation`, `last_customer_message`, `label`. `test.csv` has no `label`. On Kaggle, `label` is the tone number 1 to 5, and `label_map.csv` gives the names.

## If asked

- **Can we train on validation too?** Yes. Our logistic regression benchmark was refit on train plus validation.
- **Is the test set big enough?** It is small (101 items), which is why each leaderboard half is stratified. Expect some noise between public and private scores.

[Previous](slide_07_label_distribution.md) | [Next](slide_09_how_it_works.md)
