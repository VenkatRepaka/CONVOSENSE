# Slide 8: Competition Data: Stratified Split

**Time:** 0:20

**Goal:** Explain how the data was split and which files participants get.

## On the slide

- Table of tones by split: Train 261, Validation 87, Test 88, Public 44, Private 44
- Files: `train.csv`, `validation.csv`, `test.csv`, `sample_submission.csv`
- `solution.csv` is only the hidden answer key

## What to say

> We split the data 60/20/20, stratified by tone, so every tone appears everywhere. The test set is split again in half into public and private leaderboards, and even the smallest tone has five examples on each one. You get a training file and a separate validation file to compare models, and the test file has no labels. The answer key is uploaded only to Kaggle.

## Key facts

| Tone | Train | Validation | Test | Public | Private |
| --- | ---: | ---: | ---: | ---: | ---: |
| Appreciative | 30 | 10 | 10 | 5 | 5 |
| Neutral | 32 | 11 | 11 | 6 | 5 |
| Help-seeking | 52 | 17 | 17 | 8 | 9 |
| Frustrated | 114 | 38 | 39 | 20 | 19 |
| Escalatory | 33 | 11 | 11 | 5 | 6 |
| **Total** | **261** | **87** | **88** | **44** | **44** |

- Split with scikit-learn `train_test_split`, seed 42
- Columns in `train.csv` and `validation.csv`: `id`, `conversation`, `last_customer_message`, `label`. `test.csv` has no `label`.

## If asked

- **Can we train on validation too?** Yes. Our logistic regression benchmark was refit on train plus validation.
- **Is the test set big enough?** It is small (88 items), which is why each leaderboard half is stratified. Expect some noise between public and private scores.

[Previous](slide_07_label_distribution.md) | [Next](slide_09_how_it_works.md)
