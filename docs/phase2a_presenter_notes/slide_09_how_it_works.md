# Slide 9: How the Competition Works

**Time:** 0:30

**Goal:** Show how to submit, how scoring works, and the rules. Switch to the Kaggle page here if there is time.

## On the slide

- Submit: one row per test id with `id,label`. The label is the tone number, 1 to 5 (`label_map.csv`).
- Scored by macro F1
- Leaderboards and rules
- An example `sample_submission.csv`

## What to say

> To submit, upload a CSV with one row per test id: the id and the tone's number, 1 to 5. The numbers follow the order of the five tones, and `label_map.csv` lists them. Scoring is macro F1. That's the F1 for each tone, averaged equally, so always guessing Frustrated scores only 0.12.
>
> The public leaderboard uses half the test set and is visible now. The private half decides the final ranking at the Phase 2b deadline. Predictions must come from a model, and labeling the test set by hand isn't allowed. The competition is private, and you join with the invite link.

## Key facts

- Macro F1 = the unweighted mean of the five per-tone F1 scores, where F1 = 2 · precision · recall / (precision + recall).
- Our perfect submission scores 1.0 on both leaderboards. The sample submission scores 0.118 public and 0.121 private.
- Data license: CC BY-NC-SA 4.0, for ARI 410/510 only.
- Set your Kaggle team name to include your project team's name.

## If asked

- **Can we use pretrained models or outside data?** Yes, pretrained models and public text are allowed. Using any tone label for a test conversation from outside the competition files is not.
- **How many submissions per day?** Whatever limit is set on the Kaggle page. Check it there.

[Previous](slide_08_competition_data.md) | [Next](slide_10_baselines.md)
