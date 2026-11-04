# Slide 5: Ground Truth: Reliability-Weighted Vote

**Time:** 0:35

**Goal:** Explain how the final labels were computed and why we chose the weighted vote. This is one of the two things the assignment asks you to be ready to explain.

## On the slide

- Three method cards: Single label (406), Majority vote (28 decided, 2 ties), Weighted vote, used (30 decided, 0 ties)
- A worked tie: conversation 1003675, Appreciative 1.419 vs. Help-seeking 1.186
- The other tie, 1257160: Escalatory 1.356 vs. Frustrated 1.249
- A "Why this method" card with three reasons
- A bar chart of annotator weights

## What to say

> Ground truth comes from three rules. 406 conversations had one annotator, and their label is the ground truth. That's 93% of the data. On the 30 shared conversations, majority vote decided 28 and left two exact ties, 3 votes against 3.
>
> The weighted vote fixes that. Each annotator's weight is their average Cohen's kappa with the others, and for each label we add up the weights of the people who chose it. In this tie, the three annotators who chose Appreciative add up to 1.42, against 1.19 for Help-seeking, so Appreciative wins.
>
> We chose it for three reasons. Kaggle needs a label on every row. A fixed rule breaks the ties, so none of us picked the answer. And it only changes two labels, both in validation. The test answer key is the same either way, though our baselines are tuned on validation, so their scores could shift slightly.

## Key facts

| Annotator | Weight |
| --- | ---: |
| A1 | 0.348 |
| A2 | 0.444 |
| A3 | 0.483 |
| A4 | 0.394 |
| A5 | 0.479 |
| A6 | 0.457 |

| Tie | Side 1 | Side 2 | Winner |
| --- | --- | --- | --- |
| 1003675 | Appreciative: A3 + A5 + A6 = 1.419 | Help-seeking: A1 + A2 + A4 = 1.186 | Appreciative or positive |
| 1257160 | Escalatory: A3 + A4 + A5 = 1.356 | Frustrated: A1 + A2 + A6 = 1.249 | Escalatory or requesting immediate resolution |

- Ties are never broken alphabetically.
- On the other 28 shared conversations, majority vote and weighted vote pick the same label.
- Files (kept locally, not in git): `outputs/ground_truth_weighted_vote.csv`, which is used for the competition, and `outputs/ground_truth_majority_vote.csv`.

## If asked

- **Why not adjudicate by hand?** We could, and the list of disagreements is ready (next slide). The weighted vote gives a label to every item without a team member choosing the outcome.
- **Aren't the weights computed from the same 30 items?** Yes. That's the only overlap we have, and the weights only matter on those 30 items.
- **Isn't trusting one annotator for 406 items risky?** It's a limitation. Krippendorff's alpha was 0.427 on the 30 conversations labeled by multiple annotators. The other 406 conversations each had one label, so we couldn't measure agreement on those items. The shared 30 were chosen as clear-cut cases, so agreement on the single-label conversations could be lower.
- **Do the weights change much?** No. They range from 0.35 to 0.48, so in practice they act as a principled tie-breaker.
- **Would majority vote give the same leaderboard?** Not exactly. With the current split, the test answer key is identical, so any fixed submission scores the same. But the baselines pick their settings on validation and are refit on train plus validation. Giving the two ties the other label moved the logistic regression to 0.357 public / 0.348 private (from 0.339 / 0.377). Building the split from majority-vote labels from the start would also have produced a different test set.

Full method reference: [docs/ground_truth_method.md](../ground_truth_method.md)

[Previous](slide_04_agreement.md) | [Next](slide_06_disagreements.md)
