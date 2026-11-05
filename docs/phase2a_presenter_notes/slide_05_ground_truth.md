# Slide 5: Ground Truth: Reliability-Weighted Vote

**Time:** 0:35

**Goal:** Explain how the final labels were computed and why we chose the weighted vote. This is one of the two things the assignment asks you to be ready to explain.

## On the slide

- Three method cards: Single label (476), Majority vote (28 decided, 2 ties), Weighted vote, used (30 decided, 0 ties)
- The two ties: 1100995 (Escalatory 1.318 vs. Frustrated 1.268) and 233771 (Neutral 1.423 vs. Help-seeking 1.294)
- A "Why this method" card with three reasons
- A bar chart of annotator weights

## What to say

> Ground truth comes from three rules. 476 conversations had one annotator, and their label is the ground truth. On the 30 shared conversations, majority vote decided 28 and left two exact ties, 3 votes against 3.
>
> The weighted vote fixes that. Each annotator's weight is their average Cohen's kappa with the others, and for each label we add up the weights of the people who chose it. In the first tie, Escalatory's three annotators add up to 1.32, against 1.27 for Frustrated, so Escalatory wins. In the second, Neutral wins 1.42 to 1.29.
>
> We chose the weighted vote because Kaggle needs a label on every row, and a fixed rule breaks the tie so none of us picked the answer. It changes only those two labels.

## Key facts

| Annotator | Weight |
| --- | ---: |
| A1 | 0.345 |
| A2 | 0.457 |
| A3 | 0.482 |
| A4 | 0.391 |
| A5 | 0.475 |
| A6 | 0.466 |
| A7 | 0.445 |

| Tie | Sides | Winner | Split |
| --- | --- | --- | --- |
| 1100995 | Escalatory: A3 + A4 + A7 = 1.318. Frustrated: A1 + A2 + A6 = 1.268. A5 skipped it | Escalatory or requesting immediate resolution | Test (private) |
| 233771 | Neutral: A3 + A5 + A6 = 1.423. Help-seeking: A2 + A4 + A7 = 1.294. Frustrated: A1 = 0.345 | Neutral or informational | Validation |

- Ties are never broken alphabetically.
- Annotator 7 broke both six-annotator ties: `1003675` is now Appreciative 4–3, and `1257160` is now Frustrated 4–3.
- On the other 28 shared conversations, majority vote and weighted vote pick the same label.
- `1100995` is one private test row whose answer depends on the tie-break.
- Files (kept locally, not in git): `outputs/ground_truth_weighted_vote.csv`, which is used for the competition, and `outputs/ground_truth_majority_vote.csv`.

## If asked

- **Why not adjudicate by hand?** We could, and the list of disagreements is ready (next slide). The weighted vote gives a label to every item without a team member choosing the outcome.
- **Aren't the weights computed from the same 30 items?** Yes. That's the only overlap we have, and the weights only matter on those 30 items.
- **Isn't trusting one annotator for 476 items risky?** It's a limitation. Krippendorff's alpha was 0.433 on the 30 conversations labeled by multiple annotators. The other 476 conversations each had one label, so we couldn't measure agreement on those items.
- **Do the weights change much?** They range from 0.35 to 0.48. That is too narrow to overturn a clear majority, so in practice the weighted vote is majority vote with a principled tie-breaker.

Full method reference: [docs/ground_truth_method.md](../ground_truth_method.md)

[Previous](slide_04_agreement.md) | [Next](slide_06_disagreements.md)
