# Slide 4: Krippendorff's Alpha: Moderate Agreement (0.43)

**Time:** 0:30

**Goal:** Give the agreement score, say why alpha is the right metric, and point out the best and worst pairs.

## On the slide

- Bar chart: Krippendorff's alpha 0.427 (main), Cohen's kappa 0.434, Fleiss' kappa 0.447, percentage agreement 0.566
- Heatmap of pairwise Cohen's kappa
- Highest pair 0.75 (A5 and A6), lowest pair 0.26 (A1 and A5)

## What to say

> Our main agreement metric is Krippendorff's alpha, which is 0.43: moderate agreement. We chose alpha because it works with more than two annotators, with categories that have no order, and with missing labels. Annotator 5 skipped four of the shared items, and alpha keeps those. Fleiss' kappa would have to drop them.
>
> The other chance-corrected scores are within 0.02 of alpha, so the conclusion doesn't depend on the metric. Pairs range from 0.26 to 0.75. Tone is subjective, especially between Frustrated and Escalatory.

## Key facts

| Metric | Score | Scored on |
| --- | ---: | --- |
| Krippendorff's alpha (main) | 0.427 | All 30 shared items; missing labels kept |
| Cohen's kappa | 0.434 | Mean of 15 annotator pairs, 26 or 30 items each |
| Fleiss' kappa | 0.447 | 26 items labeled by all six |
| Percentage agreement | 0.566 | Mean of the same 15 pairs |

- Libraries: scikit-learn (Cohen's kappa, percentage agreement), statsmodels (Fleiss' kappa), simpledorff (alpha)
- Percentage agreement is only reported next to the chance-corrected scores, as the assignment asks.

## If asked

- **Why not percentage agreement?** It doesn't subtract the agreement you'd get by chance. With one tone making up 44% of the data, chance agreement is high.
- **Is 0.43 good enough?** It is moderate and typical for subjective tone labels. The disagreements are mostly between neighboring tones. Across the 25 shared items with disagreement, the most common mismatched annotator pairs are Frustrated vs. Escalatory (57 pairs) and Appreciative vs. Help-seeking (38).

[Previous](slide_03_annotation_process.md) | [Next](slide_05_ground_truth.md)
