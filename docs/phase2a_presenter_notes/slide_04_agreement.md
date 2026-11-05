# Slide 4: Krippendorff's Alpha: Moderate Agreement (0.43)

**Time:** 0:30

**Goal:** Give the agreement score, say why alpha is the right metric, and point out the best and worst pairs.

## On the slide

- Bar chart: Krippendorff's alpha 0.433 (main), Cohen's kappa 0.437, Fleiss' kappa 0.443, percentage agreement 0.572
- Heatmap of pairwise Cohen's kappa
- Highest pair 0.75 (A5 and A6), lowest pair 0.26 (A1 and A5)

## What to say

> Our main agreement metric is Krippendorff's alpha, which is 0.43: moderate agreement. We chose alpha because it works with more than two annotators, with categories that have no order, and with missing labels. Annotator 5 skipped four of the shared items, and alpha keeps those. Fleiss' kappa would have to drop them.
>
> The other chance-corrected scores are within about 0.01 of alpha, so the conclusion doesn't depend on the metric. Pairs range from 0.26 to 0.75. Tone is subjective, especially between Frustrated and Escalatory, and between Appreciative and Help-seeking.

## Key facts

| Metric | Score | Scored on |
| --- | ---: | --- |
| Krippendorff's alpha (main) | 0.433 | All 30 shared items; missing labels kept |
| Cohen's kappa | 0.437 | Mean of 21 annotator pairs, 26 or 30 items each |
| Fleiss' kappa | 0.443 | 26 items labeled by all seven |
| Percentage agreement | 0.572 | Mean of the same 21 pairs |

- Libraries: scikit-learn (Cohen's kappa, percentage agreement), statsmodels (Fleiss' kappa), simpledorff (alpha)
- Percentage agreement is only reported next to the chance-corrected scores, as the assignment asks.
- Moderate is 0.41–0.60 on the Landis–Koch scale. Six-annotator alpha was 0.427, so adding annotator 7 barely moved it.
- Annotator 07's kappa with each of the others ranges from 0.33 to 0.53.

## If asked

- **Why not percentage agreement?** It doesn't subtract the agreement you'd get by chance. With one tone making up 43% of the data, chance agreement is high.
- **Is 0.43 good enough?** It is moderate and typical for subjective tone labels. Across the 27 shared items with disagreement, the most common mismatched annotator pairs are Frustrated vs. Escalatory (75 pairs) and Appreciative vs. Help-seeking (48).

[Previous](slide_03_annotation_process.md) | [Next](slide_05_ground_truth.md)
