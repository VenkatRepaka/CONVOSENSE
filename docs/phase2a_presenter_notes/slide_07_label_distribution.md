# Slide 7: The Classes Are Imbalanced

**Time:** 0:20

**Goal:** Show the label distribution and link it to the choice of metric.

## On the slide

- Bar chart of conversations per tone
- 43.8% Frustrated, 11.5% Appreciative
- Chi-square test against an equal share: χ² = 164.0, p < 0.001

## What to say

> Here is the label distribution. Frustrated is 44% of the data, almost four times the smallest tone. A chi-square test against an equal share per tone rejects balance with p below 0.001. This is why we score the competition with macro F1 rather than accuracy. Otherwise a model could do well just by predicting Frustrated.

## Key facts

| Tone | Conversations | Share |
| --- | ---: | ---: |
| Appreciative or positive | 50 | 11.5% |
| Neutral or informational | 54 | 12.4% |
| Help-seeking or asking a question | 86 | 19.7% |
| Frustrated or complaining | 191 | 43.8% |
| Escalatory or requesting immediate resolution | 55 | 12.6% |

- An equal share would be 87.2 per tone.
- Majority-vote labels (434 conversations) give the same result: χ² = 166.3, p < 0.001.

## If asked

- **Why is Frustrated so common?** People mostly tweet at a support account when something has gone wrong.

[Previous](slide_06_disagreements.md) | [Next](slide_08_competition_data.md)
