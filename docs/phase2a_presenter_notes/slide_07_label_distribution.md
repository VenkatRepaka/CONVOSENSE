# Slide 7: The Classes Are Imbalanced

**Time:** 0:20

**Goal:** Show the label distribution and link it to the choice of metric.

## On the slide

- Bar chart of conversations per tone
- 42.9% Frustrated; Appreciative and Escalatory 11.3% each
- Chi-square test against an equal share: χ² = 175.7, p < 0.001

## What to say

> Here is the label distribution. Frustrated is 43% of the data, almost four times the smallest tones. A chi-square test against an equal share per tone rejects balance with p below 0.001. This is why we score the competition with macro F1 rather than accuracy. Otherwise a model could do well just by predicting Frustrated.

## Key facts

| Tone | Conversations | Share |
| --- | ---: | ---: |
| Appreciative or positive | 57 | 11.3% |
| Neutral or informational | 81 | 16.0% |
| Help-seeking or asking a question | 94 | 18.6% |
| Frustrated or complaining | 217 | 42.9% |
| Escalatory or requesting immediate resolution | 57 | 11.3% |

- An equal share would be 101.2 per tone.
- Majority-vote labels (504 conversations, two ties dropped) give the same result: χ² = 177.6, p < 0.001.

## If asked

- **Why is Frustrated so common?** People mostly tweet at a support account when something has gone wrong.

[Previous](slide_06_disagreements.md) | [Next](slide_08_competition_data.md)
