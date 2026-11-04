# Slide 6: Where Annotators Disagreed

**Time:** 0:20

**Goal:** Show that the disagreements are recorded, and which tones people confuse. The assignment asks you to keep this list.

## On the slide

- 25 of 30 shared conversations had at least one disagreement. 5 were unanimous.
- "Kept for later": `outputs/adjudication_disagreements.csv`
- A bar chart of the most-confused tone pairs (top 6)
- A takeaway card

## What to say

> 25 of our 30 shared conversations had at least one disagreement, and we kept the full list, with every annotator's label and a vote summary, for adjudication or error analysis later.
>
> Counting every pair of annotators who disagreed, the biggest confusion is Frustrated versus Escalatory, then Appreciative versus Help-seeking. Those are neighboring tones. Neutral also gets mixed up with Help-seeking and Frustrated, which helps explain why Neutral is the hardest tone for our models.

## Key facts

Annotator pairs that disagreed, by tone pair, across the 25 conversations:

| Tone pair | Disagreeing pairs |
| --- | ---: |
| Frustrated vs. Escalatory | 57 |
| Appreciative vs. Help-seeking | 38 |
| Help-seeking vs. Neutral | 29 |
| Frustrated vs. Neutral | 24 |
| Frustrated vs. Help-seeking | 18 |
| Appreciative vs. Neutral | 12 |

- How it's counted: in each conversation, every pair of annotators with different labels adds 1 to that tone pair. For example, a 3–3 split adds 9.
- The file has one row per conversation, with each annotator's label and a vote summary such as "Frustrated or complaining (3); Escalatory or requesting immediate resolution (3)". It's kept locally and not in git, because it includes test conversations.

## If asked

- **What will you use the list for?** Adjudicating if we change the ground truth, checking the guidelines (for example, a clearer Frustrated vs. Escalatory rule), and error analysis on model mistakes.
- **Are those 25 conversations in the test set?** Some are. That's why the list stays out of the public repository.

[Previous](slide_05_ground_truth.md) | [Next](slide_07_label_distribution.md)
