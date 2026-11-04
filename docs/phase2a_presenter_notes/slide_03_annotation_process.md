# Slide 3: Annotation Process

**Time:** 0:25

**Goal:** Explain who labeled what, and which items were used to measure agreement.

## On the slide

- 6 annotators, 582 labels, 436 conversations, 30 shared
- Who labeled what, and the four-step pipeline

## What to say

> Six classmates labeled our data in Potato. Each got 100 items: 30 shared with everyone and 70 of their own. One annotator submitted 82 of 100, so we have 582 labels on 436 conversations. The 30 shared items are where we measure agreement. The other 406 conversations have a single label, and that label becomes their ground truth.

## Key facts

- 26 shared items have all six labels. 4 have five, because annotator 05 skipped them.
- 406 conversations have one annotator.
- Only the primary label counts. The optional secondary label is context, not a vote.
- Pipeline: Potato JSONL per annotator → label matrix (conversation × annotator) → agreement on the 30 shared items → ground truth for all 436.

## If asked

- **Isn't one label per conversation risky?** Yes. That is a limitation: agreement is measured on 30 items, and the other 406 labels are not cross-checked. The agreement score is our estimate of how reliable those single labels are.
- **Why did the annotator IDs change?** We use folder names (`annotator_01` to `annotator_06`) because the `user_id` inside some files was a display name.

[Previous](slide_02_task_and_data_point.md) | [Next](slide_04_agreement.md)
