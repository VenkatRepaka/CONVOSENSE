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

- **Isn't one label per conversation risky?** Yes, that's a limitation. Krippendorff's alpha was 0.427 on the 30 conversations labeled by multiple annotators. The other 406 conversations each had one label, so we couldn't measure agreement on those items. The 30 shared conversations were also picked to be clear-cut and to cover all five tones evenly, so agreement on the rest could be lower.
- **Why did the annotator IDs change?** We use folder names (`annotator_01` to `annotator_06`) because the `user_id` inside some files was a display name.

[Previous](slide_02_task_and_data_point.md) | [Next](slide_04_agreement.md)
