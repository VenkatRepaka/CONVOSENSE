# Slide 3: Annotation Process

**Time:** 0:25

**Goal:** Explain who labeled what, and which items were used to measure agreement.

## On the slide

- 7 annotators, 682 labels, 506 conversations, 30 shared
- Who labeled what, and the four-step pipeline

## What to say

> Seven annotators labeled our data in Potato. Each got 100 items: 30 shared and 70 of their own. Annotator 5 submitted 82 of 100. That is 682 labels on 506 conversations. The 30 shared items are where we measure agreement. The other 476 conversations have a single label, and that label becomes their ground truth.

## Key facts

- 26 shared items have all seven labels. 4 have six, because annotator 05 skipped them. Annotator 07 labeled all 30.
- 476 conversations have one annotator.
- Only the primary label counts. The optional secondary label is context, not a vote.
- Pipeline: Potato JSONL per annotator → label matrix (conversation × annotator) → agreement on the 30 shared items → ground truth for all 506.

## If asked

- **Isn't one label per conversation risky?** Yes, that's a limitation. Krippendorff's alpha was 0.433 on the 30 conversations labeled by multiple annotators. The other 476 conversations each had one label, so we couldn't measure agreement on those items. The 30 shared conversations were also picked to be clear-cut and to cover all five tones evenly, so agreement on the rest could be lower.
- **How are annotators identified?** By submission folder name, `annotator_01` to `annotator_07`, because the `user_id` inside the files is a display name for some people and an id for others.

[Previous](slide_02_task_and_data_point.md) | [Next](slide_04_agreement.md)
