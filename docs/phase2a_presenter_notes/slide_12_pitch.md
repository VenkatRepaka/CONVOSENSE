# Slide 12: Beat the 0.339 Benchmark

**Time:** 0:15

**Goal:** Close with a reason for other teams to join, and give them the number they will actually see on the leaderboard.

## On the slide

- Beat the 0.339 Benchmark
- Our logistic regression scores 0.339 macro F1 on the public leaderboard (0.377 on the private one that decides the final ranking).
- Small data (348 labeled conversations), five subtle tones, and sarcasm that needs context
- Can your model find the Neutral customers? Join through the Kaggle invite link on Discord.

## What to say

> Our logistic regression benchmark scores 0.339 on the public leaderboard, and there's lots of room above it. The final ranking is on the private leaderboard, where the same benchmark scores 0.377. You get 348 labeled conversations, five subtle tones, and sarcasm that only makes sense in context. The open challenge is the Neutral class. Can your model find those customers? The invite link is in #project-competitions on Discord. Thank you!

## Key facts

- **0.339** is the logistic regression Benchmark on the public leaderboard: the number classmates compare against during the competition.
- **0.377** is the same Benchmark on the private leaderboard, which decides the final ranking at the Phase 2b deadline.
- 0.378 is the full-test score (all 88 rows). Earlier drafts rounded it to 0.38, but Kaggle never shows it, so don't quote it to classmates.
- 348 = 261 train + 87 validation
- Ideas to suggest if asked: pretrained sentence embeddings with logistic regression, fine-tuning a small transformer, or using the full conversation as context.

## If asked

- **Why are public and private different?** Each half has only 44 conversations, and a rare tone has 5 or 6 rows per half, so one prediction can move a tone's F1 a lot.
- **What's realistic?** On the 30 shared conversations, annotators averaged 0.537 macro F1 against the weighted consensus of the other annotators. Logistic regression was evaluated on a different set of 88 test conversations, so these scores aren't a direct human–model comparison. It shows the labels are subjective, not a limit on models. For a same-items comparison: on 50 test conversations, GPT-5 zero-shot scores 0.44 and logistic regression 0.30. (See `docs/baselines_method.md`, section 9.)

[Previous](slide_11_llm_baseline.md) | [Back to index](README.md)
