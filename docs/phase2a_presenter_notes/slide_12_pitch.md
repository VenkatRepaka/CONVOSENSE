# Slide 12: Beat the 0.398 Benchmark

**Time:** 0:15

**Goal:** Close with a reason for other teams to join, and give them the number they will actually see on the leaderboard.

## On the slide

- Beat the 0.398 Benchmark
- Our logistic regression scores 0.398 macro F1 on the public leaderboard (0.463 on the private one that decides the final ranking).
- Small data (405 labeled conversations), five subtle tones, and sarcasm that needs context
- Can your model find the Neutral customers? Join through the Kaggle invite link on Discord.

## What to say

> Our logistic regression benchmark scores 0.398 on the public leaderboard, and there's lots of room above it: GPT-5 zero-shot reached 0.45 on half the test set. The final ranking is on the private leaderboard, where the same benchmark scores 0.463. You get 405 labeled conversations, five subtle tones, and sarcasm that only makes sense in context. The open challenge is the Neutral class, where our baseline scores 0.17. Can your model find those customers? The invite link is in #project-competitions on Discord. Thank you!

## Key facts

- **0.398** is the logistic regression Benchmark on the public leaderboard: the number classmates compare against during the competition.
- **0.463** is the same Benchmark on the private leaderboard, which decides the final ranking at the Phase 2b deadline.
- 0.428 is the full-test score (all 101 rows). Kaggle never shows it, so don't quote it to classmates.
- 405 = 302 train + 103 validation
- Ideas to suggest if asked: pretrained sentence embeddings with logistic regression, fine-tuning a small transformer, or using the full conversation as context.

## If asked

- **Why are public and private different?** Each half has only 50 or 51 conversations, and a rare tone has 5 or 6 rows per half, so one prediction can move a tone's F1 a lot.

[Previous](slide_11_llm_baseline.md) | [Back to index](README.md)
