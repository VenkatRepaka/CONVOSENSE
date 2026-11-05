# Slide 11: LLM Baseline: GPT-5 Zero-Shot

**Time:** 0:30

**Goal:** Present the frontier-LLM baseline and say why it is not on the leaderboard.

## On the slide

- GPT-5 (ChatGPT), zero-shot, 50 of the 101 test conversations
- Bar chart on the same 50 items: Majority 0.118, Random 0.260, Logistic regression 0.303, GPT-5 0.441
- Not on the leaderboard: it covers only 50 test items

## What to say

> For the LLM baseline we gave GPT-5 the five tone definitions and 50 test conversations, with no labeled examples. On those 50 it scores 0.44 macro F1, against 0.30 for our logistic regression on the same items. So a general-purpose LLM already beats our trained baseline without seeing any training data. It stays off the leaderboard because the assignment only allows an LLM benchmark that covers the whole test set.

## Key facts

| Model, on the same 50 test items | Macro F1 |
| --- | ---: |
| Majority | 0.118 |
| Random, seed 3 | 0.260 |
| Logistic regression | 0.303 |
| GPT-5 zero-shot | 0.441 |

- GPT-5 per-tone F1 on the 50: Appreciative 0.667, Neutral 0.235, Help-seeking 0.308, Frustrated 0.526, Escalatory 0.471.
- The 50 items were drawn (seed 42) from the six-annotator test set. The frozen split keeps all of them in test, and their labels did not change with annotator 7, so the score is the same 0.441.
- Files: `baselines/llm/prompt.txt`, `baselines/llm/items.jsonl`, `baselines/llm/raw_responses.jsonl`, `baselines/llm/predictions.csv`, `baselines/llm_scores.csv`, `baselines/llm_sample_comparison.csv`.

## If asked

- **What exactly was the prompt?** The five definitions with cue words, the precedence order, and a JSON output format. No labeled examples.
- **Is 50 items enough?** It's a rough estimate. One changed prediction can move a rare tone's F1 a lot. That's another reason it isn't a leaderboard benchmark.
- **Can it be reproduced?** Roughly. It was a manual ChatGPT run, and temperature and model snapshot aren't visible there.

[Previous](slide_10_baselines.md) | [Next](slide_12_pitch.md)
