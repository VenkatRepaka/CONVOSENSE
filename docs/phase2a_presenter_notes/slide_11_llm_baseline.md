# Slide 11: LLM Baseline: GPT-5 Zero-Shot

**Time:** 0:30

**Goal:** Present the frontier-LLM baseline and say why it is not on the leaderboard.

## On the slide

- GPT-5 (ChatGPT), zero-shot, 50 of the 101 test conversations
- Bar chart on the same 50 items: Majority 0.118, Random 0.260, Logistic regression 0.303, GPT-5 0.447
- Result card: 0.447 macro F1, 46% accuracy (23 of 50)
- Not on the leaderboard: it covers only 50 test items

## What to say

> For the LLM baseline we gave GPT-5 the five tone definitions and 50 test conversations, with no labeled examples. It got 23 of the 50 right, and averaged over the five tones it scores 0.45 macro F1, against 0.30 for our logistic regression on the same items. So a general-purpose LLM already beats our trained baseline without seeing any training data. It's good at spotting thanks, but it misses most help-seeking questions and labels too many complaints as Neutral or Escalatory. It stays off the leaderboard because the assignment only allows an LLM benchmark that covers the whole test set.

## Key facts

| Model, on the same 50 test items | Macro F1 |
| --- | ---: |
| Majority | 0.118 |
| Random, seed 3 | 0.260 |
| Logistic regression | 0.303 |
| GPT-5 zero-shot | 0.447 |

GPT-5 per tone:

| Tone | Really this tone | GPT-5 said this tone | Correct | F1 |
| --- | ---: | ---: | ---: | ---: |
| Appreciative | 7 | 8 | 5 | 0.67 |
| Escalatory | 7 | 12 | 5 | 0.53 |
| Frustrated | 21 | 15 | 9 | 0.50 |
| Help-seeking | 10 | 3 | 2 | 0.31 |
| Neutral | 5 | 12 | 2 | 0.24 |

- Accuracy: 23 of 50 correct (46%). Macro F1 is the plain average of the five tone F1 scores, so each tone counts the same.
- Always saying Frustrated would get 42% accuracy on these 50 but only 0.118 macro F1. That's why the competition uses macro F1.
- Files: `baselines/llm/prompt.txt`, `baselines/llm/items.jsonl`, `baselines/llm/chat_input.txt` (what was pasted), `baselines/llm/raw_responses.jsonl`, `baselines/llm/predictions.csv`, `baselines/llm_scores.csv`, `baselines/llm_sample_comparison.csv`, `baselines/llm/run.json`.

## If asked

- **What exactly was the prompt?** The five definitions with cue words, the precedence order, and a JSON output format. No labeled examples.
- **Is 50 items enough?** It's a rough estimate. One changed prediction can move a rare tone's F1 a lot. That's another reason it isn't a leaderboard benchmark.
- **Can it be reproduced?** Roughly. It was a manual ChatGPT run in a new chat, and temperature and model snapshot aren't visible there.

[Previous](slide_10_baselines.md) | [Next](slide_12_pitch.md)
