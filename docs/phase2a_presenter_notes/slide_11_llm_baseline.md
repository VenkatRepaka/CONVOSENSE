# Slide 11: LLM Baseline: Zero-Shot on 50 Test Items

**Time:** 0:30

**Goal:** Report the frontier-LLM result: the model, the prompt, how the items were sent, and the score.

> **Update before presenting.** These numbers are from the first run, which can't be verified. Redo it in a fresh chat: paste `baselines/llm/prompt.txt` with 10 items per message, start a new chat for each batch of 10, and do 5 batches. Save the labels to `baselines/llm/predictions.csv`, then run `python src/baselines/llm_baseline.py score baselines/llm/predictions.csv`. Update the model name, the score and the per-tone line here and on the slide.

## On the slide

- Macro F1 on the same 50 test conversations: Majority 0.12, Random 0.14, Logistic regression 0.30, LLM 0.40
- Model, Prompt, Items and Result cards

## What to say

> In our proposal we argued the task isn't trivial for a frontier LLM. We gave [MODEL] 50 test conversations zero-shot: just the five tone definitions and the tie-break order, with no labeled examples. It scored [0.40] macro F1. That's ahead of logistic regression on the same items, but far from solved. It handles Appreciative well and struggles with Neutral and Frustrated, the same tones our baseline finds hard. It isn't on the leaderboard because it covers 50 of the 88 test items.

## Key facts (first run)

| Model | Macro F1 on the same 50 |
| --- | ---: |
| Majority | 0.118 |
| Random, seed 16 | 0.142 |
| Logistic regression | 0.300 |
| LLM zero-shot | 0.404 |

LLM per-tone F1: Appreciative 0.82, Escalatory 0.45, Help-seeking 0.31, Frustrated 0.30, Neutral 0.13.

- The 50 items were drawn from the test set with seed 42. Labels were never shown to the model.
- None of the annotation-guideline examples are in the test set.

## If asked

- **What exactly was the prompt?** It's in `baselines/llm/prompt.txt`: the five definitions with cue words, the precedence order, and a JSON output format.
- **Why only 50 items?** The assignment requires at least 50. Running all 88 would let us put the LLM on the leaderboard.

[Previous](slide_10_baselines.md) | [Next](slide_12_pitch.md)
