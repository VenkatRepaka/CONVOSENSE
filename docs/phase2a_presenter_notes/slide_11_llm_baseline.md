# Slide 11: LLM Baseline: Zero-Shot on 50 Test Items

**Time:** 0:30

**Goal:** Report the frontier-LLM result: the model, the prompt, how the items were sent, and the score.

> The slide, its chart and its speaker notes in `docs/phase2a.pptx` match these numbers (GPT-5 rerun, 4 Nov 2026).

## On the slide

- Macro F1 on the same 50 test conversations: Majority 0.12, Random 0.14, Logistic regression 0.30, GPT-5 0.44
- Model, Prompt, Items and Result cards

## What to say

> In our proposal we argued the task isn't trivial for a frontier LLM. We gave GPT-5 50 test conversations zero-shot: just the five tone definitions and the tie-break order, with no labeled examples. It scored 0.44 macro F1. That's ahead of logistic regression on the same items, but far from solved. It handles Appreciative well and struggles with Neutral and Help-seeking, tones our baseline also finds hard. It isn't on the leaderboard because it covers 50 of the 88 test items.

## Key facts (GPT-5, ChatGPT, 4 Nov 2026)

| Model | Macro F1 on the same 50 |
| --- | ---: |
| Majority | 0.118 |
| Random, seed 16 | 0.14 |
| Logistic regression | 0.300 |
| GPT-5 zero-shot | 0.441 |

GPT-5 per-tone F1: Appreciative 0.67, Frustrated 0.53, Escalatory 0.47, Help-seeking 0.31, Neutral 0.24.

- The 50 items were drawn from the test set with seed 42. Labels were never shown to the model.
- None of the annotation-guideline examples are in the test set.
- Sent by hand in one new ChatGPT conversation: the prompt and all 50 items in one message. The raw reply is in `baselines/llm/raw_responses.jsonl`.

## If asked

- **What exactly was the prompt?** It's in `baselines/llm/prompt.txt`: the five definitions with cue words, the precedence order, and a JSON output format.
- **Why ChatGPT by hand, not an API?** We had no API key. The raw reply is saved, but a rerun may differ because the chat interface doesn't expose temperature or the model snapshot.
- **Why only 50 items?** The assignment requires at least 50. Running all 88 would let us put the LLM on the leaderboard.

[Previous](slide_10_baselines.md) | [Next](slide_12_pitch.md)
