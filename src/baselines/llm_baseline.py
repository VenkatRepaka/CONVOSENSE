#!/usr/bin/env python3
"""Prepare and score the zero-shot LLM baseline.

Step 6 asks for a frontier model on at least 50 test conversations, with no
labeled examples in the prompt. ``write`` samples 50 test ids and saves the
prompt plus the conversation text. ``score`` reads a prediction file and
reports macro F1 with the competition metric. The answer key stays in
``competition/private/`` and is not copied into the LLM files.

Run from the repository root::

    python src/baselines/llm_baseline.py write
    python src/baselines/llm_baseline.py score baselines/llm/predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "competition"))
from macro_f1 import LABEL_COLUMN, TONE_LABELS, score  # noqa: E402

PARTICIPANT_DIR = REPO_ROOT / "competition" / "participant"
PRIVATE_DIR = REPO_ROOT / "competition" / "private"
LLM_DIR = REPO_ROOT / "baselines" / "llm"
ID_COLUMN = "id"
SAMPLE_SIZE = 50
SAMPLE_SEED = 42

# Worked examples in Customer Tone Annotation Guidelines.md. They are left
# out of the 50 so the scored items are not those demonstrations.
GUIDELINE_EXAMPLE_IDS = (
    "1024230",
    "2473401",
    "166430",
    "2922471",
    "379709",
    "873728",
    "785097",
    "1273449",
    "1532184",
)

PROMPT = """\
You label the customer's tone in the last customer message of one AmazonHelp Twitter support conversation. Earlier turns are context. Judge the customer's words, not the agent's tone and not whether the problem was solved.

Pick exactly one of these five labels. Copy the label text exactly.

1. Appreciative or positive. The customer is pleased, grateful, or praising. A bare polite thanks, such as "Thanks, I'll try that", is not this label. Sarcastic thanks are not this label.
2. Neutral or informational. The customer states facts or answers a question, with no visible emotion, no question, and no demand. A late order described flatly is this label.
3. Help-seeking or asking a question. The customer wants an answer, instructions, or help, in a calm tone. An angry or rhetorical question is not this label.
4. Frustrated or complaining. The customer is unhappy or critical, and is not pressing, threatening, or accusing. Cues include sarcasm, evaluative words, capitals for emphasis, and "still waiting".
5. Escalatory or requesting immediate resolution. The customer presses for action or raises the stakes: a demand, a threat, an accusation, or anger that has clearly intensified. Cues include asking for a manager, "ASAP", "lawsuit", "cancel my Prime", "fraud", or "liar".

When the last message shows more than one tone, use the first label that applies:
1. Escalatory, if there is a demand, a threat, an accusation, or anger that has intensified.
2. Frustrated, if there is negative emotion or a complaint and no demand or threat.
3. Help-seeking, if the customer asks a question or requests help and shows no negative emotion.
4. Appreciative, if the thanks or praise is genuine.
5. Neutral, if none of the above applies.

The last customer message decides the label. Use earlier turns when that message is short, sarcastic, or refers back to the problem. Ignore the agent's tone except as context for the customer's reaction.

Reply with one JSON object per item and no other text. Each object has the item id and one label:
{"id": "<id>", "label": "<one of the five labels>"}
"""


def main(argv: list[str] | None = None) -> None:
    """Write the 50-item pack, or score a prediction file against it."""
    parser = argparse.ArgumentParser(description="Prepare or score the LLM baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    write_parser = subparsers.add_parser("write", help="Write the prompt and 50 test items.")
    write_parser.add_argument("--participant-dir", type=Path, default=PARTICIPANT_DIR)
    write_parser.add_argument("--output-dir", type=Path, default=LLM_DIR)
    score_parser = subparsers.add_parser("score", help="Score predictions for those 50 items.")
    score_parser.add_argument("predictions", type=Path)
    score_parser.add_argument("--participant-dir", type=Path, default=PARTICIPANT_DIR)
    score_parser.add_argument("--private-dir", type=Path, default=PRIVATE_DIR)
    score_parser.add_argument("--llm-dir", type=Path, default=LLM_DIR)
    score_parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "baselines" / "llm_scores.csv",
    )
    args = parser.parse_args(argv)
    if args.command == "write":
        _write_pack(args.participant_dir, args.output_dir)
        return
    _score_file(args.predictions, args.participant_dir, args.private_dir, args.llm_dir, args.results)


def _write_pack(participant_dir: Path, output_dir: Path) -> None:
    test = pd.read_csv(participant_dir / "test.csv", dtype=str, keep_default_na=False)
    chosen = _sample_ids(test[ID_COLUMN])
    items = test[test[ID_COLUMN].isin(chosen)].copy()
    items["_sort_id"] = pd.to_numeric(items[ID_COLUMN])
    items = items.sort_values("_sort_id", kind="mergesort").drop(columns="_sort_id")
    if len(items) != SAMPLE_SIZE:
        raise ValueError(f"Expected {SAMPLE_SIZE} items, got {len(items)}.")
    if LABEL_COLUMN in items.columns:
        raise ValueError("test.csv contains labels.")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    item_path = output_dir / "items.jsonl"
    with item_path.open("w", encoding="utf-8") as sink:
        for record in items.to_dict(orient="records"):
            if LABEL_COLUMN in record:
                raise ValueError("An item record contains a label.")
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    held_out = sorted(set(GUIDELINE_EXAMPLE_IDS) & set(test[ID_COLUMN]))
    (output_dir / "sample.json").write_text(
        json.dumps(
            {
                "n_items": SAMPLE_SIZE,
                "seed": SAMPLE_SEED,
                "excluded_guideline_example_ids_in_test": held_out,
                "prompt_file": "prompt.txt",
                "items_file": "items.jsonl",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {SAMPLE_SIZE} items to {item_path}")
    print(f"Guideline example ids held out of the test file: {held_out or 'none'}")


def _sample_ids(ids: pd.Series) -> list[str]:
    eligible = [item_id for item_id in ids.tolist() if item_id not in set(GUIDELINE_EXAMPLE_IDS)]
    if len(eligible) < SAMPLE_SIZE:
        raise ValueError(f"Only {len(eligible)} test ids are eligible for the LLM sample.")
    generator = np.random.default_rng(SAMPLE_SEED)
    chosen = generator.choice(np.array(eligible), size=SAMPLE_SIZE, replace=False)
    return [str(item_id) for item_id in chosen]


def _score_file(
    predictions_path: Path,
    participant_dir: Path,
    private_dir: Path,
    llm_dir: Path,
    results_path: Path,
) -> None:
    items = _read_items(llm_dir / "items.jsonl")
    predicted = pd.read_csv(predictions_path, dtype=str, keep_default_na=False)
    if list(predicted.columns) != [ID_COLUMN, LABEL_COLUMN]:
        raise ValueError(f"Predictions columns are {list(predicted.columns)}.")
    if set(predicted[ID_COLUMN]) != set(items[ID_COLUMN]):
        raise ValueError("Prediction ids do not match the 50 sampled items.")
    if predicted[ID_COLUMN].duplicated().any():
        raise ValueError("Prediction ids are not unique.")
    unknown = set(predicted[LABEL_COLUMN]) - set(TONE_LABELS)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")

    solution = pd.read_csv(private_dir / "solution.csv", dtype=str, keep_default_na=False)
    test = pd.read_csv(participant_dir / "test.csv", dtype=str, keep_default_na=False)
    if not set(items[ID_COLUMN]).issubset(set(test[ID_COLUMN])):
        raise ValueError("The LLM sample is not a subset of test.csv.")
    key = solution[solution[ID_COLUMN].isin(items[ID_COLUMN])].copy()
    aligned = key[[ID_COLUMN]].merge(predicted, on=ID_COLUMN, how="left")
    macro_f1 = float(score(key, aligned, ID_COLUMN))

    from sklearn.metrics import f1_score

    y_true = key[LABEL_COLUMN].to_numpy()
    y_pred = aligned[LABEL_COLUMN].to_numpy()
    present = [label for label in TONE_LABELS if (y_true == label).any()]
    per_class = f1_score(y_true, y_pred, average=None, labels=present, zero_division=0)
    rows = [
        {
            "model": "llm_zero_shot",
            "split": "llm_sample_50",
            "label": "macro",
            "f1": macro_f1,
        }
    ]
    rows.extend(
        {
            "model": "llm_zero_shot",
            "split": "llm_sample_50",
            "label": label,
            "f1": float(value),
        }
        for label, value in zip(present, per_class)
    )
    results = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False, lineterminator="\n")
    comparison = _score_same_items(key, predictions_path)
    comparison_path = results_path.parent / "llm_sample_comparison.csv"
    comparison.to_csv(comparison_path, index=False, lineterminator="\n")
    print(f"LLM zero-shot macro F1 on {len(key)} test items: {macro_f1:.4f}")
    print(comparison.round(4).to_string(index=False))
    print(f"Same-item comparison: {comparison_path}")
    print(results.round(4).to_string(index=False))
    print(f"Scores: {results_path}")


def _score_same_items(key: pd.DataFrame, llm_predictions: Path) -> pd.DataFrame:
    """Macro F1 of each baseline on the same 50 conversations."""
    paths = {
        "majority": REPO_ROOT / "competition" / "submissions" / "majority_baseline.csv",
        "random": REPO_ROOT / "competition" / "submissions" / "dummy_baseline.csv",
        "logistic_regression": REPO_ROOT / "competition" / "submissions" / "logistic_regression.csv",
        "llm_zero_shot": llm_predictions,
    }
    rows = []
    for model_name, path in paths.items():
        if not path.is_file():
            continue
        predicted = pd.read_csv(path, dtype=str, keep_default_na=False)
        aligned = key[[ID_COLUMN]].merge(predicted, on=ID_COLUMN, how="left")
        rows.append(
            {
                "model": model_name,
                "split": "llm_sample_50",
                "macro_f1": float(score(key, aligned, ID_COLUMN)),
            }
        )
    return pd.DataFrame(rows)


def _read_items(path: Path) -> pd.DataFrame:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if LABEL_COLUMN in record:
                raise ValueError("items.jsonl contains labels.")
            records.append({ID_COLUMN: str(record[ID_COLUMN])})
    return pd.DataFrame(records)


if __name__ == "__main__":
    main()
