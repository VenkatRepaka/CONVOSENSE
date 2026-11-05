#!/usr/bin/env python3
"""Write the Kaggle files for the customer-tone competition.

Step 5 of ``docs/ARI410-510_Project_Leaderboard.pdf`` asks for a training,
validation, and test split, four CSV files with an id column first, and a
hidden test answer key. This script builds those files from the
reliability-weighted vote. That vote gives every annotated conversation one
label. Majority vote leaves two ties unlabeled, so it is not the label used
here.

About 60% of conversations are train, 20% validation, and 20% test, with
every tone in each part. The test rows are split about in half into Public
and Private, stratified by tone.

The split is frozen. ``competition/split_assignments.csv`` records the part
(train, validation, or test) of every conversation that has been split
before, with ids only. Those conversations keep their part on every run, so
a test conversation never moves into a training file, whose labels are
public. Only conversations missing from that file are split, tone by tone,
and then added to it. Commit the updated file with the participant files.

The answer key is written to ``competition/private/``, which is gitignored.
``competition/participant/`` is the folder to upload on Kaggle's Data tab.

Run from the repository root::

    python src/competition/prepare_competition_files.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .macro_f1 import LABEL_COLUMN, TONE_LABELS, ParticipantVisibleError, score
except ImportError:
    # Allows `python src/competition/prepare_competition_files.py`.
    from macro_f1 import LABEL_COLUMN, TONE_LABELS, ParticipantVisibleError, score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUND_TRUTH = REPO_ROOT / "outputs" / "ground_truth_weighted_vote.csv"
DEFAULT_CONVERSATIONS = (
    REPO_ROOT / "data" / "conversations_multi_jsonl" / "AmazonHelp" / "conversations.jsonl"
)
DEFAULT_PARTICIPANT_DIR = REPO_ROOT / "competition" / "participant"
DEFAULT_PRIVATE_DIR = REPO_ROOT / "competition" / "private"
DEFAULT_SPLIT_ASSIGNMENTS = REPO_ROOT / "competition" / "split_assignments.csv"

# 60% train, 20% validation, 20% test. The validation cut is 25% of the
# rows that remain after the test set is held out (0.25 * 0.80 = 0.20).
TEST_SIZE = 0.20
VALIDATION_SIZE_OF_REMAINDER = 0.25
PUBLIC_SIZE_OF_TEST = 0.50
SPLIT_SEED = 42

ID_COLUMN = "id"
SPLIT_COLUMN = "split"
SPLIT_NAMES = ("train", "validation", "test")
USAGE_COLUMN = "Usage"
PARTICIPANT_COLUMNS = [ID_COLUMN, "conversation", "last_customer_message", LABEL_COLUMN]
TEST_COLUMNS = [ID_COLUMN, "conversation", "last_customer_message"]
SOLUTION_COLUMNS = [ID_COLUMN, LABEL_COLUMN, USAGE_COLUMN]
SUBMISSION_COLUMNS = [ID_COLUMN, LABEL_COLUMN]

# Both leaderboard halves need several examples of the smallest tone.
MINIMUM_PER_CLASS = 4

PRIVATE_NOTICE = """\
These files are the competition answer key.

Upload solution.csv only in Kaggle's solution / answer-key slot.
Upload perfect_submission.csv only as your own test submission, to confirm
the metric returns 1. Upload sample_submission.csv from the participant
folder to confirm a non-perfect submission is scored.

Do not put this folder on the participant Data tab, GitHub, Google Drive,
or Discord. The test labels have to stay hidden.
"""


def load_labeled_conversations(
    ground_truth_csv: Path,
    conversations_jsonl: Path,
) -> pd.DataFrame:
    """Join weighted-vote labels to the plain-text conversations.

    Parameters
    ----------
    ground_truth_csv:
        ``ground_truth_weighted_vote.csv``. One row per conversation.
    conversations_jsonl:
        AmazonHelp JSONL with ``id``, ``conversation``, and
        ``last_customer_message``.

    Returns
    -------
    pandas.DataFrame
        Columns ``id``, ``conversation``, ``last_customer_message``, and
        ``label``. One row per labeled conversation.

    Raises
    ------
    FileNotFoundError
        If either input file is missing.
    ValueError
        If a label is blank, unknown, duplicated, or has no conversation.
    """
    ground_truth_csv = Path(ground_truth_csv)
    conversations_jsonl = Path(conversations_jsonl)
    if not ground_truth_csv.is_file():
        raise FileNotFoundError(ground_truth_csv)
    if not conversations_jsonl.is_file():
        raise FileNotFoundError(conversations_jsonl)

    labels = pd.read_csv(ground_truth_csv, dtype=str)
    required = {"instance_id", "ground_truth_label"}
    missing_columns = required - set(labels.columns)
    if missing_columns:
        raise ValueError(
            f"{ground_truth_csv} is missing columns: {sorted(missing_columns)}"
        )
    labels = labels.rename(
        columns={"instance_id": ID_COLUMN, "ground_truth_label": LABEL_COLUMN}
    )
    labels[ID_COLUMN] = labels[ID_COLUMN].astype(str).str.strip()
    labels[LABEL_COLUMN] = labels[LABEL_COLUMN].astype(str).str.strip()
    labels = labels[[ID_COLUMN, LABEL_COLUMN]]

    if labels[ID_COLUMN].duplicated().any():
        repeated = labels.loc[labels[ID_COLUMN].duplicated(), ID_COLUMN]
        raise ValueError(f"Duplicate ids in the ground truth: {repeated.tolist()}")
    blank = labels[LABEL_COLUMN].isna() | labels[LABEL_COLUMN].isin(("", "nan"))
    if blank.any():
        raise ValueError(
            "Ground truth has unlabeled conversations: "
            + ", ".join(labels.loc[blank, ID_COLUMN].tolist())
        )
    unknown = sorted(set(labels[LABEL_COLUMN]) - set(TONE_LABELS))
    if unknown:
        raise ValueError(f"Ground truth has unknown labels: {unknown}")

    conversations = _load_conversations(conversations_jsonl)
    merged = labels.merge(conversations, on=ID_COLUMN, how="left", indicator=True)
    missing = merged.loc[merged["_merge"] != "both", ID_COLUMN]
    if not missing.empty:
        raise ValueError(
            "No conversation text for "
            f"{len(missing)} labeled ids, including {missing.iloc[0]}"
        )
    empty_text = merged["conversation"].str.strip().eq("") | merged[
        "last_customer_message"
    ].str.strip().eq("")
    if empty_text.any():
        raise ValueError(
            "Empty conversation text for ids: "
            + ", ".join(merged.loc[empty_text, ID_COLUMN].tolist())
        )
    return merged[PARTICIPANT_COLUMNS].reset_index(drop=True)


def load_split_assignments(path: Path) -> pd.DataFrame:
    """Read the frozen split, or return an empty one if the file is missing.

    Returns
    -------
    pandas.DataFrame
        Columns ``id`` and ``split``. ``split`` is train, validation, or test.

    Raises
    ------
    ValueError
        If an id repeats or a split name is unknown.
    """
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=[ID_COLUMN, SPLIT_COLUMN])
    assignments = _read_csv(path)
    if list(assignments.columns) != [ID_COLUMN, SPLIT_COLUMN]:
        raise ValueError(
            f"{path} columns are {list(assignments.columns)}; "
            f"expected {[ID_COLUMN, SPLIT_COLUMN]}"
        )
    if assignments[ID_COLUMN].duplicated().any():
        repeated = assignments.loc[assignments[ID_COLUMN].duplicated(), ID_COLUMN]
        raise ValueError(f"Duplicate ids in {path}: {repeated.tolist()}")
    unknown = sorted(set(assignments[SPLIT_COLUMN]) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"Unknown split names in {path}: {unknown}")
    return assignments


def split_labeled_conversations(
    labeled: pd.DataFrame,
    seed: int = SPLIT_SEED,
    assignments: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Split conversations into train, validation, public test, and private test.

    Parameters
    ----------
    labeled:
        Output of ``load_labeled_conversations``.
    seed:
        Seed for the new conversations and for the Public/Private cut.
    assignments:
        Output of ``load_split_assignments``. A conversation listed there
        keeps its part. The rest are split tone by tone: about 20% of each
        tone to test, 20% to validation, and the rest to train.

    Returns
    -------
    dict
        Keys ``train``, ``validation``, and ``test``. ``test`` includes a
        ``Usage`` column with ``Public`` or ``Private``. Rows are sorted by
        numeric id.

    Raises
    ------
    ValueError
        If a frozen id has no label. Dropping it would silently change the
        test set.
    """
    if assignments is None:
        assignments = pd.DataFrame(columns=[ID_COLUMN, SPLIT_COLUMN])
    unlabeled = sorted(set(assignments[ID_COLUMN]) - set(labeled[ID_COLUMN]))
    if unlabeled:
        raise ValueError(
            f"{len(unlabeled)} ids in the frozen split have no label, "
            f"including {unlabeled[0]}"
        )

    frozen = labeled.merge(assignments, on=ID_COLUMN, how="inner")
    new = labeled[~labeled[ID_COLUMN].isin(assignments[ID_COLUMN])]
    assigned = pd.concat([frozen, _split_new_by_tone(new, seed)], ignore_index=True)

    train = assigned[assigned[SPLIT_COLUMN] == "train"].drop(columns=SPLIT_COLUMN)
    validation = assigned[assigned[SPLIT_COLUMN] == "validation"].drop(columns=SPLIT_COLUMN)
    # Sorted first, so the Public/Private cut depends only on the test ids
    # and labels, not on the order they were loaded in.
    test = _sort_by_id(assigned[assigned[SPLIT_COLUMN] == "test"].drop(columns=SPLIT_COLUMN))
    public, private = train_test_split(
        test,
        test_size=1 - PUBLIC_SIZE_OF_TEST,
        random_state=seed,
        stratify=test[LABEL_COLUMN],
    )
    public = public.copy()
    private = private.copy()
    public[USAGE_COLUMN] = "Public"
    private[USAGE_COLUMN] = "Private"
    test = pd.concat([public, private], ignore_index=True)

    splits = {
        "train": _sort_by_id(train),
        "validation": _sort_by_id(validation),
        "test": _sort_by_id(test),
    }
    _check_split(labeled, splits)
    return splits


def write_competition_files(
    splits: dict[str, pd.DataFrame],
    participant_dir: Path,
    private_dir: Path,
) -> None:
    """Write the participant CSVs and the gitignored answer key.

    ``sample_submission.csv`` predicts the most common training label for
    every test id. ``perfect_submission.csv`` is the answer key without the
    ``Usage`` column, for the host's perfect-score check.
    """
    participant_dir = Path(participant_dir)
    private_dir = Path(private_dir)
    participant_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    majority_label = _majority_label(train[LABEL_COLUMN])

    _write_csv(train[PARTICIPANT_COLUMNS], participant_dir / "train.csv")
    _write_csv(validation[PARTICIPANT_COLUMNS], participant_dir / "validation.csv")
    _write_csv(test[TEST_COLUMNS], participant_dir / "test.csv")
    sample = test[[ID_COLUMN]].copy()
    sample[LABEL_COLUMN] = majority_label
    _write_csv(sample[SUBMISSION_COLUMNS], participant_dir / "sample_submission.csv")

    solution = test[SOLUTION_COLUMNS]
    _write_csv(solution, private_dir / "solution.csv")
    _write_csv(
        solution[[ID_COLUMN, LABEL_COLUMN]],
        private_dir / "perfect_submission.csv",
    )
    _write_csv(_split_summary(splits), private_dir / "split_summary.csv")
    _write_csv(_split_ids(splits), private_dir / "split_ids.csv")
    (private_dir / "DO_NOT_PUBLISH.txt").write_text(PRIVATE_NOTICE, encoding="utf-8")
    _check_text_roundtrip(splits, participant_dir)


def check_written_files(participant_dir: Path, private_dir: Path) -> dict[str, float]:
    """Reload the CSVs and score the perfect and sample submissions.

    Returns
    -------
    dict
        Macro F1 of the sample submission on the full test key and on each
        leaderboard half. The perfect submission is required to score 1.

    Raises
    ------
    ValueError
        If a file drifted from the layout Kaggle expects, or the perfect
        submission scores below 1.
    """
    participant_dir = Path(participant_dir)
    private_dir = Path(private_dir)
    train = _read_csv(participant_dir / "train.csv")
    validation = _read_csv(participant_dir / "validation.csv")
    test = _read_csv(participant_dir / "test.csv")
    sample = _read_csv(participant_dir / "sample_submission.csv")
    solution = _read_csv(private_dir / "solution.csv")
    perfect = _read_csv(private_dir / "perfect_submission.csv")

    for name, frame, columns in (
        ("train.csv", train, PARTICIPANT_COLUMNS),
        ("validation.csv", validation, PARTICIPANT_COLUMNS),
        ("test.csv", test, TEST_COLUMNS),
        ("sample_submission.csv", sample, SUBMISSION_COLUMNS),
        ("solution.csv", solution, SOLUTION_COLUMNS),
        ("perfect_submission.csv", perfect, SUBMISSION_COLUMNS),
    ):
        if list(frame.columns) != columns:
            raise ValueError(f"{name} columns are {list(frame.columns)}; expected {columns}")

    pairwise_overlap = (
        set(train[ID_COLUMN]) & set(validation[ID_COLUMN]),
        set(train[ID_COLUMN]) & set(test[ID_COLUMN]),
        set(validation[ID_COLUMN]) & set(test[ID_COLUMN]),
    )
    if any(pairwise_overlap):
        raise ValueError("An id appears in more than one split.")
    if list(sample[ID_COLUMN]) != list(test[ID_COLUMN]):
        raise ValueError("sample_submission.csv ids do not match test.csv.")
    if list(solution[ID_COLUMN]) != list(test[ID_COLUMN]):
        raise ValueError("solution.csv ids do not match test.csv.")
    if list(perfect[ID_COLUMN]) != list(test[ID_COLUMN]):
        raise ValueError("perfect_submission.csv ids do not match test.csv.")
    if set(solution[USAGE_COLUMN]) != {"Public", "Private"}:
        raise ValueError("Usage must be Public or Private.")
    if not solution[LABEL_COLUMN].equals(perfect[LABEL_COLUMN]):
        raise ValueError("perfect_submission.csv does not match the answer key.")

    for name, key in (
        ("full test", solution),
        ("public", solution[solution[USAGE_COLUMN] == "Public"]),
        ("private", solution[solution[USAGE_COLUMN] == "Private"]),
    ):
        perfect_score = score(key, _rows_for_key(key, perfect), ID_COLUMN)
        if perfect_score != 1.0:
            raise ValueError(f"Perfect submission scored {perfect_score} on {name}.")

    bad = perfect.copy()
    bad.loc[0, LABEL_COLUMN] = "not a tone"
    try:
        score(solution, bad, ID_COLUMN)
    except ParticipantVisibleError:
        pass
    else:
        raise ValueError("A submission with an unknown label was accepted.")

    return {
        "full": score(solution, sample, ID_COLUMN),
        "public": score(
            solution[solution[USAGE_COLUMN] == "Public"],
            _rows_for_key(solution[solution[USAGE_COLUMN] == "Public"], sample),
            ID_COLUMN,
        ),
        "private": score(
            solution[solution[USAGE_COLUMN] == "Private"],
            _rows_for_key(solution[solution[USAGE_COLUMN] == "Private"], sample),
            ID_COLUMN,
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Load the weighted-vote labels, split them, and write the Kaggle files."""
    parser = argparse.ArgumentParser(
        description=(
            "Split the labeled AmazonHelp conversations and write the "
            "Kaggle competition files."
        )
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Weighted-vote ground truth CSV.",
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=DEFAULT_CONVERSATIONS,
        help="AmazonHelp conversations JSONL.",
    )
    parser.add_argument(
        "--participant-dir",
        type=Path,
        default=DEFAULT_PARTICIPANT_DIR,
        help="Folder of files participants download.",
    )
    parser.add_argument(
        "--private-dir",
        type=Path,
        default=DEFAULT_PRIVATE_DIR,
        help="Folder for the answer key. Keep it out of git.",
    )
    parser.add_argument(
        "--split-assignments",
        type=Path,
        default=DEFAULT_SPLIT_ASSIGNMENTS,
        help="Frozen split (ids only). Read, then rewritten with any new ids.",
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args(argv)

    labeled = load_labeled_conversations(args.ground_truth, args.conversations)
    assignments = load_split_assignments(args.split_assignments)
    splits = split_labeled_conversations(labeled, seed=args.seed, assignments=assignments)
    write_competition_files(splits, args.participant_dir, args.private_dir)
    _write_split_assignments(splits, args.split_assignments)
    new_count = len(labeled) - len(assignments)
    sample_score = check_written_files(args.participant_dir, args.private_dir)

    summary = _split_summary(splits)
    totals = summary.groupby("split")["conversations"].sum()
    split_order = ["train", "validation", "test", "public", "private"]
    print(f"Labeled conversations: {len(labeled)}")
    print(f"Kept from the frozen split: {len(assignments)}; newly split: {new_count}")
    print(f"Seed: {args.seed}")
    print(
        "Split sizes: "
        + ", ".join(f"{name} {int(totals[name])}" for name in split_order)
    )
    print("Conversations per tone:")
    print(_summary_table(summary).to_string())
    print(f"Sample submission label: {_majority_label(splits['train'][LABEL_COLUMN])}")
    print("Perfect submission macro F1: 1.0")
    print(
        "Sample submission macro F1: "
        f"public {sample_score['public']:.4f}, "
        f"private {sample_score['private']:.4f}, "
        f"full test {sample_score['full']:.4f}"
    )
    score_check = Path(args.private_dir) / "score_check.txt"
    score_check.write_text(
        "\n".join(
            [
                "Upload perfect_submission.csv and expect 1.0000 on the public and private leaderboards.",
                "Upload sample_submission.csv and expect these macro F1 scores:",
                f"public {sample_score['public']:.4f}",
                f"private {sample_score['private']:.4f}",
                f"full test {sample_score['full']:.4f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Participant files: {args.participant_dir}")
    print(f"Answer key (do not publish): {args.private_dir}")


def _load_conversations(path: Path) -> pd.DataFrame:
    """Read id and the two text fields from the AmazonHelp JSONL."""
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                {
                    ID_COLUMN: str(record["id"]).strip(),
                    "conversation": str(record["conversation"]),
                    "last_customer_message": str(record["last_customer_message"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No conversations in {path}")
    if frame[ID_COLUMN].duplicated().any():
        repeated = frame.loc[frame[ID_COLUMN].duplicated(), ID_COLUMN]
        raise ValueError(f"Duplicate conversation ids: {repeated.head(5).tolist()}")
    return frame


def _split_new_by_tone(new: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign conversations that are not in the frozen split, one tone at a time.

    ``train_test_split`` cannot stratify a tone with a single conversation,
    and a batch of new labels can be that small. Each tone is shuffled with
    ``seed`` instead. About 20% of each tone goes to test and 25% of the rest
    to validation, matching the 60/20/20 split.
    """
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for label in TONE_LABELS:
        tone = _sort_by_id(new[new[LABEL_COLUMN] == label])
        if tone.empty:
            continue
        tone = tone.iloc[rng.permutation(len(tone))].reset_index(drop=True)
        n_test = int(len(tone) * TEST_SIZE + 0.5)
        n_validation = int((len(tone) - n_test) * VALIDATION_SIZE_OF_REMAINDER + 0.5)
        tone[SPLIT_COLUMN] = "train"
        tone.loc[: n_test - 1, SPLIT_COLUMN] = "test"
        tone.loc[n_test : n_test + n_validation - 1, SPLIT_COLUMN] = "validation"
        parts.append(tone)
    if not parts:
        return new.assign(**{SPLIT_COLUMN: pd.Series(dtype=str)})
    return pd.concat(parts, ignore_index=True)


def _write_split_assignments(splits: dict[str, pd.DataFrame], path: Path) -> None:
    """Write id and part for every conversation, so the next run keeps them."""
    rows = [
        splits[name][[ID_COLUMN]].assign(**{SPLIT_COLUMN: name}) for name in SPLIT_NAMES
    ]
    _write_csv(pd.concat(rows, ignore_index=True), Path(path))


def _sort_by_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by numeric tweet id so the CSV order stays stable across reruns."""
    ordered = frame.copy()
    ordered["_sort_id"] = pd.to_numeric(ordered[ID_COLUMN], errors="raise")
    ordered = ordered.sort_values("_sort_id", kind="mergesort").drop(columns="_sort_id")
    return ordered.reset_index(drop=True)


def _majority_label(labels: pd.Series) -> str:
    """Return the most common tone, breaking a tie toward the guideline order."""
    counts = labels.value_counts()
    top = int(counts.max())
    tied = [label for label in TONE_LABELS if int(counts.get(label, 0)) == top]
    return tied[0]


def _check_split(labeled: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    """Require a complete partition and several examples of every tone."""
    train_ids = set(splits["train"][ID_COLUMN])
    validation_ids = set(splits["validation"][ID_COLUMN])
    test_ids = set(splits["test"][ID_COLUMN])
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("The split assigned an id to more than one part.")
    combined = train_ids | validation_ids | test_ids
    if combined != set(labeled[ID_COLUMN]):
        raise ValueError("The split dropped or added a conversation.")

    test = splits["test"]
    if set(test[USAGE_COLUMN]) != {"Public", "Private"}:
        raise ValueError("Every test row needs Usage of Public or Private.")
    public = test[test[USAGE_COLUMN] == "Public"]
    private = test[test[USAGE_COLUMN] == "Private"]
    for name, frame in (
        ("train", splits["train"]),
        ("validation", splits["validation"]),
        ("public", public),
        ("private", private),
    ):
        counts = frame[LABEL_COLUMN].value_counts()
        for label in TONE_LABELS:
            count = int(counts.get(label, 0))
            if count < MINIMUM_PER_CLASS:
                raise ValueError(
                    f"{name} has {count} conversations labeled {label}; "
                    f"need at least {MINIMUM_PER_CLASS}."
                )
    public_counts = public[LABEL_COLUMN].value_counts()
    private_counts = private[LABEL_COLUMN].value_counts()
    for label in TONE_LABELS:
        gap = abs(int(public_counts.get(label, 0)) - int(private_counts.get(label, 0)))
        if gap > 1:
            raise ValueError(f"Public and Private differ by {gap} on {label}.")


def _split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count conversations by split and tone, including the two test halves."""
    test = splits["test"]
    parts = {
        "train": splits["train"],
        "validation": splits["validation"],
        "test": test,
        "public": test[test[USAGE_COLUMN] == "Public"],
        "private": test[test[USAGE_COLUMN] == "Private"],
    }
    rows: list[dict[str, object]] = []
    for split_name, frame in parts.items():
        counts = frame[LABEL_COLUMN].value_counts()
        for label in TONE_LABELS:
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "conversations": int(counts.get(label, 0)),
                }
            )
    return pd.DataFrame(rows)


def _summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long summary so each split is a column."""
    table = summary.pivot(index="label", columns="split", values="conversations")
    table = table.reindex(list(TONE_LABELS))
    column_order = ["train", "validation", "test", "public", "private"]
    return table[column_order].astype(int)


def _split_ids(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """List every id and its split, without labels."""
    rows: list[pd.DataFrame] = []
    for split_name in ("train", "validation"):
        part = splits[split_name][[ID_COLUMN]].copy()
        part["split"] = split_name
        part[USAGE_COLUMN] = ""
        rows.append(part)
    test = splits["test"][[ID_COLUMN, USAGE_COLUMN]].copy()
    test["split"] = "test"
    rows.append(test[[ID_COLUMN, "split", USAGE_COLUMN]])
    return pd.concat(rows, ignore_index=True)


def _rows_for_key(key: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
    """Return ``submission`` rows in the same order as ``key``."""
    aligned = key[[ID_COLUMN]].merge(submission, on=ID_COLUMN, how="left")
    if aligned[LABEL_COLUMN].isna().any() or aligned[LABEL_COLUMN].eq("").any() or len(aligned) != len(key):
        raise ValueError("Submission is missing a row from the answer key.")
    return aligned


def _check_text_roundtrip(splits: dict[str, pd.DataFrame], participant_dir: Path) -> None:
    """Require the written text to match the conversations that were split."""
    expected = {
        "train.csv": splits["train"],
        "validation.csv": splits["validation"],
        "test.csv": splits["test"],
    }
    for name, frame in expected.items():
        written = _read_csv(participant_dir / name)
        for column in ("conversation", "last_customer_message"):
            if column not in written.columns:
                continue
            if not written[column].equals(frame[column].astype(str).reset_index(drop=True)):
                raise ValueError(f"{name} changed {column} while writing the CSV.")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


if __name__ == "__main__":
    main()
