#!/usr/bin/env python3
"""Write copies of the competition files with numeric tone labels for Kaggle.

Kaggle rejected the tone names as labels, so the upload files use the
numbers 1 to 5 in the guideline order (1 = Appreciative or positive, ...,
5 = Escalatory or requesting immediate resolution). The repository's own
files in ``competition/participant/``, ``competition/private/``, and
``competition/submissions/`` keep the tone names, because the baselines and
the LLM scorer read them.

Outputs:

- ``competition/kaggle_upload/``: ``train.csv``, ``validation.csv``,
  ``test.csv``, ``sample_submission.csv``, and ``label_map.csv``, for the
  Data tab.
- ``competition/kaggle_upload/benchmarks/``: the two Benchmark submissions.
- ``competition/private/kaggle_upload/``: ``solution.csv`` and
  ``perfect_submission.csv``. That folder is gitignored.

Run from the repository root after ``prepare_competition_files.py`` and
``train_baselines.py``::

    python src/competition/make_kaggle_numeric_files.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .macro_f1 import LABEL_COLUMN, TONE_LABELS, score
except ImportError:
    # Allows `python src/competition/make_kaggle_numeric_files.py`.
    from macro_f1 import LABEL_COLUMN, TONE_LABELS, score

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPETITION_DIR = REPO_ROOT / "competition"
DEFAULT_PARTICIPANT_DIR = COMPETITION_DIR / "participant"
DEFAULT_PRIVATE_DIR = COMPETITION_DIR / "private"
DEFAULT_SUBMISSION_DIR = COMPETITION_DIR / "submissions"
DEFAULT_UPLOAD_DIR = COMPETITION_DIR / "kaggle_upload"

ID_COLUMN = "id"
USAGE_COLUMN = "Usage"
TONE_TO_NUMBER = {tone: number for number, tone in enumerate(TONE_LABELS, start=1)}

LABELED_FILES = ("train.csv", "validation.csv", "sample_submission.csv")
BENCHMARK_FILES = ("dummy_baseline.csv", "logistic_regression.csv")
PRIVATE_FILES = ("solution.csv", "perfect_submission.csv")


def to_numeric_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace each tone name in ``label`` with its number, 1 to 5.

    Raises
    ------
    ValueError
        If a label is not one of the five tone names.
    """
    unknown = sorted(set(frame[LABEL_COLUMN]) - set(TONE_TO_NUMBER))
    if unknown:
        raise ValueError(f"Labels outside the five tones: {unknown}")
    converted = frame.copy()
    converted[LABEL_COLUMN] = converted[LABEL_COLUMN].map(TONE_TO_NUMBER).astype(int)
    return converted


def write_numeric_files(
    participant_dir: Path,
    private_dir: Path,
    submission_dir: Path,
    upload_dir: Path,
) -> None:
    """Convert every Kaggle file and check that the scores do not change."""
    public_out = Path(upload_dir)
    benchmark_out = public_out / "benchmarks"
    private_out = Path(private_dir) / "kaggle_upload"
    for folder in (public_out, benchmark_out, private_out):
        folder.mkdir(parents=True, exist_ok=True)

    for name in LABELED_FILES:
        _write_csv(to_numeric_labels(_read_csv(Path(participant_dir) / name)), public_out / name)
    _write_csv(_read_csv(Path(participant_dir) / "test.csv"), public_out / "test.csv")
    _write_csv(
        pd.DataFrame({LABEL_COLUMN: list(TONE_TO_NUMBER.values()), "tone": list(TONE_TO_NUMBER)}),
        public_out / "label_map.csv",
    )
    for name in BENCHMARK_FILES:
        _write_csv(to_numeric_labels(_read_csv(Path(submission_dir) / name)), benchmark_out / name)
    for name in PRIVATE_FILES:
        _write_csv(to_numeric_labels(_read_csv(Path(private_dir) / name)), private_out / name)

    _check_scores_match(Path(private_dir), private_out, Path(participant_dir), public_out, Path(submission_dir), benchmark_out)


def _check_scores_match(
    private_dir: Path,
    private_out: Path,
    participant_dir: Path,
    public_out: Path,
    submission_dir: Path,
    benchmark_out: Path,
) -> None:
    """Require every submission to score the same with numbers as with names."""
    text_key = _read_csv(private_dir / "solution.csv")
    number_key = _read_csv(private_out / "solution.csv")
    pairs = [
        (private_dir / "perfect_submission.csv", private_out / "perfect_submission.csv"),
        (participant_dir / "sample_submission.csv", public_out / "sample_submission.csv"),
    ] + [(submission_dir / name, benchmark_out / name) for name in BENCHMARK_FILES]
    for text_path, number_path in pairs:
        for usage in ("Public", "Private"):
            text_score = score(
                text_key[text_key[USAGE_COLUMN] == usage],
                _aligned(text_key, usage, _read_csv(text_path)),
                ID_COLUMN,
            )
            number_score = score(
                number_key[number_key[USAGE_COLUMN] == usage],
                _aligned(number_key, usage, _read_csv(number_path)),
                ID_COLUMN,
            )
            if abs(text_score - number_score) > 1e-12:
                raise ValueError(
                    f"{number_path.name} scores {number_score} on {usage} "
                    f"but the tone-name file scores {text_score}."
                )
            print(f"{number_path.name:<26} {usage:<8} {number_score:.4f}")


def _aligned(key: pd.DataFrame, usage: str, submission: pd.DataFrame) -> pd.DataFrame:
    """Return ``submission`` rows in the order of one leaderboard half of ``key``."""
    rows = key.loc[key[USAGE_COLUMN] == usage, [ID_COLUMN]]
    return rows.merge(submission, on=ID_COLUMN, how="left")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> None:
    """Write the numeric-label Kaggle files and print their scores."""
    parser = argparse.ArgumentParser(
        description="Write Kaggle upload files with tone labels as the numbers 1 to 5."
    )
    parser.add_argument("--participant-dir", type=Path, default=DEFAULT_PARTICIPANT_DIR)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    args = parser.parse_args(argv)
    write_numeric_files(args.participant_dir, args.private_dir, args.submission_dir, args.upload_dir)
    print(f"Data tab and benchmarks: {args.upload_dir}")
    print(f"Answer key (do not publish): {Path(args.private_dir) / 'kaggle_upload'}")


if __name__ == "__main__":
    main()
