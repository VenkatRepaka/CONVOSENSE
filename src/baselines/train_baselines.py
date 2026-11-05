#!/usr/bin/env python3
"""Train the dummy and logistic-regression baselines for the tone task.

Step 6 of ``docs/ARI410-510_Project_Leaderboard.pdf`` asks for a dummy
baseline and one trained model. The dummy is whichever of a uniform random
labeler and a majority labeler scores higher on the validation set. The
trained model is word and bigram TF-IDF plus logistic regression
(``sklearn.linear_model.LogisticRegression``). Its regularization and class
weight are chosen on the validation macro F1, then the chosen settings are
refit on train and validation together for the file that goes to Kaggle.

Macro F1 is ``src/competition/macro_f1.py``, the same function the
competition uses. Test labels are read from ``competition/private/`` and
are not written back out. Submission files contain only ``id`` and ``label``.

Run from the repository root::

    python src/baselines/train_baselines.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "competition"))
from macro_f1 import LABEL_COLUMN, TONE_LABELS, score  # noqa: E402

PARTICIPANT_DIR = REPO_ROOT / "competition" / "participant"
PRIVATE_DIR = REPO_ROOT / "competition" / "private"
SUBMISSION_DIR = REPO_ROOT / "competition" / "submissions"
RESULTS_DIR = REPO_ROOT / "baselines"

ID_COLUMN = "id"
USAGE_COLUMN = "Usage"
TEXT_COLUMNS = ["last_customer_message", "conversation"]
RANDOM_SEED = 42
RANDOM_MEAN_SEEDS = 20
C_GRID = (0.25, 1.0, 4.0)
CLASS_WEIGHTS = (None, "balanced")


def main(argv: list[str] | None = None) -> None:
    """Fit the baselines, score them, and write Kaggle submission files."""
    parser = argparse.ArgumentParser(
        description="Fit the majority, random, and logistic-regression baselines."
    )
    parser.add_argument("--participant-dir", type=Path, default=PARTICIPANT_DIR)
    parser.add_argument("--private-dir", type=Path, default=PRIVATE_DIR)
    parser.add_argument("--submission-dir", type=Path, default=SUBMISSION_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    train = _read_split(args.participant_dir / "train.csv")
    validation = _read_split(args.participant_dir / "validation.csv")
    test = _read_split(args.participant_dir / "test.csv")
    solution = _read_split(args.private_dir / "solution.csv")
    _check_solution_covers_test(solution, test)

    majority_test = _majority_labels(train[LABEL_COLUMN], test[ID_COLUMN])
    majority_validation = _majority_labels(train[LABEL_COLUMN], validation[ID_COLUMN])
    majority_label = str(majority_test[LABEL_COLUMN].iloc[0])
    random_by_seed = _random_scores_by_seed(train, validation, test, solution)
    random_seed = _representative_random_seed(random_by_seed)
    random_test = _random_labels(train[LABEL_COLUMN], test[ID_COLUMN], random_seed)
    random_validation = _random_labels(
        train[LABEL_COLUMN], validation[ID_COLUMN], random_seed
    )

    rows: list[dict[str, object]] = []
    per_class: list[dict[str, object]] = []
    _add_model_scores(
        rows,
        per_class,
        "majority",
        validation,
        solution,
        {"validation": majority_validation, "test": majority_test},
        "Always predicts the most common training label, "
        f"{majority_label}. sklearn DummyClassifier(strategy='most_frequent').",
    )
    _add_model_scores(
        rows,
        per_class,
        "random",
        validation,
        solution,
        {"validation": random_validation, "test": random_test},
        "Uniform draw over the five tones. "
        f"DummyClassifier(strategy='uniform', random_state={random_seed}). "
        "This seed is the one whose validation score is closest to the "
        f"mean of seeds 0 to {RANDOM_MEAN_SEEDS - 1}.",
    )
    for split_name in ("validation", "test"):
        rows.append(
            {
                "model": "random_mean",
                "split": split_name,
                "macro_f1": float(random_by_seed[split_name].mean()),
                "notes": (
                    f"Mean macro F1 of {RANDOM_MEAN_SEEDS} uniform seeds "
                    f"(0 to {RANDOM_MEAN_SEEDS - 1})."
                ),
            }
        )

    chosen_dummy = _choose_dummy(rows)
    selected_predictions = majority_test if chosen_dummy == "majority" else random_test
    _add_model_scores(
        rows,
        per_class,
        "dummy_selected",
        validation,
        solution,
        {
            "validation": majority_validation if chosen_dummy == "majority" else random_validation,
            "test": selected_predictions,
        },
        "The dummy uploaded to the leaderboard. "
        f"Same predictions as {chosen_dummy}"
        + (f", seed {random_seed}." if chosen_dummy == "random" else "."),
    )

    search = _search_logistic_regression(train, validation)
    best = search.iloc[0]
    train_only = _fit_from_search_row(train, best)
    submitted = _fit_logistic_regression(
        pd.concat([train, validation], ignore_index=True),
        C=float(best["C"]),
        class_weight=None if best["class_weight"] == "none" else "balanced",
    )
    logistic_test = _predict_frame(submitted, test)
    logistic_note = (
        "Word and bigram TF-IDF on the last customer message and on the full "
        f"conversation, then logistic regression. Selected on validation: "
        f"C={best['C']}, class_weight={best['class_weight']}. "
        "The validation score is the train-only fit. The test score is the "
        "submission refit on train and validation."
    )
    _add_model_scores(
        rows,
        per_class,
        "logistic_regression",
        validation,
        solution,
        {
            "validation": _predict_frame(train_only, validation),
            "test": logistic_test,
        },
        logistic_note,
    )

    args.submission_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    _write_submission(majority_test, args.submission_dir / "majority_baseline.csv")
    _write_submission(random_test, args.submission_dir / "random_baseline.csv")
    _write_submission(selected_predictions, args.submission_dir / "dummy_baseline.csv")
    _write_submission(logistic_test, args.submission_dir / "logistic_regression.csv")

    scores = pd.DataFrame(rows)
    per_class_frame = pd.DataFrame(per_class)
    scores.to_csv(args.results_dir / "baseline_scores.csv", index=False, lineterminator="\n")
    per_class_frame.to_csv(
        args.results_dir / "per_class_f1.csv", index=False, lineterminator="\n"
    )
    search.to_csv(
        args.results_dir / "logistic_regression_search.csv", index=False, lineterminator="\n"
    )
    random_by_seed.to_csv(
        args.results_dir / "random_baseline_seeds.csv", index=False, lineterminator="\n"
    )
    _check_majority_matches_sample(args.participant_dir, majority_test)
    _check_submissions_match_test(args.submission_dir, test)

    print(f"Majority label: {majority_label}")
    print(f"Selected dummy: {chosen_dummy} (random seed {random_seed})")
    print(
        "Logistic regression: "
        f"C={best['C']}, class_weight={best['class_weight']}, "
        f"validation macro F1={float(best['validation_macro_f1']):.4f}"
    )
    print(_score_table(scores).to_string())
    print(f"Submissions: {args.submission_dir}")
    print(f"Scores: {args.results_dir / 'baseline_scores.csv'}")


def _read_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"{path} is missing the id column.")
    return frame


def _check_solution_covers_test(solution: pd.DataFrame, test: pd.DataFrame) -> None:
    if list(solution[ID_COLUMN]) != list(test[ID_COLUMN]):
        raise ValueError("solution.csv ids do not match test.csv in order.")
    if LABEL_COLUMN not in solution.columns or USAGE_COLUMN not in solution.columns:
        raise ValueError("solution.csv needs label and Usage columns.")


def _majority_labels(train_labels: pd.Series, ids: pd.Series) -> pd.DataFrame:
    classifier = DummyClassifier(strategy="most_frequent")
    features = np.zeros((len(train_labels), 1))
    classifier.fit(features, train_labels.to_numpy())
    predicted = classifier.predict(np.zeros((len(ids), 1)))
    return pd.DataFrame({ID_COLUMN: ids.to_numpy(), LABEL_COLUMN: predicted})


def _random_labels(train_labels: pd.Series, ids: pd.Series, seed: int) -> pd.DataFrame:
    classifier = DummyClassifier(strategy="uniform", random_state=seed)
    features = np.zeros((len(train_labels), 1))
    classifier.fit(features, train_labels.to_numpy())
    predicted = classifier.predict(np.zeros((len(ids), 1)))
    return pd.DataFrame({ID_COLUMN: ids.to_numpy(), LABEL_COLUMN: predicted})


def _random_scores_by_seed(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    solution: pd.DataFrame,
) -> pd.DataFrame:
    """Score one uniform draw per seed. The seed for the file is chosen later."""
    rows = []
    for seed in range(RANDOM_MEAN_SEEDS):
        rows.append(
            {
                "seed": seed,
                "validation": _macro_f1(
                    validation,
                    _random_labels(train[LABEL_COLUMN], validation[ID_COLUMN], seed),
                ),
                "test": _macro_f1(
                    solution,
                    _random_labels(train[LABEL_COLUMN], test[ID_COLUMN], seed),
                ),
            }
        )
    return pd.DataFrame(rows)


def _representative_random_seed(scores_by_seed: pd.DataFrame) -> int:
    """Return the seed whose validation score is closest to the seed mean.

    Closeness uses the validation scores only. The lowest seed wins a tie.
    """
    mean_validation = float(scores_by_seed["validation"].mean())
    ranked = scores_by_seed.copy()
    ranked["distance"] = (ranked["validation"] - mean_validation).abs()
    ranked = ranked.sort_values(["distance", "seed"], kind="mergesort")
    return int(ranked.iloc[0]["seed"])


def _choose_dummy(rows: list[dict[str, object]]) -> str:
    """Pick the dummy with the higher validation macro F1.

    The random side of the comparison is the mean over seeds, because one
    uniform draw moves around on 87 validation conversations.
    """
    validation_scores = {
        row["model"]: float(row["macro_f1"])
        for row in rows
        if row["split"] == "validation" and row["model"] in ("majority", "random_mean")
    }
    if validation_scores["random_mean"] > validation_scores["majority"]:
        return "random"
    return "majority"


def _search_logistic_regression(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Score a small grid on validation and return it best-first.

    The grid is not scored on the test labels.
    """
    records = []
    for C in C_GRID:
        for class_weight in CLASS_WEIGHTS:
            model = _fit_logistic_regression(train, C=C, class_weight=class_weight)
            predicted = _predict_frame(model, validation)
            weight_name = "none" if class_weight is None else class_weight
            records.append(
                {
                    "C": C,
                    "class_weight": weight_name,
                    "validation_macro_f1": _macro_f1(validation, predicted),
                }
            )
    search = pd.DataFrame(records)
    search = search.sort_values(
        ["validation_macro_f1", "class_weight", "C"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return search.reset_index(drop=True)


def _fit_from_search_row(train: pd.DataFrame, row: pd.Series) -> Pipeline:
    class_weight = None if row["class_weight"] == "none" else "balanced"
    return _fit_logistic_regression(train, C=float(row["C"]), class_weight=class_weight)


def _fit_logistic_regression(
    train: pd.DataFrame,
    C: float,
    class_weight: str | None,
) -> Pipeline:
    """Fit TF-IDF features and logistic regression on the training rows."""
    features = ColumnTransformer(
        transformers=[
            (
                "last_message",
                TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                "last_customer_message",
            ),
            (
                "conversation",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                "conversation",
            ),
        ]
    )
    classifier = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=2000,
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    pipeline = Pipeline([("features", features), ("classifier", classifier)])
    pipeline.fit(train[TEXT_COLUMNS], train[LABEL_COLUMN])
    iterations = int(np.max(pipeline.named_steps["classifier"].n_iter_))
    if iterations >= 2000:
        raise RuntimeError(f"Logistic regression stopped at max_iter for C={C}.")
    return pipeline


def _predict_frame(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    predicted = model.predict(frame[TEXT_COLUMNS])
    return pd.DataFrame({ID_COLUMN: frame[ID_COLUMN].to_numpy(), LABEL_COLUMN: predicted})


def _add_model_scores(
    rows: list[dict[str, object]],
    per_class: list[dict[str, object]],
    model_name: str,
    validation: pd.DataFrame,
    solution: pd.DataFrame,
    predictions_by_split: dict[str, pd.DataFrame],
    notes: str,
) -> None:
    """Record validation, full-test, public, and private macro F1."""
    if "validation" in predictions_by_split:
        predicted = predictions_by_split["validation"]
        rows.append(
            {
                "model": model_name,
                "split": "validation",
                "macro_f1": _macro_f1(validation, predicted),
                "notes": notes,
            }
        )
        per_class.extend(_per_class_rows(model_name, "validation", validation, predicted))
    if "test" in predictions_by_split:
        predicted = predictions_by_split["test"]
        rows.append(
            {
                "model": model_name,
                "split": "test",
                "macro_f1": _macro_f1(solution, predicted),
                "notes": notes,
            }
        )
        per_class.extend(_per_class_rows(model_name, "test", solution, predicted))
        for usage in ("Public", "Private"):
            usage_key = solution[solution[USAGE_COLUMN] == usage]
            rows.append(
                {
                    "model": model_name,
                    "split": usage.lower(),
                    "macro_f1": _macro_f1(usage_key, predicted),
                    "notes": notes,
                }
            )
            per_class.extend(
                _per_class_rows(model_name, usage.lower(), usage_key, predicted)
            )


def _macro_f1(key: pd.DataFrame, predicted: pd.DataFrame) -> float:
    """Score ``predicted`` with the competition metric.

    ``key`` must contain the true ``label`` column. For the validation file
    that column is already named ``label``.
    """
    if LABEL_COLUMN not in key.columns:
        raise ValueError("The answer key is missing the label column.")
    aligned = _align(key, predicted)
    return float(score(key, aligned, ID_COLUMN))


def _per_class_rows(
    model_name: str,
    split_name: str,
    key: pd.DataFrame,
    predicted: pd.DataFrame,
) -> list[dict[str, object]]:
    from sklearn.metrics import f1_score

    aligned = _align(key, predicted)
    y_true = key[LABEL_COLUMN].astype(str).to_numpy()
    y_pred = aligned[LABEL_COLUMN].astype(str).to_numpy()
    present = [label for label in TONE_LABELS if (y_true == label).any()]
    values = f1_score(y_true, y_pred, average=None, labels=present, zero_division=0)
    return [
        {
            "model": model_name,
            "split": split_name,
            "label": label,
            "f1": float(value),
        }
        for label, value in zip(present, values)
    ]


def _align(key: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    aligned = key[[ID_COLUMN]].merge(predicted, on=ID_COLUMN, how="left")
    if len(aligned) != len(key) or aligned[LABEL_COLUMN].eq("").any():
        raise ValueError("Predictions are missing an id from the scored split.")
    unknown = set(aligned[LABEL_COLUMN]) - set(TONE_LABELS)
    if unknown:
        raise ValueError(f"Predictions contain unknown labels: {sorted(unknown)}")
    return aligned


def _write_submission(frame: pd.DataFrame, path: Path) -> None:
    frame[[ID_COLUMN, LABEL_COLUMN]].to_csv(path, index=False, lineterminator="\n")


def _check_majority_matches_sample(participant_dir: Path, majority: pd.DataFrame) -> None:
    sample = _read_split(participant_dir / "sample_submission.csv")
    if list(sample[ID_COLUMN]) != list(majority[ID_COLUMN]):
        raise ValueError("Majority submission ids do not match sample_submission.csv.")
    if list(sample[LABEL_COLUMN]) != list(majority[LABEL_COLUMN]):
        raise ValueError("Majority submission labels do not match sample_submission.csv.")


def _check_submissions_match_test(submission_dir: Path, test: pd.DataFrame) -> None:
    for path in sorted(submission_dir.glob("*.csv")):
        submission = _read_split(path)
        if list(submission.columns) != [ID_COLUMN, LABEL_COLUMN]:
            raise ValueError(f"{path.name} columns are {list(submission.columns)}.")
        if list(submission[ID_COLUMN]) != list(test[ID_COLUMN]):
            raise ValueError(f"{path.name} ids do not match test.csv.")


def _score_table(scores: pd.DataFrame) -> pd.DataFrame:
    table = scores.pivot(index="model", columns="split", values="macro_f1")
    order = ["majority", "random", "random_mean", "dummy_selected", "logistic_regression"]
    columns = ["validation", "public", "private", "test"]
    table = table.reindex(order)
    present_columns = [column for column in columns if column in table.columns]
    return table[present_columns].astype(float).round(4)


if __name__ == "__main__":
    main()
