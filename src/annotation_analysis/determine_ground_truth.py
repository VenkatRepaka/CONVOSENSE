#!/usr/bin/env python3
"""Determine ground-truth customer-tone labels from annotator submissions.

Running this file also computes step 2.2 (annotator agreement) and step 4
(label distribution and class balance). Those metrics live in
``agreement.py`` and ``analyze_distribution.py``. This module implements
step 3,
"Determine Ground Truth Labels," of
``docs/ARI410-510_Project_Leaderboard.pdf``:

* **Majority vote.** The label chosen most often. ``pandas.DataFrame.mode``
  returns every label that shares that top count, so a tie stays unlabeled
  for adjudication instead of being broken alphabetically.
* **Weighted vote.** Each annotator's weight is their mean pairwise Cohen's
  kappa with the other annotators (``sklearn.metrics.cohen_kappa_score``).
  The label with the greatest total weight wins. Ties stay unlabeled.
* **Adjudication list.** Every item whose annotators did not all choose the
  same primary label.

The label used here is the primary Potato field ``conversation_sense``.
Optional ``secondary_sense`` labels are context for the annotator and are
not votes. An item labeled by one annotator keeps that label. The annotator
id is the submission folder name (``annotator_01``), because ``user_id`` in
the JSONL is a display name for some classmates and an id for others.

Each of annotators 01–06 turned in the same JSONL twice (the file at the
top of the folder, and Potato's export copy). This loader reads the
top-level ``annotations.jsonl`` when it is there. Annotator 07 submitted
only ``annotator_07/exports/jsonl/annotations.jsonl``.

Run from the repository root::

    python src/annotation_analysis/determine_ground_truth.py

That writes the agreement table, the ground-truth tables, the label
distribution, and three charts: ``outputs/annotator_agreement.png``,
``outputs/ground_truth_label_distribution.png``, and
``outputs/label_distribution.png``.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

# Repository root: src/annotation_analysis/this_file.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBMISSIONS_DIR = REPO_ROOT / "annotation_submissions"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

# Potato schema. The selected label is stored as {label: label}.
PRIMARY_LABEL_FIELD = "conversation_sense"

# Order used in Customer Tone Annotation Guidelines.md, calm to demanding.
TONE_LABELS = (
    "Appreciative or positive",
    "Neutral or informational",
    "Help-seeking or asking a question",
    "Frustrated or complaining",
    "Escalatory or requesting immediate resolution",
)

# Okabe-Ito blue and orange: distinct under the common forms of color blindness.
METHOD_COLORS = ("#0072B2", "#E69F00")


def load_primary_tone_annotations(submissions_dir: Path) -> pd.DataFrame:
    """Read one primary tone label per annotator and conversation.

    Parameters
    ----------
    submissions_dir:
        Directory of annotator folders. A folder contributes one annotator
        when it contains ``annotations.jsonl`` or, failing that,
        ``exports/jsonl/annotations.jsonl``. The annotator id is the folder
        name.

    Returns
    -------
    pandas.DataFrame
        Columns ``instance_id``, ``annotator_id``, and ``label``. One row
        per submitted primary label.

    Raises
    ------
    FileNotFoundError
        If the directory has no ``annotations.jsonl`` files.
    ValueError
        If the same annotator labeled the same instance more than once.
    """
    submissions_dir = Path(submissions_dir)
    rows: list[dict[str, str]] = []
    annotator_dirs = sorted(
        path for path in submissions_dir.iterdir() if path.is_dir()
    )
    for annotator_dir in annotator_dirs:
        annotation_file = _find_annotation_file(annotator_dir)
        if annotation_file is None:
            continue
        rows.extend(_read_annotator_file(annotator_dir.name, annotation_file))

    if not rows:
        raise FileNotFoundError(
            f"No annotations.jsonl files under {submissions_dir}"
        )

    annotations = pd.DataFrame(rows)
    duplicate = annotations.duplicated(["annotator_id", "instance_id"])
    if duplicate.any():
        repeated = annotations.loc[duplicate, ["annotator_id", "instance_id"]]
        raise ValueError(
            "Duplicate annotations for the same annotator and instance:\n"
            f"{repeated.to_string(index=False)}"
        )
    return annotations


def build_annotator_label_matrix(annotations: pd.DataFrame) -> pd.DataFrame:
    """Pivot annotations to one row per conversation and one column per annotator.

    Missing cells are conversations that annotator did not label. Columns
    are annotator ids in sorted order.

    Parameters
    ----------
    annotations:
        Output of :func:`load_primary_tone_annotations`.

    Returns
    -------
    pandas.DataFrame
        Index ``instance_id``, one column per annotator, values are tone labels.
    """
    matrix = annotations.pivot(
        index="instance_id", columns="annotator_id", values="label"
    )
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    matrix.index.name = "instance_id"
    matrix.columns.name = "annotator_id"
    return matrix


def assign_ground_truth_by_majority_vote(matrix: pd.DataFrame) -> pd.DataFrame:
    """Assign the unique modal label, using ``DataFrame.mode``.

    ``DataFrame.mode`` lists every label tied for the highest count. A
    conversation whose top count is shared stays unlabeled
    (``needs_adjudication`` is true). A conversation with one annotation
    keeps that label and is recorded as ``single_annotator``.

    Parameters
    ----------
    matrix:
        Output of :func:`build_annotator_label_matrix`.

    Returns
    -------
    pandas.DataFrame
        Index ``instance_id``. Columns: ``ground_truth_label``,
        ``resolution``, ``supporting_votes``, ``annotator_count``,
        ``distinct_label_count``, ``needs_adjudication``.
    """
    modes = matrix.mode(axis=1, dropna=True)
    mode_count = modes.notna().sum(axis=1)
    annotator_count = matrix.notna().sum(axis=1)
    distinct_label_count = matrix.apply(
        lambda row: int(row.dropna().nunique()), axis=1
    )
    supporting_votes = matrix.apply(_highest_vote_count, axis=1)

    ground_truth_label = modes.iloc[:, 0].where(mode_count == 1)
    resolution = _resolution_labels(
        annotator_count=annotator_count,
        decided=mode_count == 1,
        decided_name="majority_vote",
    )
    return _ground_truth_frame(
        matrix=matrix,
        ground_truth_label=ground_truth_label,
        resolution=resolution,
        support=supporting_votes.where(mode_count == 1),
        support_column="supporting_votes",
        annotator_count=annotator_count,
        distinct_label_count=distinct_label_count,
        needs_adjudication=mode_count != 1,
    )


def reliability_weights_from_pairwise_cohen_kappa(
    matrix: pd.DataFrame,
) -> pd.Series:
    """Weight each annotator by their mean pairwise Cohen's kappa.

    For every pair of annotators, kappa is computed with
    ``sklearn.metrics.cohen_kappa_score`` on the conversations both of them
    labeled. An annotator's weight is the mean of those kappas. This is the
    reliability weight described for weighted voting in the assignment:
    there is no separate experience score in the submissions.

    A non-positive mean kappa is clipped to zero so an annotator who agrees
    below chance cannot pull the vote toward the opposite label. An
    annotator with no overlapping partner receives the mean of the weights
    that could be computed. If every weight is then zero, every annotator
    receives weight 1.

    Parameters
    ----------
    matrix:
        Output of :func:`build_annotator_label_matrix`.

    Returns
    -------
    pandas.Series
        Index is annotator id, values are weights, name ``reliability_weight``.
    """
    pairwise: dict[str, list[float]] = {annotator: [] for annotator in matrix.columns}
    for left, right in combinations(matrix.columns, 2):
        pair = matrix[[left, right]].dropna()
        both_constant = pair[left].nunique() < 2 and pair[right].nunique() < 2
        if len(pair) < 2 or both_constant:
            continue
        kappa = float(cohen_kappa_score(pair[left], pair[right]))
        if np.isnan(kappa):
            continue
        pairwise[left].append(kappa)
        pairwise[right].append(kappa)

    weights = pd.Series(
        {
            annotator: float(np.mean(kappas)) if kappas else np.nan
            for annotator, kappas in pairwise.items()
        },
        name="reliability_weight",
    )
    computed = weights.dropna()
    fill_value = float(computed.mean()) if not computed.empty else 1.0
    weights = weights.fillna(fill_value).clip(lower=0.0)
    if not (weights > 0).any():
        weights[:] = 1.0
    weights.index.name = "annotator_id"
    return weights


def assign_ground_truth_by_weighted_vote(
    matrix: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    """Assign the label with the greatest total annotator weight.

    For each conversation the weights of the annotators who chose a label
    are summed with ``DataFrame.groupby``. The label with the largest sum
    wins. Two labels whose sums are equal within floating-point tolerance
    (``numpy.isclose``) stay unlabeled for adjudication.

    Parameters
    ----------
    matrix:
        Output of :func:`build_annotator_label_matrix`.
    weights:
        One positive weight per annotator column, as from
        :func:`reliability_weights_from_pairwise_cohen_kappa`.

    Returns
    -------
    pandas.DataFrame
        Same shape as :func:`assign_ground_truth_by_majority_vote`, with
        ``supporting_weight`` in place of ``supporting_votes``.

    Raises
    ------
    KeyError
        If any annotator column has no weight.
    """
    missing = [annotator for annotator in matrix.columns if annotator not in weights.index]
    if missing:
        raise KeyError(f"No reliability weight for annotator(s): {missing}")

    ground_truth: list[object] = []
    support: list[float] = []
    needs_adjudication: list[bool] = []
    for _, row in matrix.iterrows():
        totals = _weighted_label_totals(row, weights)
        if totals.empty:
            ground_truth.append(pd.NA)
            support.append(np.nan)
            needs_adjudication.append(True)
            continue
        top = totals.max()
        winners = totals.index[np.isclose(totals.to_numpy(), top)].tolist()
        if len(winners) == 1:
            ground_truth.append(winners[0])
            support.append(float(top))
            needs_adjudication.append(False)
        else:
            ground_truth.append(pd.NA)
            support.append(np.nan)
            needs_adjudication.append(True)

    annotator_count = matrix.notna().sum(axis=1)
    decided = pd.Series(
        (not tied for tied in needs_adjudication),
        index=matrix.index,
    )
    resolution = _resolution_labels(
        annotator_count=annotator_count,
        decided=decided,
        decided_name="weighted_vote",
    )
    distinct_label_count = matrix.apply(
        lambda row: int(row.dropna().nunique()), axis=1
    )
    return _ground_truth_frame(
        matrix=matrix,
        ground_truth_label=pd.Series(ground_truth, index=matrix.index, dtype="object"),
        resolution=resolution,
        support=pd.Series(support, index=matrix.index, dtype="float"),
        support_column="supporting_weight",
        annotator_count=annotator_count,
        distinct_label_count=distinct_label_count,
        needs_adjudication=pd.Series(needs_adjudication, index=matrix.index),
    )


def list_items_annotators_disagreed_on(matrix: pd.DataFrame) -> pd.DataFrame:
    """List conversations where the submitted primary labels are not identical.

    These are the items the assignment asks the team to keep for adjudication.
    Single-annotator rows are absent, because there is no second label to
    disagree with.

    Parameters
    ----------
    matrix:
        Output of :func:`build_annotator_label_matrix`.

    Returns
    -------
    pandas.DataFrame
        Disagreed rows of ``matrix``, plus ``distinct_label_count`` and a
        ``vote_summary`` such as ``Frustrated or complaining (4); Escalatory
        or requesting immediate resolution (2)``.
    """
    distinct_label_count = matrix.apply(
        lambda row: int(row.dropna().nunique()), axis=1
    )
    disagreed = matrix.loc[distinct_label_count > 1].copy()
    disagreed.insert(0, "distinct_label_count", distinct_label_count.loc[disagreed.index])
    disagreed.insert(1, "vote_summary", disagreed.apply(_vote_summary, axis=1))
    return disagreed.sort_index()


def plot_ground_truth_label_distribution(
    majority_vote: pd.DataFrame,
    weighted_vote: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save a grouped bar chart of resolved ground-truth label counts.

    Both methods are drawn on the tone order from the annotation guidelines.
    Unresolved ties are omitted from the bars; their counts are printed under
    the chart so the totals stay interpretable.

    Parameters
    ----------
    majority_vote:
        Output of :func:`assign_ground_truth_by_majority_vote`.
    weighted_vote:
        Output of :func:`assign_ground_truth_by_weighted_vote`.
    output_path:
        Destination PNG path. Parent directories are created.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    plot_rows = []
    unresolved = {}
    for method_name, table in (
        ("Majority vote", majority_vote),
        ("Reliability-weighted vote", weighted_vote),
    ):
        resolved = table["ground_truth_label"].dropna()
        counts = (
            resolved.value_counts()
            .reindex(TONE_LABELS, fill_value=0)
            .rename_axis("label")
            .reset_index(name="conversations")
        )
        counts["method"] = f"{method_name} (n={len(resolved)})"
        plot_rows.append(counts)
        unresolved[method_name] = int(table["needs_adjudication"].sum())

    plot_data = pd.concat(plot_rows, ignore_index=True)
    method_order = list(dict.fromkeys(plot_data["method"]))

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(11, 6.2))
    sns.barplot(
        data=plot_data,
        y="label",
        x="conversations",
        hue="method",
        order=list(TONE_LABELS),
        hue_order=method_order,
        palette=list(METHOD_COLORS),
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=9)

    ax.set_title("Ground-truth customer tone labels")
    ax.set_xlabel("Conversations")
    ax.set_ylabel("Tone label")
    ax.set_xlim(0, plot_data["conversations"].max() * 1.12)
    ax.legend(title="Method", loc="lower right")
    fig.text(
        0.01,
        0.01,
        "Unresolved ties left off this chart: "
        f"{unresolved['Majority vote']} under majority vote, "
        f"{unresolved['Reliability-weighted vote']} under reliability-weighted vote. "
        "Those conversations are listed for adjudication.",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> None:
    """Load submissions, then write agreement, ground truth, and the label distribution."""
    try:
        from .agreement import plot_annotator_agreement, summarize_annotator_agreement
    except ImportError:
        # Allows `python src/annotation_analysis/determine_ground_truth.py`.
        from agreement import plot_annotator_agreement, summarize_annotator_agreement
    try:
        from .analyze_distribution import plot_label_amounts, summarize_label_distribution
    except ImportError:
        from analyze_distribution import plot_label_amounts, summarize_label_distribution

    parser = argparse.ArgumentParser(
        description=(
            "Compute annotator agreement, assign ground-truth tone labels, "
            "and test whether the label distribution is balanced."
        )
    )
    parser.add_argument(
        "--submissions",
        type=Path,
        default=DEFAULT_SUBMISSIONS_DIR,
        help="Directory of annotator submission folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the CSV tables and the PNG chart.",
    )
    args = parser.parse_args(argv)

    annotations = load_primary_tone_annotations(args.submissions)
    matrix = build_annotator_label_matrix(annotations)
    agreement_summary, pairwise = summarize_annotator_agreement(annotations, matrix)
    weights = reliability_weights_from_pairwise_cohen_kappa(matrix)
    majority = assign_ground_truth_by_majority_vote(matrix)
    weighted = assign_ground_truth_by_weighted_vote(matrix, weights)
    disagreements = list_items_annotators_disagreed_on(matrix)
    # Weighted vote labels every conversation, so it is the distribution
    # drawn for step 4. Majority vote is tested too and written beside it.
    distribution, balance = summarize_label_distribution(
        {
            "Majority vote": majority,
            "Reliability-weighted vote": weighted,
        },
        TONE_LABELS,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agreement_summary.to_csv(output_dir / "annotator_agreement_summary.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_annotator_agreement.csv", index=False)
    majority.to_csv(output_dir / "ground_truth_majority_vote.csv")
    weighted.to_csv(output_dir / "ground_truth_weighted_vote.csv")
    disagreements.to_csv(output_dir / "adjudication_disagreements.csv")
    weights.to_csv(output_dir / "annotator_reliability_weights.csv", header=True)
    distribution.to_csv(output_dir / "label_distribution.csv", index=False)
    balance.to_csv(output_dir / "label_balance.csv", index=False)
    agreement_chart = plot_annotator_agreement(
        agreement_summary,
        pairwise,
        output_dir / "annotator_agreement.png",
    )
    label_chart = plot_ground_truth_label_distribution(
        majority,
        weighted,
        output_dir / "ground_truth_label_distribution.png",
    )
    distribution_chart = plot_label_amounts(
        distribution,
        balance,
        method="Reliability-weighted vote",
        output_path=output_dir / "label_distribution.png",
    )

    print(f"Conversations: {len(matrix)}")
    print(f"Annotations: {len(annotations)}")
    print("Agreement:")
    print(agreement_summary.to_string(index=False))
    print(f"Agreement chart: {agreement_chart}")
    print("Reliability weights (mean pairwise Cohen's kappa):")
    print(weights.round(3).to_string())
    print("Majority vote:")
    print(majority["resolution"].value_counts().to_string())
    print("Weighted vote:")
    print(weighted["resolution"].value_counts().to_string())
    print(f"Disagreements for adjudication: {len(disagreements)}")
    print(f"Ground-truth chart: {label_chart}")
    print("Label distribution:")
    print(
        distribution.loc[distribution["method"] == "Reliability-weighted vote"]
        .to_string(index=False)
    )
    print("Class balance:")
    print(balance.to_string(index=False))
    print(f"Label distribution chart: {distribution_chart}")


def _find_annotation_file(annotator_dir: Path) -> Path | None:
    """Return one JSONL for this folder, preferring the top-level copy.

    Annotators 01–06 have ``annotations.jsonl`` at the top of the folder and
    the same bytes under Potato's export path. Annotator 07 submitted only
    the export path. The first hit is the one that is read.
    """
    for relative in (
        "annotations.jsonl",
        "exports/jsonl/annotations.jsonl",
    ):
        path = annotator_dir / relative
        if path.is_file():
            return path
    return None


def _read_annotator_file(annotator_id: str, annotation_file: Path) -> list[dict[str, str]]:
    """Parse one annotator's JSONL into primary-label rows."""
    rows: list[dict[str, str]] = []
    with annotation_file.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            label = _primary_label(record)
            if label is None:
                continue
            rows.append(
                {
                    "instance_id": str(record["instance_id"]),
                    "annotator_id": annotator_id,
                    "label": label,
                }
            )
    return rows


def _primary_label(record: dict) -> str | None:
    """Return the selected ``conversation_sense`` label, if the row has one."""
    field = (record.get("labels") or {}).get(PRIMARY_LABEL_FIELD) or {}
    if not isinstance(field, dict) or not field:
        return None
    return next(iter(field))


def _highest_vote_count(row: pd.Series) -> float:
    """Return how many annotators chose the most common label on this row."""
    counts = row.dropna().value_counts()
    if counts.empty:
        return np.nan
    return int(counts.iloc[0])


def _vote_summary(row: pd.Series) -> str:
    """Format a row's label counts, most common label first."""
    label_columns = [column for column in row.index if column in ("distinct_label_count", "vote_summary")]
    counts = row.drop(labels=label_columns, errors="ignore").dropna().value_counts()
    return "; ".join(f"{label} ({count})" for label, count in counts.items())


def _weighted_label_totals(row: pd.Series, weights: pd.Series) -> pd.Series:
    """Sum reliability weights for each label present on one conversation."""
    observed = row.dropna()
    if observed.empty:
        return pd.Series(dtype=float)
    totals = (
        pd.DataFrame(
            {
                "label": observed.to_numpy(),
                "weight": weights.loc[observed.index].to_numpy(dtype=float),
            }
        )
        .groupby("label", sort=False)["weight"]
        .sum()
    )
    return totals


def _resolution_labels(
    annotator_count: pd.Series,
    decided: pd.Series,
    decided_name: str,
) -> pd.Series:
    """Name how each row received, or did not receive, a ground-truth label."""
    resolution = pd.Series(decided_name, index=annotator_count.index, dtype="object")
    resolution = resolution.mask(annotator_count <= 1, "single_annotator")
    resolution = resolution.mask(~decided.astype(bool), "unresolved_tie")
    resolution = resolution.mask(annotator_count == 0, "unlabeled")
    return resolution


def _ground_truth_frame(
    matrix: pd.DataFrame,
    ground_truth_label: pd.Series,
    resolution: pd.Series,
    support: pd.Series,
    support_column: str,
    annotator_count: pd.Series,
    distinct_label_count: pd.Series,
    needs_adjudication: pd.Series,
) -> pd.DataFrame:
    """Assemble the shared ground-truth table."""
    frame = pd.DataFrame(
        {
            "ground_truth_label": ground_truth_label,
            "resolution": resolution,
            support_column: support,
            "annotator_count": annotator_count.astype(int),
            "distinct_label_count": distinct_label_count.astype(int),
            "needs_adjudication": needs_adjudication.astype(bool),
        },
        index=matrix.index,
    )
    return frame.sort_index()


if __name__ == "__main__":
    main()
