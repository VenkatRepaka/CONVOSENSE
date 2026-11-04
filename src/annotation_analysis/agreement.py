"""Annotator agreement for the primary customer-tone label.

Implements step 2.2, "Compute Agreement Between Annotators," of
``docs/ARI410-510_Project_Leaderboard.pdf``:

* **Cohen's kappa** for every annotator pair, via
  ``sklearn.metrics.cohen_kappa_score``.
* **Fleiss' kappa** for the whole group, via
  ``statsmodels.stats.inter_rater.fleiss_kappa``. That implementation
  requires every conversation to have the same number of raters, so it is
  computed on the conversations every annotator labeled.
* **Krippendorff's alpha** (nominal), via ``simpledorff``. Alpha keeps
  conversations that some annotators left unlabeled.
* **Percentage agreement**, the fraction of shared conversations on which a
  pair chose the same label, via ``sklearn.metrics.accuracy_score``. The
  reported value is the mean of those pair scores. The assignment asks for
  this only alongside a chance-corrected metric, so the chart draws it with
  the three kappa-style scores.

Agreement uses conversations labeled by at least two annotators. A
conversation with one label cannot show agreement or disagreement.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from simpledorff import calculate_krippendorffs_alpha_for_df
from simpledorff.metrics import nominal_metric
from sklearn.metrics import accuracy_score, cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

# Chance-corrected scores, then raw agreement. Matches the assignment list.
METRIC_ORDER = (
    "Cohen's kappa",
    "Fleiss' kappa",
    "Krippendorff's alpha",
    "Percentage agreement",
)

# Blue for chance-corrected scores, orange for raw agreement.
METRIC_COLORS = {
    "Cohen's kappa": "#0072B2",
    "Fleiss' kappa": "#0072B2",
    "Krippendorff's alpha": "#0072B2",
    "Percentage agreement": "#E69F00",
}


def pairwise_annotator_agreement(matrix: pd.DataFrame) -> pd.DataFrame:
    """Cohen's kappa and percentage agreement for every annotator pair.

    Each pair is scored on the conversations both annotators labeled.
    Percentage agreement is ``sklearn.metrics.accuracy_score``: the fraction
    of those conversations where the two labels match. Cohen's kappa is
    ``sklearn.metrics.cohen_kappa_score`` on the same rows.

    Parameters
    ----------
    matrix:
        Conversations by annotators, from
        ``build_annotator_label_matrix``. Empty cells are unlabeled.

    Returns
    -------
    pandas.DataFrame
        One row per pair. Columns: ``annotator_a``, ``annotator_b``,
        ``conversations``, ``percentage_agreement``, ``cohens_kappa``.
    """
    rows: list[dict[str, object]] = []
    for left, right in combinations(matrix.columns, 2):
        pair = matrix[[left, right]].dropna()
        if len(pair) < 2:
            continue
        rows.append(
            {
                "annotator_a": left,
                "annotator_b": right,
                "conversations": int(len(pair)),
                "percentage_agreement": float(accuracy_score(pair[left], pair[right])),
                "cohens_kappa": float(cohen_kappa_score(pair[left], pair[right])),
            }
        )
    if not rows:
        raise ValueError("No annotator pair shares two or more conversations")
    return pd.DataFrame(rows)


def calculate_fleiss_kappa(matrix: pd.DataFrame) -> tuple[float, int, int]:
    """Fleiss' kappa on conversations labeled by every annotator.

    ``statsmodels.stats.inter_rater.aggregate_raters`` builds the
    subject-by-category count table, and ``fleiss_kappa`` scores it.
    Rows with a missing annotator are dropped first, because that Fleiss
    implementation requires the same rater count on every row.

    Parameters
    ----------
    matrix:
        Conversations by annotators. Empty cells are unlabeled.

    Returns
    -------
    tuple
        ``(kappa, n_conversations, n_annotators)``.

    Raises
    ------
    ValueError
        If no conversation was labeled by every remaining annotator.
    """
    labeled = matrix.dropna(axis=1, how="all")
    complete = labeled.dropna(axis=0, how="any")
    if complete.shape[0] == 0 or complete.shape[1] < 2:
        raise ValueError(
            "Fleiss' kappa needs at least two annotators on the same conversations"
        )
    count_table, _categories = aggregate_raters(complete.to_numpy())
    return float(fleiss_kappa(count_table, method="fleiss")), int(complete.shape[0]), int(
        complete.shape[1]
    )


def calculate_krippendorffs_alpha(annotations: pd.DataFrame) -> tuple[float, int]:
    """Nominal Krippendorff's alpha, keeping partially labeled conversations.

    ``simpledorff.calculate_krippendorffs_alpha_for_df`` with
    ``nominal_metric``. Conversations with a single label are left out,
    because they add no annotator pair. A conversation that some annotators
    skipped stays in, which is the missing-data case alpha is for.

    Parameters
    ----------
    annotations:
        Long table with ``instance_id``, ``annotator_id``, and ``label``,
        from ``load_primary_tone_annotations``.

    Returns
    -------
    tuple
        ``(alpha, n_conversations)`` for conversations with two or more labels.

    Raises
    ------
    ValueError
        If no conversation has two or more labels.
    """
    ratings_per_item = annotations.groupby("instance_id")["annotator_id"].transform(
        "nunique"
    )
    multi = annotations.loc[ratings_per_item >= 2]
    n_conversations = int(multi["instance_id"].nunique())
    if n_conversations == 0:
        raise ValueError("Krippendorff's alpha needs an item with two or more labels")
    alpha = calculate_krippendorffs_alpha_for_df(
        multi,
        experiment_col="instance_id",
        annotator_col="annotator_id",
        class_col="label",
        metric_fn=nominal_metric,
    )
    return float(alpha), n_conversations


def summarize_annotator_agreement(
    annotations: pd.DataFrame,
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect the four assignment metrics and the pairwise table.

    Parameters
    ----------
    annotations:
        Long annotation table from ``load_primary_tone_annotations``.
    matrix:
        Wide label matrix from ``build_annotator_label_matrix``.

    Returns
    -------
    tuple
        ``(summary, pairwise)``. ``summary`` has one row per metric with
        ``metric``, ``value``, and ``sample``. ``pairwise`` is the output of
        :func:`pairwise_annotator_agreement`. Cohen's kappa and percentage
        agreement in the summary are the unweighted means of the pair rows.
    """
    pairwise = pairwise_annotator_agreement(matrix)
    fleiss_value, fleiss_items, fleiss_raters = calculate_fleiss_kappa(matrix)
    alpha, alpha_items = calculate_krippendorffs_alpha(annotations)
    pair_sizes = sorted(pairwise["conversations"].unique())
    pair_size_text = " or ".join(str(size) for size in pair_sizes)
    n_pairs = len(pairwise)
    summary = pd.DataFrame(
        [
            {
                "metric": "Cohen's kappa",
                "value": float(pairwise["cohens_kappa"].mean()),
                "sample": (
                    f"Mean of {n_pairs} annotator pairs, "
                    f"each scored on {pair_size_text} shared conversations"
                ),
            },
            {
                "metric": "Fleiss' kappa",
                "value": fleiss_value,
                "sample": (
                    f"{fleiss_items} conversations labeled by all {fleiss_raters} annotators"
                ),
            },
            {
                "metric": "Krippendorff's alpha",
                "value": alpha,
                "sample": (
                    f"{alpha_items} conversations with two or more labels; "
                    "missing labels are kept"
                ),
            },
            {
                "metric": "Percentage agreement",
                "value": float(pairwise["percentage_agreement"].mean()),
                "sample": (
                    f"Mean of {n_pairs} annotator pairs, "
                    f"each scored on {pair_size_text} shared conversations"
                ),
            },
        ]
    )
    return summary, pairwise


def plot_annotator_agreement(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save summary bars and a pairwise Cohen's kappa heatmap.

    Percentage agreement is drawn in orange. The three chance-corrected
    scores are blue. The heatmap leaves the diagonal blank.

    Parameters
    ----------
    summary:
        Output of :func:`summarize_annotator_agreement`.
    pairwise:
        Pair table from the same call.
    output_path:
        Destination PNG. Parent directories are created.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.8),
        gridspec_kw={"width_ratios": [1.05, 1.15]},
    )

    order = [metric for metric in METRIC_ORDER if metric in set(summary["metric"])]
    colors = [METRIC_COLORS[metric] for metric in order]
    sns.barplot(
        data=summary,
        y="metric",
        x="value",
        order=order,
        hue="metric",
        hue_order=order,
        palette=colors,
        legend=False,
        ax=axes[0],
    )
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    lower = min(0.0, float(summary["value"].min()))
    axes[0].set_xlim(lower - 0.05, 1.08)
    axes[0].set_xlabel("Score (1 is perfect agreement)")
    axes[0].set_ylabel("")
    axes[0].set_title("Summary")

    heatmap = _pairwise_kappa_matrix(pairwise)
    sns.heatmap(
        heatmap,
        mask=np.eye(len(heatmap), dtype=bool),
        vmin=min(0.0, float(np.nanmin(heatmap.to_numpy()))),
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Cohen's kappa"},
        ax=axes[1],
    )
    axes[1].set_title("Pairwise Cohen's kappa")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    fig.suptitle("Annotator agreement on the primary tone label", fontsize=14)
    fig.text(
        0.01,
        0.01,
        _agreement_footnote(summary),
        fontsize=8,
        color="#333333",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.20, 1, 0.94))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _pairwise_kappa_matrix(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Square annotator matrix of Cohen's kappa, diagonal left empty."""
    annotators = sorted(set(pairwise["annotator_a"]) | set(pairwise["annotator_b"]))
    heatmap = pd.DataFrame(np.nan, index=annotators, columns=annotators, dtype=float)
    for row in pairwise.itertuples(index=False):
        heatmap.loc[row.annotator_a, row.annotator_b] = row.cohens_kappa
        heatmap.loc[row.annotator_b, row.annotator_a] = row.cohens_kappa
    return heatmap


def _agreement_footnote(summary: pd.DataFrame) -> str:
    """One line per metric describing which conversations entered the score."""
    by_metric = summary.set_index("metric")["sample"]
    lines = [
        "Blue bars subtract chance agreement. The orange bar is the raw share of matching labels.",
    ]
    lines.extend(f"{metric}: {by_metric[metric]}." for metric in METRIC_ORDER)
    return "\n".join(lines)
