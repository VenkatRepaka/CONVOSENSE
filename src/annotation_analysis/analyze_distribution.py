"""Label distribution and class balance for the ground-truth tones.

Implements step 4, "Analyze Data," of
``docs/ARI410-510_Project_Leaderboard.pdf``: how many conversations carry
each tone label, and whether those classes are balanced.

Counts and shares come from ``Series.value_counts``. Balance is a
chi-square test (``scipy.stats.chisquare``) of the observed counts against
an equal share of each class. A p-value below 0.05 means the classes are
imbalanced.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chisquare

# Reject equal class sizes at the conventional 5% level.
BALANCE_SIGNIFICANCE_LEVEL = 0.05

BAR_COLOR = "#0072B2"
EQUAL_SHARE_COLOR = "#E69F00"


def label_counts(labels: pd.Series, label_order: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Count conversations and the share of the dataset for each tone.

    Unlabeled rows are omitted. Every label in ``label_order`` is kept, with
    a count of zero when it does not occur.

    Parameters
    ----------
    labels:
        Ground-truth tone for each conversation. Missing values are unlabeled.
    label_order:
        Display order, calm to demanding.

    Returns
    -------
    pandas.DataFrame
        Columns ``label``, ``conversations``, and ``share``. ``share`` is the
        proportion from ``value_counts(normalize=True)``.
    """
    observed = labels.dropna()
    counts = observed.value_counts().reindex(label_order, fill_value=0)
    shares = observed.value_counts(normalize=True).reindex(label_order, fill_value=0.0)
    return pd.DataFrame(
        {
            "label": list(label_order),
            "conversations": counts.to_numpy(dtype=int),
            "share": shares.to_numpy(dtype=float),
        }
    )


def test_class_balance(counts: pd.DataFrame) -> dict[str, object]:
    """Test whether the class counts match an equal share of each label.

    ``scipy.stats.chisquare`` compares the observed counts with the same
    expected count for every class. The classes are balanced when the
    p-value is at least 0.05.

    Parameters
    ----------
    counts:
        Output of :func:`label_counts`.

    Returns
    -------
    dict
        Chi-square statistic, p-value, the equal-share count, the largest
        and smallest classes, and ``classes_are_balanced``.
    """
    observed = counts["conversations"].to_numpy(dtype=float)
    if observed.sum() == 0 or len(observed) < 2:
        raise ValueError("Class balance needs counts for at least two labels")
    expected = np.full(observed.shape, observed.sum() / observed.size)
    test = chisquare(observed, f_exp=expected)
    largest = counts.loc[counts["conversations"].idxmax()]
    smallest = counts.loc[counts["conversations"].idxmin()]
    p_value = float(test.pvalue)
    return {
        "conversations": int(observed.sum()),
        "classes": int(len(observed)),
        "chi_square": float(test.statistic),
        "p_value": p_value,
        "equal_share_per_class": float(expected[0]),
        "largest_class": largest["label"],
        "largest_class_conversations": int(largest["conversations"]),
        "smallest_class": smallest["label"],
        "smallest_class_conversations": int(smallest["conversations"]),
        "classes_are_balanced": bool(p_value >= BALANCE_SIGNIFICANCE_LEVEL),
    }


def summarize_label_distribution(
    ground_truth_tables: dict[str, pd.DataFrame],
    label_order: tuple[str, ...] | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Distribution and balance test for each ground-truth method.

    Parameters
    ----------
    ground_truth_tables:
        Method name to a ground-truth table that has ``ground_truth_label``.
    label_order:
        Display order, calm to demanding.

    Returns
    -------
    tuple
        ``(distribution, balance)``. ``distribution`` stacks one count table
        per method. ``balance`` has one row per method from
        :func:`test_class_balance`.
    """
    distributions: list[pd.DataFrame] = []
    balance_rows: list[dict[str, object]] = []
    for method, table in ground_truth_tables.items():
        counts = label_counts(table["ground_truth_label"], label_order)
        counts.insert(0, "method", method)
        distributions.append(counts)
        balance = test_class_balance(counts)
        balance["method"] = method
        balance_rows.append(balance)
    distribution = pd.concat(distributions, ignore_index=True)
    balance = pd.DataFrame(balance_rows)
    balance_columns = ["method", *[column for column in balance.columns if column != "method"]]
    return distribution, balance[balance_columns]


def plot_label_amounts(
    distribution: pd.DataFrame,
    balance: pd.DataFrame,
    method: str,
    output_path: Path,
) -> Path:
    """Save a bar chart of how many conversations carry each tone.

    The dashed line is the equal share used by the chi-square test. The
    title states whether the classes are balanced.

    Parameters
    ----------
    distribution:
        Output of :func:`summarize_label_distribution`.
    balance:
        Balance tests from the same call.
    method:
        Which ground-truth method to draw.
    output_path:
        Destination PNG. Parent directories are created.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    counts = distribution.loc[distribution["method"] == method].reset_index(drop=True)
    result = balance.loc[balance["method"] == method].iloc[0]
    verdict = "balanced" if result["classes_are_balanced"] else "imbalanced"

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.barplot(
        data=counts,
        y="label",
        x="conversations",
        order=list(counts["label"]),
        color=BAR_COLOR,
        ax=ax,
    )
    annotations = [
        f"{int(row.conversations)} ({row.share:.1%})" for row in counts.itertuples(index=False)
    ]
    for container in ax.containers:
        ax.bar_label(container, labels=annotations, padding=3, fontsize=9)

    equal_share = float(result["equal_share_per_class"])
    ax.axvline(
        equal_share,
        color=EQUAL_SHARE_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"Equal share ({equal_share:.1f})",
    )
    ax.set_xlim(0, counts["conversations"].max() * 1.22)
    ax.set_xlabel("Conversations")
    ax.set_ylabel("Tone label")
    ax.set_title(f"Classes are {verdict}")
    ax.legend(loc="lower right")
    fig.suptitle(f"Amount of data with each ground-truth tone ({method})", fontsize=14)
    fig.text(
        0.01,
        0.01,
        (
            f"{int(result['conversations'])} labeled conversations. "
            f"Chi-square test against an equal share of each class: "
            f"chi-square = {float(result['chi_square']):.1f}, "
            f"p {_format_p_value(float(result['p_value']))}."
        ),
        fontsize=9,
        color="#333333",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _format_p_value(p_value: float) -> str:
    """Format a chi-square p-value for the chart footnote."""
    if p_value < 0.001:
        return "< 0.001"
    return f"= {p_value:.3f}"
