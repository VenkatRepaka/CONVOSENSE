"""Macro-averaged F1 for the ConvoSense customer-tone competition.

Paste this file into Kaggle's metric editor. The entry point Kaggle calls
is ``score``. Rows are already aligned on the id column before the call.

The five tones are imbalanced (step 4: chi-square p < 0.001, and
"Frustrated or complaining" is 43.8% of the labeled conversations). Macro
F1 is the unweighted mean of the per-class F1 scores, so each tone counts
the same. The score uses ``sklearn.metrics.f1_score``.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score

# Same strings as Customer Tone Annotation Guidelines.md. A submission has
# to match one of these exactly.
TONE_LABELS = (
    "Appreciative or positive",
    "Neutral or informational",
    "Help-seeking or asking a question",
    "Frustrated or complaining",
    "Escalatory or requesting immediate resolution",
)

LABEL_COLUMN = "label"


class ParticipantVisibleError(Exception):
    """Error text Kaggle shows to the person who submitted."""


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
) -> float:
    """Return the macro-averaged F1 of ``submission`` against ``solution``.

    Parameters
    ----------
    solution:
        Ground truth, already aligned to ``submission``. Must contain
        ``row_id_column_name`` and ``label``. Other columns, including
        ``Usage``, are ignored.
    submission:
        Predictions with the same id column and a ``label`` column.
    row_id_column_name:
        Id column Kaggle used to align the two frames.

    Returns
    -------
    float
        Unweighted mean of the per-class F1 scores. Classes absent from
        this solution slice are left out of the mean.

    Examples
    --------
    >>> import pandas as pd
    >>> labels = [
    ...     "Appreciative or positive",
    ...     "Neutral or informational",
    ...     "Help-seeking or asking a question",
    ...     "Frustrated or complaining",
    ...     "Escalatory or requesting immediate resolution",
    ... ]
    >>> solution = pd.DataFrame({"id": [1, 2, 3, 4, 5], "label": labels})
    >>> score(solution, solution.copy(), "id")
    1.0
    """
    solution = solution.copy()
    submission = submission.copy()
    for frame, name in ((solution, "solution"), (submission, "submission")):
        if row_id_column_name not in frame.columns:
            raise ParticipantVisibleError(
                f"The {name} file is missing the {row_id_column_name} column."
            )
        if LABEL_COLUMN not in frame.columns:
            raise ParticipantVisibleError(
                f"The {name} file is missing the {LABEL_COLUMN} column."
            )
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if len(submission) != len(solution):
        raise ParticipantVisibleError(
            f"Submission has {len(submission)} rows; expected {len(solution)}."
        )

    y_true = solution[LABEL_COLUMN].astype(str).str.strip()
    y_pred = submission[LABEL_COLUMN].astype(str).str.strip()
    unknown_truth = sorted(set(y_true) - set(TONE_LABELS))
    if unknown_truth:
        raise ValueError(
            "Solution contains labels outside the five tones: "
            + ", ".join(unknown_truth)
        )
    unknown_pred = sorted(set(y_pred) - set(TONE_LABELS))
    if unknown_pred:
        allowed = "; ".join(TONE_LABELS)
        raise ParticipantVisibleError(
            "Submission contains labels that are not one of the five tones: "
            + ", ".join(unknown_pred)
            + f". Allowed labels: {allowed}."
        )

    present = [label for label in TONE_LABELS if (y_true == label).any()]
    if not present:
        raise ParticipantVisibleError("Solution has no labels to score.")
    return float(
        f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0)
    )
