"""Macro-averaged F1 for the ConvoSense customer-tone competition.

Paste this file into Kaggle's metric editor. The entry point Kaggle calls
is ``score``. Rows are already aligned on the id column before the call.

Kaggle rejected tone names in the upload files, so the Kaggle files use the
numbers 1 to 5 in guideline order (``competition/kaggle_upload/label_map.csv``).
This scorer accepts either the numbers or the tone names, and maps numbers to
names before scoring.

The five tones are imbalanced (step 4: chi-square p < 0.001, and
"Frustrated or complaining" is 42.9% of the labeled conversations). Macro
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

# Kaggle upload files use these numbers in place of the tone names.
NUMBER_TO_TONE = {str(number): tone for number, tone in enumerate(TONE_LABELS, start=1)}


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

    y_true = _tone_names(solution[LABEL_COLUMN])
    y_pred = _tone_names(submission[LABEL_COLUMN])
    unknown_truth = sorted(set(y_true) - set(TONE_LABELS))
    if unknown_truth:
        raise ValueError(
            "Solution contains labels outside the five tones: "
            + ", ".join(unknown_truth)
        )
    unknown_pred = sorted(set(y_pred) - set(TONE_LABELS))
    if unknown_pred:
        allowed = "1 to 5, or the tone names: " + "; ".join(TONE_LABELS)
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


def _tone_names(labels: pd.Series) -> pd.Series:
    """Map the numbers 1 to 5 to tone names; leave anything else unchanged.

    A float such as ``3.0`` is read as 3, because Kaggle may load an integer
    column as floats.
    """
    text = labels.astype(str).str.strip().str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    return text.map(lambda value: NUMBER_TO_TONE.get(value, value))
