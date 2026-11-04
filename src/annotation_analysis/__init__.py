"""Agreement and ground-truth labels for the customer-tone annotations.

From ``docs/ARI410-510_Project_Leaderboard.pdf``:

* Step 2.2, "Compute Agreement Between Annotators": Cohen's kappa, Fleiss'
  kappa, Krippendorff's alpha, and percentage agreement.
* Step 3, "Determine Ground Truth Labels": majority vote, reliability-weighted
  vote, and the items left for adjudication.
* Step 4, "Analyze Data": how many conversations carry each tone, and
  whether those classes are balanced.
"""

from .agreement import (
    calculate_fleiss_kappa,
    calculate_krippendorffs_alpha,
    pairwise_annotator_agreement,
    plot_annotator_agreement,
    summarize_annotator_agreement,
)
from .analyze_distribution import (
    label_counts,
    plot_label_amounts,
    summarize_label_distribution,
    test_class_balance,
)
from .determine_ground_truth import (
    TONE_LABELS,
    assign_ground_truth_by_majority_vote,
    assign_ground_truth_by_weighted_vote,
    build_annotator_label_matrix,
    list_items_annotators_disagreed_on,
    load_primary_tone_annotations,
    plot_ground_truth_label_distribution,
    reliability_weights_from_pairwise_cohen_kappa,
)

__all__ = [
    "TONE_LABELS",
    "assign_ground_truth_by_majority_vote",
    "assign_ground_truth_by_weighted_vote",
    "build_annotator_label_matrix",
    "calculate_fleiss_kappa",
    "calculate_krippendorffs_alpha",
    "label_counts",
    "list_items_annotators_disagreed_on",
    "load_primary_tone_annotations",
    "pairwise_annotator_agreement",
    "plot_annotator_agreement",
    "plot_ground_truth_label_distribution",
    "plot_label_amounts",
    "reliability_weights_from_pairwise_cohen_kappa",
    "summarize_annotator_agreement",
    "summarize_label_distribution",
    "test_class_balance",
]
