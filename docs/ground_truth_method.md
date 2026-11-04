# How the ground-truth labels were calculated

This is a reference for step 3 of `docs/ARI410-510_Project_Leaderboard.pdf`, "Determine Ground Truth Labels". It explains how each conversation got its final tone label, why we chose that method, and which parts were our own design choices.

Code: `src/annotation_analysis/determine_ground_truth.py`. Run it from the repository root:

```bash
MPLCONFIGDIR=/tmp/mplconfig /opt/anaconda3/bin/python src/annotation_analysis/determine_ground_truth.py
```

## 1. The input

Six classmates labeled conversations in Potato. The label that counts is the primary field, `conversation_sense`. The optional `secondary_sense` is not a vote. Each annotator is identified by their submission folder name (`annotator_01` to `annotator_06`, written A1 to A6 below).

| | Count |
| --- | ---: |
| Labels submitted | 582 |
| Distinct conversations | 436 |
| Conversations labeled by one annotator | 406 |
| Shared conversations (labeled by 5 or 6 annotators) | 30 |
| Shared, all six annotators | 26 |
| Shared, five annotators (A5 skipped them) | 4 |
| Shared and unanimous | 5 |
| Shared with at least one disagreement | 25 |

The labels are arranged as a table with one row per conversation and one column per annotator. An empty cell means that annotator did not label that conversation.

## 2. The three rules

Every conversation is settled by exactly one of these rules.

| Rule | Applies to | Result |
| --- | --- | --- |
| Single annotator | 406 conversations | That annotator's label is the ground truth |
| Majority vote | 30 shared conversations | 28 decided, 2 ties left unlabeled |
| Reliability-weighted vote (used) | 30 shared conversations | 30 decided, 0 ties |

Both voting methods are written out. The competition files use the weighted vote.

## 3. Majority vote

For each shared conversation, count how many annotators chose each tone. The tone with the highest count wins.

- `pandas.DataFrame.mode` returns every tone tied for the top count. If more than one tone is tied, the conversation stays unlabeled and is marked `unresolved_tie`.
- Ties are never broken alphabetically.

Two conversations tied 3–3:

| Conversation | Votes |
| --- | --- |
| 1003675 | Appreciative or positive 3, Help-seeking or asking a question 3 |
| 1257160 | Frustrated or complaining 3, Escalatory or requesting immediate resolution 3 |

## 4. Reliability-weighted vote

This works in two steps. First each annotator gets a weight, once for the whole dataset. Then each conversation is decided by adding up the weights behind each tone.

### 4.1 Cohen's kappa for every pair of annotators

For each of the 15 annotator pairs, take the conversations both of them labeled: 30 items, or 26 for pairs that include A5. Cohen's kappa measures how much the two agree beyond chance:

κ = (p_o − p_e) / (1 − p_e)

- **p_o** is the fraction of shared items where the two chose the same tone.
- **p_e** is the agreement expected by chance, given how often each annotator uses each tone.
- 1 means perfect agreement, and 0 means no better than chance.

This is calculated with `sklearn.metrics.cohen_kappa_score`.

**Example, A5 and A6:** they agreed on 21 of 26 items, so p_o = 0.808. Chance agreement for their label habits is about 0.216. So κ = (0.808 − 0.216) / (1 − 0.216) = 0.755, the highest pair.

All 15 pairs:

| | A1 | A2 | A3 | A4 | A5 | A6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **A1** | | 0.397 | 0.311 | 0.475 | 0.264 | 0.292 |
| **A2** | 0.397 | | 0.561 | 0.363 | 0.402 | 0.495 |
| **A3** | 0.311 | 0.561 | | 0.478 | 0.610 | 0.455 |
| **A4** | 0.475 | 0.363 | 0.478 | | 0.365 | 0.289 |
| **A5** | 0.264 | 0.402 | 0.610 | 0.365 | | 0.755 |
| **A6** | 0.292 | 0.495 | 0.455 | 0.289 | 0.755 | |

### 4.2 Each annotator's weight is the average of their five kappas

Each annotator has five kappa scores, one per partner (one row of the table above). Their weight is the average: add the five and divide by 5.

**A1:** (0.397 + 0.311 + 0.475 + 0.264 + 0.292) ÷ 5 = 1.739 ÷ 5 = **0.348**

**A3:** (0.311 + 0.561 + 0.478 + 0.610 + 0.455) ÷ 5 = 2.415 ÷ 5 = **0.483**

| Annotator | Weight |
| --- | ---: |
| A1 | 0.348 |
| A2 | 0.444 |
| A3 | 0.483 |
| A4 | 0.394 |
| A5 | 0.479 |
| A6 | 0.457 |

A higher weight means the annotator agrees more with the rest of the group, on average.

Safety rules in the code. None of them were triggered by our data:

- A weight below 0 is set to 0, so an annotator who agrees less than chance cannot push the vote the wrong way.
- An annotator with no overlapping partner gets the average of the other weights.
- If every weight were 0, everyone would get weight 1, which turns the method back into plain majority vote.
- A pair where both annotators used only one tone is skipped, because kappa is undefined there.

### 4.3 The vote

For each conversation, add up the weights of the annotators who chose each tone. The tone with the biggest total wins. Totals equal within floating-point tolerance (`numpy.isclose`) would count as a tie and stay unlabeled. That did not happen. The weights are not normalized; only their relative size matters.

**Conversation 1003675:**

| Tone | Annotators | Total weight |
| --- | --- | ---: |
| Appreciative or positive (wins) | A3 + A5 + A6 = 0.483 + 0.479 + 0.457 | **1.419** |
| Help-seeking or asking a question | A1 + A2 + A4 = 0.348 + 0.444 + 0.394 | 1.186 |

**Conversation 1257160:**

| Tone | Annotators | Total weight |
| --- | --- | ---: |
| Escalatory or requesting immediate resolution (wins) | A3 + A4 + A5 = 0.483 + 0.394 + 0.479 | **1.356** |
| Frustrated or complaining | A1 + A2 + A6 = 0.348 + 0.444 + 0.457 | 1.249 |

On the other 28 shared conversations, the weighted vote picks the same tone as majority vote. The weights range only from 0.35 to 0.48, which is too narrow to overturn a split like 4–2. In practice, the weighted vote is majority vote with a principled tie-breaker.

## 5. Why we chose the weighted vote

1. **Every row needs a label.** A Kaggle answer key cannot have blank rows. The weighted vote labels all 436 conversations; majority vote leaves two blank.
2. **A fixed rule decides the ties.** The tie-breaker was set before anyone looked at the outcome, so no team member chose an answer.
3. **It changes very little.** The weighted vote differs from majority vote on 2 of 436 labels, and with the current split both are in `validation.csv`. The test answer key is the same either way, so any fixed submission gets the same score.

### What the choice does affect

The voting method still matters, for two reasons. Both were checked by re-running the pipeline in a scratch folder.

**Validation is used for tuning and refitting.** The logistic regression's settings (C and class weight) are chosen on validation, the random dummy's seed is the one whose validation score is closest to the 20-seed mean, and the submitted logistic regression is refit on train plus validation. Changing only the two tied validation labels, with the split held fixed:

| | Weighted vote (current) | Ties given the other label | Ties dropped |
| --- | ---: | ---: | ---: |
| Random dummy seed chosen | 16 | 8 | 10 |
| Random dummy, public / private | 0.181 / 0.139 | 0.186 / 0.185 | 0.207 / 0.115 |
| Logistic regression, public / private | 0.339 / 0.377 | 0.357 / 0.348 | 0.338 / 0.377 |

Participants also train on validation, so their scores could shift the same way.

**The split depends on the labels.** The split is stratified by label, so it was built from the weighted-vote labels. "Both ties are in validation" is true for that split only. Re-running `prepare_competition_files.py` with majority-vote-based labels gives a different split:

| Tie handling | Test conversations shared with the current test set | Training conversations shared |
| --- | ---: | ---: |
| Ties given the other label | 27 of 88 | 160 of 261 |
| Ties dropped (434 conversations) | 70 of 88 | 193 of 261 |

So the voting method was settled before the split, and the split should not be rebuilt after the competition launches.

## 6. What was our choice, and what was given

| Part | Source |
| --- | --- |
| Majority vote, weighted voting, and adjudication as options | The assignment PDF, step 3 |
| "Assign weights to annotators based on experience or reliability" | The assignment PDF, step 3 |
| Measuring reliability as an annotator's mean pairwise Cohen's kappa | **Our design choice** |
| Clipping negative weights, the fallback weights, and leaving ties unlabeled | **Our conventions** |

Mean pairwise kappa is a simple, common-sense measure of reliability. It is not a named standard algorithm. Better-known alternatives:

- **Dawid–Skene:** estimates each annotator's error pattern per tone and infers the most likely true label. This is the classic statistical method for this problem.
- **MACE:** a similar model that also estimates how often each annotator is guessing.
- **Accuracy against gold labels:** the team adjudicates some items, then weights each annotator by how often they matched.
- **Agreement with the majority:** weight each annotator by how often they match the majority of the others.

These alternatives would be estimated from only 30 shared conversations, so a more complex model was not justified here.

## 7. Limitations

- **Most labels are single opinions.** 406 of 436 labels (93%) come from one annotator and were never cross-checked. Krippendorff's alpha (0.427, moderate) is our estimate of how reliable they are.
- **Reliability means agreeing with the group.** A careful annotator who notices sarcasm the others miss would get a lower weight.
- **Small sample.** Each weight rests on 26–30 items, so a few labels either way would shift it by a few hundredths.
- **The weights are circular.** They come from the same 30 conversations they help decide, because those are the only conversations with overlap.

## 8. The disagreement list

The assignment says to keep a list of the items annotators disagreed on. `outputs/adjudication_disagreements.csv` lists all 25 shared conversations without a unanimous label. Each row has every annotator's label and a vote summary, such as `Frustrated or complaining (3); Escalatory or requesting immediate resolution (3)`. Use it for hand adjudication, for revising the guidelines, or for error analysis.

The tones most often confused, counted as pairs of annotators who disagreed:

| Tone pair | Disagreeing pairs |
| --- | ---: |
| Frustrated – Escalatory | 57 |
| Appreciative – Help-seeking | 38 |
| Help-seeking – Neutral | 29 |
| Frustrated – Neutral | 24 |
| Frustrated – Help-seeking | 18 |
| Appreciative – Neutral | 12 |

## 9. Files

All of these are written by `determine_ground_truth.py` into `outputs/`.

| File | Contents | In git |
| --- | --- | --- |
| `ground_truth_weighted_vote.csv` | Final label per conversation (used for the competition), with `supporting_weight` | No: contains test labels |
| `ground_truth_majority_vote.csv` | Majority-vote label, with `unresolved_tie` on the two ties | No: contains test labels |
| `adjudication_disagreements.csv` | The 25 disagreed conversations | No: contains test labels |
| `annotator_reliability_weights.csv` | The six weights | Yes |
| `pairwise_annotator_agreement.csv` | Cohen's kappa and percentage agreement for each pair | Yes |
| `ground_truth_label_distribution.png` | Label counts under both methods | Yes |

Code locations in `src/annotation_analysis/determine_ground_truth.py`:

| Function | Line |
| --- | ---: |
| `assign_ground_truth_by_majority_vote` | 149 |
| `reliability_weights_from_pairwise_cohen_kappa` | 195 |
| `assign_ground_truth_by_weighted_vote` | 250 |
| `list_items_annotators_disagreed_on` | 330 |

See also `docs/steps_2_to_6.md` (step 3) and the slide 5 and 6 presenter notes in `docs/phase2a_presenter_notes/`.
