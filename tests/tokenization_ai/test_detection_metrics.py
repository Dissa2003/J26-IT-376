"""Tests for detection-metrics-1 (draft protocol heldout-eval-1.2.0-draft, sections 7 and 12.11).

Small, manually constructed score arrays only. No model is trained or scored, no dataset is generated, and the confirmatory
seeds 7101-7103 are never used. Expected values below are derived by hand in the comments, and a brute-force pairwise / explicit
resampling reference is used as an independent check.
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import random
from pathlib import Path

import pytest

import tokenization_ai.heldout_eval.detection_metrics as dm
from tokenization_ai.heldout_eval.detection_metrics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    ConfusionCounts,
    DetectionMetricsError,
    average_precision,
    confusion_counts,
    evaluate_at_threshold,
    paired_group_bootstrap,
    roc_auc,
    select_threshold,
)

SOURCE = Path(dm.__file__)


def brute_auc(scores, labels):
    """Independent reference: average over all positive/negative pairs, ties worth one half."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def brute_ap(scores, labels):
    """Independent reference: precision at every distinct threshold, weighted by the recall gained there."""
    total_pos = sum(labels)
    ap, prev_recall = 0.0, 0.0
    for t in sorted(set(scores), reverse=True):
        predicted = [y for s, y in zip(scores, labels) if s >= t]
        recall = sum(predicted) / total_pos
        ap += (recall - prev_recall) * (sum(predicted) / len(predicted))
        prev_recall = recall
    return ap


# -- constants --------------------------------------------------------------------------------------------------------


def test_protocol_constants():
    assert dm.METRICS_ID == "detection-metrics-1"
    assert (BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, dm.BOOTSTRAP_CONFIDENCE) == (2000, 8201, 0.95)
    assert dm.MIN_VALID_PERCENT == 95


# -- average precision ------------------------------------------------------------------------------------------------


def test_ap_known_example_sorted_by_hand():
    # Ranking: 0.8(1) 0.4(0) 0.35(1) 0.1(0): precision at the two positives is 1/1 and 2/3, each worth recall 1/2 => 5/6.
    scores, labels = [0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]
    assert average_precision(scores, labels) == pytest.approx(5 / 6)
    assert brute_ap(scores, labels) == pytest.approx(5 / 6)


def test_ap_perfect_and_worst_ranking():
    assert average_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    # Positives ranked last of 4: precision 1/3 and 2/4 at the positives => (1/3 + 1/2) / 2 = 5/12.
    assert average_precision([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(5 / 12)


def test_ap_has_no_interpolation():
    # Ranking 1,0,1,1,0: step-wise AP = (1/1 + 2/3 + 3/4) / 3 = 29/36. Interpolated AP would raise 2/3 to 3/4.
    scores, labels = [5, 4, 3, 2, 1], [1, 0, 1, 1, 0]
    assert average_precision(scores, labels) == pytest.approx(29 / 36)
    assert average_precision(scores, labels) < (1 + 0.75 + 0.75) / 3


def test_ap_all_equal_scores_is_prevalence():
    assert average_precision([0.5] * 4, [1, 0, 0, 1]) == pytest.approx(0.5)
    assert average_precision([0.5] * 5, [1, 0, 0, 0, 0]) == pytest.approx(0.2)


def test_ap_groups_tied_scores_as_one_block():
    # Block {1.0: 1 pos + 1 neg} then {0.5: 1 pos}: block precisions 1/2 and 2/3 => (1/2 + 2/3) / 2 = 7/12.
    scores, labels = [1.0, 1.0, 0.5], [1, 0, 1]
    expected = (1 / 2 + 2 / 3) / 2
    assert average_precision(scores, labels) == pytest.approx(expected)
    assert average_precision(list(reversed(scores)), list(reversed(labels))) == pytest.approx(expected)
    assert brute_ap(scores, labels) == pytest.approx(expected)


def test_ap_tie_does_not_depend_on_listing_order_inside_a_block():
    scores = [0.7, 0.7, 0.7, 0.3, 0.3]
    labels = [1, 0, 0, 1, 0]
    reference = average_precision(scores, labels)
    for order in itertools.permutations(range(5)):
        assert average_precision([scores[i] for i in order], [labels[i] for i in order]) == reference


def test_ap_is_input_order_independent_on_random_permutations():
    rng = random.Random(1)
    scores = [round(rng.random(), 1) for _ in range(30)]  # many ties
    labels = [rng.randint(0, 1) for _ in range(30)]
    labels[0], labels[1] = 1, 0
    reference = average_precision(scores, labels)
    assert reference == pytest.approx(brute_ap(scores, labels))
    for _ in range(25):
        order = list(range(30))
        rng.shuffle(order)
        assert average_precision([scores[i] for i in order], [labels[i] for i in order]) == reference


def test_ap_signed_zero_scores_tie():
    assert average_precision([0.0, -0.0, 1.0], [1, 0, 1]) == average_precision([0.0, 0.0, 1.0], [1, 0, 1])


def test_ap_scores_may_be_ints_and_numpy_floats():
    np = pytest.importorskip("numpy")
    assert average_precision([3, 2, 1], [1, 0, 1]) == average_precision([np.float64(3), np.float64(2), np.float64(1)], [1, 0, 1])


def test_ap_undefined_without_positives():
    assert average_precision([0.1, 0.2, 0.3], [0, 0, 0]) is None


def test_ap_defined_with_only_positives():
    assert average_precision([0.1, 0.2, 0.3], [1, 1, 1]) == 1.0


# -- ROC-AUC ----------------------------------------------------------------------------------------------------------


def test_roc_auc_known_example():
    # Positives {0.35, 0.8}, negatives {0.1, 0.4}: 3 of the 4 pairs are won => 0.75.
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)


def test_roc_auc_perfect_reversed_and_chance():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5


def test_roc_auc_counts_ties_as_half():
    # Positives {1.0, 0.5}, negative {1.0}: the 1.0/1.0 pair is a half win, 0.5 loses => 0.5 / 2 = 0.25.
    assert roc_auc([1.0, 1.0, 0.5], [1, 0, 1]) == pytest.approx(0.25)
    scores, labels = [0.9, 0.5, 0.5, 0.5, 0.1, 0.9], [1, 1, 0, 1, 0, 0]
    assert roc_auc(scores, labels) == pytest.approx(brute_auc(scores, labels))


def test_roc_auc_matches_brute_force_on_random_tied_data_and_is_order_independent():
    rng = random.Random(2)
    scores = [rng.randint(0, 5) / 5 for _ in range(40)]
    labels = [rng.randint(0, 1) for _ in range(40)]
    labels[0], labels[1] = 1, 0
    reference = roc_auc(scores, labels)
    assert reference == pytest.approx(brute_auc(scores, labels))
    order = list(range(40))
    rng.shuffle(order)
    assert roc_auc([scores[i] for i in order], [labels[i] for i in order]) == reference


def test_roc_auc_undefined_when_a_class_is_missing():
    assert roc_auc([0.1, 0.2], [1, 1]) is None
    assert roc_auc([0.1, 0.2], [0, 0]) is None


# -- input validation -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scores", [[0.1, float("nan")], [0.1, float("inf")], [0.1, -float("inf")], [0.1, True], [0.1, "0.5"], [0.1, None]])
def test_invalid_scores_are_rejected(scores):
    for metric in (average_precision, roc_auc):
        with pytest.raises(DetectionMetricsError):
            metric(scores, [1, 0])
    with pytest.raises(DetectionMetricsError):
        confusion_counts(scores, [1, 0], 0.5)
    with pytest.raises(DetectionMetricsError):
        select_threshold(scores, [1, 0])


@pytest.mark.parametrize("labels", [[1, 2], [1, -1], [1, True], [1, 0.5], [1, "1"], [1, None]])
def test_invalid_labels_are_rejected(labels):
    with pytest.raises(DetectionMetricsError):
        average_precision([0.1, 0.2], labels)
    with pytest.raises(DetectionMetricsError):
        roc_auc([0.1, 0.2], labels)


def test_shape_errors_are_rejected():
    with pytest.raises(DetectionMetricsError):
        average_precision([0.1, 0.2], [1])
    with pytest.raises(DetectionMetricsError):
        average_precision([], [])
    with pytest.raises(DetectionMetricsError):
        average_precision("ab", [1, 0])
    with pytest.raises(DetectionMetricsError):
        average_precision({"a": 1}, [1])
    with pytest.raises(DetectionMetricsError):
        average_precision(5, [1])


def test_invalid_threshold_is_rejected():
    with pytest.raises(DetectionMetricsError):
        confusion_counts([0.1, 0.2], [1, 0], float("nan"))


def test_error_messages_do_not_echo_values():
    with pytest.raises(DetectionMetricsError) as caught:
        average_precision([0.123456789, "SECRET-VALUE"], [1, 0])
    assert "SECRET-VALUE" not in str(caught.value) and "0.123456789" not in str(caught.value)


# -- confusion matrix and threshold metrics ---------------------------------------------------------------------------


def test_confusion_counts_by_hand():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    labels = [1, 1, 0, 1, 0, 0]
    counts = confusion_counts(scores, labels, 0.6)  # predicted positive: 0.9, 0.8, 0.7, 0.6
    assert counts == ConfusionCounts(tp=3, fp=1, tn=2, fn=0)
    m = counts.metrics()
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == 1.0
    assert m["f1"] == pytest.approx(6 / 7)  # 2*3 / (2*3 + 1 + 0)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert m["false_negative_rate"] == 0.0
    assert counts.positives == 3 and counts.negatives == 3 and counts.records == 6


def test_threshold_is_inclusive():
    assert confusion_counts([0.5, 0.4], [1, 0], 0.5) == ConfusionCounts(tp=1, fp=0, tn=1, fn=0)
    assert confusion_counts([0.5, 0.5], [1, 0], 0.5) == ConfusionCounts(tp=1, fp=1, tn=0, fn=0)


def test_all_negative_prediction_counts():
    counts = confusion_counts([0.9, 0.1, 0.5], [1, 0, 1], None)
    assert counts == ConfusionCounts(tp=0, fp=0, tn=1, fn=2)
    m = counts.metrics()
    assert m["precision"] is None  # nothing predicted positive
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0  # positives exist, so 2TP / (2TP + FP + FN) = 0 / 2
    assert m["false_positive_rate"] == 0.0
    assert m["false_negative_rate"] == 1.0


def test_zero_division_is_explicit_never_zero_or_nan():
    no_positives = confusion_counts([0.9, 0.1], [0, 0], None).metrics()
    assert no_positives["recall"] is None and no_positives["false_negative_rate"] is None
    assert no_positives["precision"] is None and no_positives["f1"] is None  # nothing predicted, no positives
    assert no_positives["false_positive_rate"] == 0.0
    no_negatives = confusion_counts([0.9, 0.1], [1, 1], 0.5).metrics()
    assert no_negatives["false_positive_rate"] is None
    assert no_negatives["precision"] == 1.0 and no_negatives["recall"] == 0.5
    wrong_only = confusion_counts([0.9, 0.1], [0, 0], 0.5).metrics()
    assert wrong_only["precision"] == 0.0 and wrong_only["recall"] is None and wrong_only["f1"] == 0.0


# -- validation-only threshold selection ------------------------------------------------------------------------------


def test_threshold_maximizes_validation_f1():
    # F1 at >=0.9, 0.8, 0.7, 0.6, 0.5, 0.4: 1/2, 4/5, 2/3, 6/7, 3/4, 2/3 => best at 0.6.
    sel = select_threshold([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [1, 1, 0, 1, 0, 0])
    assert sel.threshold == 0.6
    assert sel.f1 == pytest.approx(6 / 7)
    assert sel.validation_counts == ConfusionCounts(tp=3, fp=1, tn=2, fn=0)
    assert sel.candidates_considered == 7  # six distinct scores plus the all-negative candidate


def test_threshold_tie_goes_to_the_highest_threshold():
    # F1 is 2/3 at >=0.9 and again at >=0.6 (0.8 and 0.7 are worse); the higher threshold must win.
    scores, labels = [0.9, 0.8, 0.7, 0.6], [1, 0, 0, 1]
    assert confusion_counts(scores, labels, 0.9).metrics()["f1"] == pytest.approx(2 / 3)
    assert confusion_counts(scores, labels, 0.6).metrics()["f1"] == pytest.approx(2 / 3)
    assert select_threshold(scores, labels).threshold == 0.9


def test_threshold_ties_are_decided_exactly_not_by_float_noise():
    # F1 = 2/3 is reached at >=0.9 (TP1 FN1 -> 2/3) and at >=0.6 (TP2 FP2 -> 4/6); exact fractions must tie them.
    scores, labels = [0.9, 0.8, 0.7, 0.6], [1, 0, 0, 1]
    sel = select_threshold(scores, labels)
    assert sel.threshold == 0.9 and sel.f1 == pytest.approx(2 / 3)


def test_threshold_is_independent_of_input_order():
    rng = random.Random(3)
    scores = [round(rng.random(), 1) for _ in range(30)]
    labels = [rng.randint(0, 1) for _ in range(30)]
    labels[0], labels[1] = 1, 0
    ref = select_threshold(scores, labels)
    order = list(range(30))
    rng.shuffle(order)
    assert select_threshold([scores[i] for i in order], [labels[i] for i in order]) == ref


def test_threshold_predicts_positive_at_or_above_candidate():
    scores, labels = [0.9, 0.8, 0.8, 0.1], [1, 1, 0, 0]
    sel = select_threshold(scores, labels)
    # >=0.9: TP1 FN1 => 2/3; >=0.8: TP2 FP1 => 4/5; >=0.1: TP2 FP2 => 2/3. Tied scores move together.
    assert sel.threshold == 0.8
    assert sel.f1 == pytest.approx(4 / 5)


def test_threshold_requires_both_classes_in_validation():
    with pytest.raises(DetectionMetricsError):
        select_threshold([0.1, 0.2], [1, 1])
    with pytest.raises(DetectionMetricsError):
        select_threshold([0.1, 0.2], [0, 0])


def test_threshold_all_equal_scores_predict_everything():
    sel = select_threshold([0.5, 0.5, 0.5], [1, 0, 0])
    assert sel.threshold == 0.5  # all-negative has F1 0; predicting all positive has F1 1/2
    assert sel.f1 == pytest.approx(0.5)


def test_all_negative_candidate_wins_nothing_when_a_real_threshold_is_at_least_as_good():
    # The all-negative candidate has F1 = 0 and is the highest "threshold", so it is chosen only if nothing scores above 0.
    sel = select_threshold([0.9, 0.1], [0, 1])
    assert sel.threshold == 0.1 and sel.f1 == pytest.approx(2 / 3)  # TP1 FP1 FN0


def test_selection_signature_cannot_receive_test_data():
    import inspect
    names = list(inspect.signature(select_threshold).parameters)
    assert names == ["validation_scores", "validation_labels"]


def test_evaluate_at_threshold_does_not_reselect():
    sel = select_threshold([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [1, 1, 0, 1, 0, 0])
    test_scores, test_labels = [0.95, 0.65, 0.55, 0.2], [0, 1, 0, 1]
    assert evaluate_at_threshold(test_scores, test_labels, sel) == confusion_counts(test_scores, test_labels, 0.6)
    assert sel.threshold == 0.6


def test_selection_to_dict_is_json_safe():
    sel = select_threshold([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [1, 1, 0, 1, 0, 0])
    payload = json.loads(json.dumps(sel.to_dict(), allow_nan=False))
    assert payload["threshold"] == 0.6 and payload["predicts_all_negative"] is False
    assert payload["basis"] == "validation_scores_and_labels_only"
    assert payload["validation_counts"]["precision"] == pytest.approx(0.75)


# -- paired group bootstrap -------------------------------------------------------------------------------------------

# Eight records in five template groups; g3 holds both classes (a mixed-label template).
RECORDS = {
    # id: (group, label, score A, score B)
    "r1": ("g1", 1, 0.9, 0.9),
    "r2": ("g1", 0, 0.2, 0.7),
    "r3": ("g2", 0, 0.8, 0.1),
    "r4": ("g2", 0, 0.3, 0.2),
    "r5": ("g3", 1, 0.4, 0.8),
    "r6": ("g3", 0, 0.7, 0.3),
    "r7": ("g4", 1, 0.6, 0.6),
    "r8": ("g5", 0, 0.1, 0.4),
}


def arrays(records=RECORDS):
    a = {i: v[2] for i, v in records.items()}
    b = {i: v[3] for i, v in records.items()}
    y = {i: v[1] for i, v in records.items()}
    g = {i: v[0] for i, v in records.items()}
    return a, b, y, g


def explicit_replicate(records, draws, group_ids):
    """Resample by physically duplicating the records of each drawn group; independent reference for the weighted code."""
    scores_a, scores_b, labels = [], [], []
    for k in draws:
        for _, (group, label, sa, sb) in records.items():
            if group == group_ids[k]:
                scores_a.append(sa)
                scores_b.append(sb)
                labels.append(label)
    return scores_a, scores_b, labels


def test_bootstrap_point_estimates_match_direct_ap():
    a, b, y, g = arrays()
    res = paired_group_bootstrap(a, b, y, g, replicates=10)
    ids = sorted(RECORDS)
    assert res.ap_a == pytest.approx(average_precision([a[i] for i in ids], [y[i] for i in ids]))
    assert res.ap_b == pytest.approx(average_precision([b[i] for i in ids], [y[i] for i in ids]))
    assert res.difference == pytest.approx(res.ap_b - res.ap_a)
    # Arm A ranks 0.9(1) 0.8(0) 0.7(0) 0.6(1) 0.4(1) 0.3 0.2 0.1: precisions 1, 2/4, 3/5 => (1 + 1/2 + 3/5)/3 = 0.7.
    assert res.ap_a == pytest.approx(0.7)
    # Arm B ranks 0.9(1) 0.8(1) 0.7(0) 0.6(1) 0.4 0.3 0.2 0.1: precisions 1, 1, 3/4 => 11/12.
    assert res.ap_b == pytest.approx(11 / 12)
    assert res.statistic == "AP(B) - AP(A)"


def test_bootstrap_uses_the_same_groups_for_both_arms_and_keeps_groups_whole():
    a, b, y, g = arrays()
    res = paired_group_bootstrap(a, b, y, g, replicates=60, keep_replicates=True)
    group_ids = sorted({v[0] for v in RECORDS.values()})
    rng = random.Random(8201)
    for diff in res.replicate_differences:
        draws = [rng.randrange(len(group_ids)) for _ in group_ids]
        sa, sb, lab = explicit_replicate(RECORDS, draws, group_ids)
        if sum(lab) == 0:
            assert diff is None
            continue
        # One draw, applied to both arms, over physically duplicated whole groups (the mixed g3 contributes both labels).
        assert diff == pytest.approx(average_precision(sb, lab) - average_precision(sa, lab))


def test_bootstrap_keeps_mixed_label_groups_intact():
    a, b, y, g = arrays()
    res = paired_group_bootstrap(a, b, y, g, replicates=5)
    assert res.mixed_label_groups == 2  # g1 (r1 positive, r2 negative) and g3 (r5 positive, r6 negative)
    assert res.groups == 5 and res.records == 8 and res.positive_records == 3 and res.groups_with_positive == 3
    # Drawing only g3 gives exactly its two records with both labels; the group is never split.
    group_ids = sorted({v[0] for v in RECORDS.values()})
    _, _, lab = explicit_replicate(RECORDS, [group_ids.index("g3")], group_ids)
    assert sorted(lab) == [0, 1]
    # Replicates that draw g1 or g3 but no other positive group are still defined, because the positive travels with its group.
    only_mixed = {i: v for i, v in RECORDS.items() if v[0] in ("g1", "g3")}
    ma, mb, my, mg = arrays(only_mixed)
    res2 = paired_group_bootstrap(ma, mb, my, mg, replicates=60, keep_replicates=True)
    assert res2.groups == 2 and res2.mixed_label_groups == 2
    assert res2.replicates_undefined == 0  # every resample holds at least one whole mixed group, hence a positive


def test_bootstrap_pairs_by_record_id_not_by_order():
    a, b, y, g = arrays()
    ref = paired_group_bootstrap(a, b, y, g, replicates=50, keep_replicates=True)
    shuffled = list(RECORDS)
    random.Random(4).shuffle(shuffled)
    again = paired_group_bootstrap({i: a[i] for i in shuffled}, {i: b[i] for i in reversed(shuffled)}, {i: y[i] for i in shuffled},
                                   {i: g[i] for i in reversed(shuffled)}, replicates=50, keep_replicates=True)
    assert again == ref


def test_bootstrap_rejects_unpaired_ids():
    a, b, y, g = arrays()
    for broken in ("scores_a", "scores_b", "groups"):
        args = {"scores_a": dict(a), "scores_b": dict(b), "labels": dict(y), "groups": dict(g)}
        del args[broken]["r3"]
        with pytest.raises(DetectionMetricsError):
            paired_group_bootstrap(args["scores_a"], args["scores_b"], args["labels"], args["groups"], replicates=5)
    extra = dict(b)
    extra["r9"] = 0.5
    with pytest.raises(DetectionMetricsError):
        paired_group_bootstrap(a, extra, y, g, replicates=5)
    renamed = {("x" + i): v for i, v in b.items()}
    with pytest.raises(DetectionMetricsError):
        paired_group_bootstrap(a, renamed, y, g, replicates=5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0.5"])
def test_bootstrap_rejects_invalid_scores_and_labels(bad):
    a, b, y, g = arrays()
    with pytest.raises(DetectionMetricsError):
        paired_group_bootstrap({**a, "r1": bad}, b, y, g, replicates=5)
    with pytest.raises(DetectionMetricsError):
        paired_group_bootstrap(a, {**b, "r1": bad}, y, g, replicates=5)
    with pytest.raises(DetectionMetricsError):
        paired_group_bootstrap(a, b, {**y, "r1": 2}, g, replicates=5)


def test_bootstrap_rejects_bad_settings_and_groups():
    a, b, y, g = arrays()
    for kwargs in ({"replicates": 0}, {"replicates": True}, {"replicates": 2.5}, {"seed": 1.5}, {"confidence": 1.0}, {"confidence": 0.0}):
        with pytest.raises(DetectionMetricsError):
            paired_group_bootstrap(a, b, y, g, **kwargs)
    with pytest.raises(DetectionMetricsError):
        paired_group_bootstrap(a, b, y, {i: "only" for i in g})
    with pytest.raises(DetectionMetricsError):
        paired_group_bootstrap(a, b, y, {**g, "r1": ""})
    with pytest.raises(DetectionMetricsError):
        paired_group_bootstrap({}, {}, {}, {})


def test_bootstrap_defaults_are_the_protocol_settings():
    a, b, y, g = arrays()
    res = paired_group_bootstrap(a, b, y, g)
    assert (res.replicates_requested, res.seed, res.confidence, res.min_valid_replicates) == (2000, 8201, 0.95, 1900)
    assert res.replicates_valid + res.replicates_undefined == 2000
    assert res.replicate_differences is None


def test_bootstrap_is_reproducible_and_seed_sensitive():
    a, b, y, g = arrays()
    first = paired_group_bootstrap(a, b, y, g, replicates=300, keep_replicates=True)
    second = paired_group_bootstrap(a, b, y, g, replicates=300, keep_replicates=True)
    assert first == second and first.to_dict() == second.to_dict()
    other = paired_group_bootstrap(a, b, y, g, replicates=300, seed=8202, keep_replicates=True)
    assert other.replicate_differences != first.replicate_differences


def test_bootstrap_does_not_touch_global_random_state():
    a, b, y, g = arrays()
    random.seed(99)
    state = random.getstate()
    paired_group_bootstrap(a, b, y, g, replicates=100)
    assert random.getstate() == state


def test_bootstrap_interval_is_the_linear_percentile_of_valid_replicates():
    a, b, y, g = arrays()
    res = paired_group_bootstrap(a, b, y, g, replicates=400, keep_replicates=True)
    assert res.status == dm.STATUS_OK
    valid = sorted(d for d in res.replicate_differences if d is not None)
    assert len(valid) == res.replicates_valid
    lo, hi = res.interval

    def pct(q):
        pos = (len(valid) - 1) * q
        i = math.floor(pos)
        j = min(i + 1, len(valid) - 1)
        return valid[i] + (valid[j] - valid[i]) * (pos - i)

    assert lo == pytest.approx(pct(0.025)) and hi == pytest.approx(pct(0.975))
    assert valid[0] <= lo <= hi <= valid[-1]


def test_undefined_replicates_are_tracked_not_replaced():
    # Eight groups, one positive record: a resample misses the positive group with probability (7/8)^8 = 0.34.
    records = {f"r{k}": (f"g{k}", 1 if k == 0 else 0, 0.1 * k, 0.9 - 0.1 * k) for k in range(8)}
    a, b, y, g = arrays(records)
    res = paired_group_bootstrap(a, b, y, g, replicates=200, keep_replicates=True)
    assert res.replicates_requested == 200
    assert res.replicates_undefined > 0
    assert res.replicates_valid + res.replicates_undefined == 200
    assert res.undefined_reasons == {dm.REASON_NO_POSITIVE: res.replicates_undefined}
    assert len(res.replicate_differences) == 200  # nothing was redrawn to fill the gaps
    assert sum(d is None for d in res.replicate_differences) == res.replicates_undefined
    # Exactly the draws that skip the single positive group are the undefined ones.
    rng = random.Random(8201)
    expected_undefined = sum(0 not in [rng.randrange(8) for _ in range(8)] for _ in range(200))
    assert res.replicates_undefined == expected_undefined


def test_too_many_undefined_replicates_withhold_the_interval():
    records = {f"r{k}": (f"g{k}", 1 if k == 0 else 0, 0.1 * k, 0.9 - 0.1 * k) for k in range(8)}
    a, b, y, g = arrays(records)
    res = paired_group_bootstrap(a, b, y, g, replicates=200)
    assert res.replicates_valid < res.min_valid_replicates == 190
    assert res.status == dm.STATUS_INSUFFICIENT
    assert res.interval is None
    assert res.difference is not None  # the full-sample estimate is still reported, only the interval is withheld


def test_minimum_valid_replicate_rule_is_95_percent_rounded_up():
    a, b, y, g = arrays()
    assert paired_group_bootstrap(a, b, y, g, replicates=1).min_valid_replicates == 1
    assert paired_group_bootstrap(a, b, y, g, replicates=20).min_valid_replicates == 19
    assert paired_group_bootstrap(a, b, y, g, replicates=21).min_valid_replicates == 20
    assert paired_group_bootstrap(a, b, y, g, replicates=2000).min_valid_replicates == 1900


def test_rule_boundary_exactly_at_minimum_is_accepted_and_one_above_is_not(monkeypatch):
    records = {f"r{k}": (f"g{k}", 1 if k == 0 else 0, 0.1 * k, 0.9 - 0.1 * k) for k in range(8)}
    a, b, y, g = arrays(records)
    valid = paired_group_bootstrap(a, b, y, g, replicates=200).replicates_valid  # the stream is fixed, so this is stable
    assert 0 < valid < 200
    # With 200 replicates the minimum is ceil(2 * percent). percent = valid // 2 needs at most `valid`; one more percent needs more.
    monkeypatch.setattr(dm, "MIN_VALID_PERCENT", valid // 2)
    assert paired_group_bootstrap(a, b, y, g, replicates=200).status == dm.STATUS_OK
    monkeypatch.setattr(dm, "MIN_VALID_PERCENT", valid // 2 + 1)
    assert paired_group_bootstrap(a, b, y, g, replicates=200).status == dm.STATUS_INSUFFICIENT


def test_undefined_point_estimate_when_no_positives():
    records = {f"r{k}": (f"g{k % 3}", 0, 0.1 * k, 0.2 * k) for k in range(6)}
    a, b, y, g = arrays(records)
    res = paired_group_bootstrap(a, b, y, g, replicates=50)
    assert res.status == dm.STATUS_UNDEFINED_POINT
    assert res.ap_a is None and res.ap_b is None and res.difference is None and res.interval is None
    assert res.replicates_valid == 0 and res.replicates_undefined == 0  # no replicate was attempted


def test_all_positive_sample_has_ap_one_for_both_arms_and_zero_difference():
    records = {f"r{k}": (f"g{k % 3}", 1, 0.1 * k, 0.3 * k) for k in range(6)}
    a, b, y, g = arrays(records)
    res = paired_group_bootstrap(a, b, y, g, replicates=40)
    assert res.ap_a == res.ap_b == 1.0 and res.difference == 0.0
    assert res.interval == (0.0, 0.0) and res.status == dm.STATUS_OK


def test_identical_arms_give_a_zero_difference_everywhere():
    a, _, y, g = arrays()
    res = paired_group_bootstrap(a, dict(a), y, g, replicates=100, keep_replicates=True)
    assert res.difference == 0.0 and res.interval == (0.0, 0.0)
    assert {d for d in res.replicate_differences if d is not None} == {0.0}


def test_swapping_arms_negates_the_difference():
    a, b, y, g = arrays()
    forward = paired_group_bootstrap(a, b, y, g, replicates=100, keep_replicates=True)
    backward = paired_group_bootstrap(b, a, y, g, replicates=100, keep_replicates=True)
    assert backward.difference == pytest.approx(-forward.difference)
    for f, r in zip(forward.replicate_differences, backward.replicate_differences):
        assert (f is None) == (r is None)
        if f is not None:
            assert r == pytest.approx(-f)


def test_tied_scores_across_groups_are_handled_as_one_block_in_replicates():
    records = {
        "r1": ("g1", 1, 0.5, 0.9),
        "r2": ("g2", 0, 0.5, 0.1),
        "r3": ("g3", 1, 0.5, 0.8),
        "r4": ("g4", 0, 0.2, 0.2),
    }
    a, b, y, g = arrays(records)
    res = paired_group_bootstrap(a, b, y, g, replicates=40, keep_replicates=True)
    group_ids = sorted(g.values())
    rng = random.Random(8201)
    for diff in res.replicate_differences:
        draws = [rng.randrange(4) for _ in range(4)]
        sa, sb, lab = explicit_replicate(records, draws, group_ids)
        if sum(lab) == 0:
            assert diff is None
        else:
            assert diff == pytest.approx(average_precision(sb, lab) - average_precision(sa, lab))


def test_result_serializes_to_strict_json():
    a, b, y, g = arrays()
    for res in (paired_group_bootstrap(a, b, y, g, replicates=80, keep_replicates=True),
                paired_group_bootstrap(a, b, y, g, replicates=80),
                paired_group_bootstrap({i: 0.0 for i in a}, {i: 0.0 for i in a}, {i: 0 for i in y}, g, replicates=10)):
        payload = json.loads(json.dumps(res.to_dict(), allow_nan=False))
        assert payload["metrics_id"] == "detection-metrics-1"
        assert payload["undefined_replicates_replaced"] is False
        assert payload["scores_are_uncalibrated"] is True
        assert payload["resampling_unit"] == "normalized_template_group"
        assert "not a real-world detection claim" in payload["scope"]
    ok = json.loads(json.dumps(paired_group_bootstrap(a, b, y, g, replicates=80).to_dict(), allow_nan=False))
    assert ok["status"] == "ok" and len(ok["interval"]) == 2 and "replicate_differences" not in ok


def test_confusion_to_dict_is_strict_json_with_nulls_for_undefined():
    payload = json.loads(json.dumps(confusion_counts([0.9, 0.1], [0, 0], None).to_dict(), allow_nan=False))
    assert payload["recall"] is None and payload["precision"] is None and payload["false_positive_rate"] == 0.0


def test_full_default_replicate_count_runs_on_the_small_example():
    a, b, y, g = arrays()
    res = paired_group_bootstrap(a, b, y, g)
    assert res.replicates_requested == BOOTSTRAP_REPLICATES
    if res.status == dm.STATUS_OK:
        lo, hi = res.interval
        assert -1.0 <= lo <= hi <= 1.0


# -- scope and hygiene ------------------------------------------------------------------------------------------------


def test_module_imports_only_the_standard_library():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports: the module must not depend on production or other Member 4 code"
            imported.add((node.module or "").split(".")[0])
    stdlib = {"__future__", "math", "numbers", "random", "collections", "dataclasses", "fractions", "typing"}
    assert imported <= stdlib, imported - stdlib


def test_module_does_not_use_global_random_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "random"]
    assert {c.attr for c in calls} == {"Random"}


def test_no_confirmatory_seed_is_referenced():
    text = SOURCE.read_text(encoding="utf-8")
    assert not any(seed in text for seed in ("7101", "7102", "7103"))
