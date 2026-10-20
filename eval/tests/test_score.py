"""Scorer behaviour on synthetic predictions over the frozen dev slice."""

import random

import pytest

from tar_eval.common import read_jsonl, slice_path
from tar_eval.compare import compare, mcnemar_exact
from tar_eval.score import aggregate, letter_parse, score_items, yesno_parse

DEV = read_jsonl(slice_path("dev"))
CLOSED = [r for r in DEV if r["task"] in ("bcq", "mcq", "temporal_localization")]


def test_parse_kinds():
    assert yesno_parse("Yes.") == "strict"
    assert yesno_parse("  no, the car stops") == "strict"
    assert yesno_parse("I think the answer is no.") == "lenient"
    assert yesno_parse("I cannot tell.") == "fail"
    assert yesno_parse("") == "fail"
    assert letter_parse("B") == "strict"
    assert letter_parse("(C) the truck swerves") == "strict"
    assert letter_parse("The correct option is D.") == "lenient"
    assert letter_parse("None of these.") == "fail"
    # Upstream quirk, reproduced on purpose: a leading article parses as A.
    assert letter_parse("A car turns left.") == "strict"


def test_perfect_predictions():
    s = aggregate(score_items(CLOSED, {r["item_id"]: r["answer"] for r in CLOSED}))
    assert s["official"]["bcq_accuracy"] == 1.0
    assert s["official"]["mcq_accuracy"] == 1.0
    assert s["official"]["temporal_localization_miou"] >= 0.98   # zero-length GT scores 0
    assert s["parse_rates"]["bcq"]["strict"] == 1.0
    assert s["parse_rates"]["mcq"]["strict"] == 1.0


def test_random_predictions_land_near_chance():
    rng = random.Random(0)
    preds = {}
    for r in CLOSED:
        if r["task"] == "bcq":
            preds[r["item_id"]] = rng.choice(["Yes", "No"])
        elif r["task"] == "mcq":
            preds[r["item_id"]] = rng.choice("ABCD")
        else:
            preds[r["item_id"]] = '{"start": "00:00", "end": "00:01"}'
    s = aggregate(score_items(CLOSED, preds))
    assert 0.35 < s["official"]["bcq_accuracy"] < 0.65
    assert 0.10 < s["official"]["mcq_accuracy"] < 0.40


def test_malformed_predictions_fail_to_parse():
    s = aggregate(score_items(CLOSED, {r["item_id"]: "I'm not sure." for r in CLOSED}))
    assert s["official"]["bcq_accuracy"] == 0.0
    assert s["official"]["mcq_accuracy"] == 0.0
    assert s["official"]["temporal_localization_miou"] == 0.0
    assert s["parse_rates"]["bcq"]["fail"] == 1.0
    assert s["parse_rates"]["temporal_localization"]["fail"] == 1.0


def test_partially_parseable_is_lenient_but_correct():
    rows = [r for r in CLOSED if r["task"] == "bcq"][:20]
    preds = {r["item_id"]: f"Looking closely, the answer is {r['answer'].lower()}." for r in rows}
    s = aggregate(score_items(rows, preds))
    assert s["official"]["bcq_accuracy"] == 1.0
    assert s["parse_rates"]["bcq"]["lenient"] == 1.0


def test_temporal_skips_unparseable_like_upstream():
    rows = [r for r in CLOSED if r["task"] == "temporal_localization"][:10]
    preds = {r["item_id"]: (r["answer"] if i < 5 else "no idea") for i, r in enumerate(rows)}
    s = aggregate(score_items(rows, preds))
    # Official mIoU averages only the 5 parseable items; the strict variant counts all 10.
    assert s["official"]["temporal_localization_miou"] > 0.9
    assert s["extra"]["temporal_localization_miou_unparseable_as_zero"] == pytest.approx(
        s["official"]["temporal_localization_miou"] / 2, abs=1e-6)


def test_missing_predictions_are_an_error_unless_allowed():
    rows = CLOSED[:10]
    preds = {r["item_id"]: r["answer"] for r in rows[:5]}
    with pytest.raises(SystemExit):
        score_items(rows, preds)
    assert len(score_items(rows, preds, allow_missing=True)) == 5


def test_bertscore_identity_scores_one():
    rows = [r for r in DEV if r["task"] == "causal_linkage"][:8]
    s = aggregate(score_items(rows, {r["item_id"]: r["answer"] for r in rows}))
    assert s["official"]["causal_linkage_bertscore_f1"] == pytest.approx(1.0, abs=1e-4)


def test_compare_recovers_a_known_delta():
    rows = [r for r in CLOSED if r["task"] == "bcq"]
    a = score_items(rows, {r["item_id"]: r["answer"] for r in rows})
    flipped = {r["item_id"]: ("No" if r["answer"].startswith("Y") else "Yes") if i % 10 == 0
               else r["answer"] for i, r in enumerate(rows)}
    b = score_items(rows, flipped)
    res = next(x for x in compare(a, b, n_boot=500)["results"] if x["task"] == "bcq")
    expected = -sum(1 for i in range(len(rows)) if i % 10 == 0) / len(rows)
    assert res["delta"] == pytest.approx(expected)
    assert res["ci95"][0] <= res["delta"] <= res["ci95"][1]
    assert res["discordant_a_only"] == -round(expected * len(rows))
    assert res["mcnemar_p"] < 0.001


def test_mcnemar_exact_edges():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)


def test_final_answer_extraction():
    from tar_eval.score import final_interval, final_letter, final_yesno
    # answer first: the same answer the official extractors take
    assert final_yesno("Yes. No vehicle stops.") == "yes"
    assert final_letter("C. The truck turns.") == "C"
    # explain first: the last answer, not the first stray token
    assert final_letter("Based on the frames, A car turns... so the answer is D") == "D"
    assert final_yesno("I see no collision at first, but then... yes") == "yes"
    s = 'The SUV passes.\n\n{"start": "00:04.68", "end": "00:05.96"}'
    assert final_interval(s) is not None
