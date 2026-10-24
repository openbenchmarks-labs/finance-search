from finsearch.judge import parse_judge_payload


def test_verdict_after_other_braces_is_read():
    text = 'Checked the set {1, 2}.\n{"correct": true, "reason": "within tolerance"}'
    assert parse_judge_payload(text) == {"correct": True, "reason": "within tolerance"}


def test_prose_without_a_verdict_is_empty():
    assert parse_judge_payload("no json here") == {}


def test_an_earlier_object_without_correct_is_skipped():
    text = '{"note": "see {a}"} {"correct": false, "reason": "wrong year"}'
    assert parse_judge_payload(text)["correct"] is False
