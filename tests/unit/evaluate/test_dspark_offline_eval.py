import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[3] / "scripts" / "evaluate" / "dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("dspark_offline_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prompt_from_deepspec_turns():
    module = _load_module()

    prompt = module._prompt_from_record(
        {"turns": ["Solve this.", "Now continue."]},
        tokenizer=None,
        source="sample.jsonl:1",
    )

    assert prompt == "Solve this.\n\nNow continue."


def test_prompt_from_raw_problem_field():
    module = _load_module()

    prompt = module._prompt_from_record(
        {"problem": "What is 1+1?"},
        tokenizer=None,
        source="sample.jsonl:1",
    )

    assert prompt == "What is 1+1?"


def test_load_jsonl_rejects_non_object(tmp_path: Path):
    module = _load_module()
    path = tmp_path / "bad.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    try:
        module._load_jsonl(path)
    except ValueError as exc:
        assert "expected JSON object" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_discover_datasets_filters_by_stem(tmp_path: Path):
    module = _load_module()
    keep = tmp_path / "humaneval.jsonl"
    drop = tmp_path / "math.jsonl"
    keep.write_text(json.dumps({"prompt": "a"}) + "\n", encoding="utf-8")
    drop.write_text(json.dumps({"prompt": "b"}) + "\n", encoding="utf-8")

    paths = module._discover_datasets(tmp_path, ["humaneval"])

    assert paths == [keep]


def test_eval_stats_acceptance_length():
    module = _load_module()
    stats = module.EvalStats(
        num_proposals=4,
        num_proposed_draft_tokens=28,
        num_accepted_draft_tokens=12,
    )

    assert stats.acceptance_length == 4.0
    assert stats.accepted_draft_length == 3.0
