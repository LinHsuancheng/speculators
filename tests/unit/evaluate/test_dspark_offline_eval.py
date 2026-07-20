import importlib.util
import json
import math
from types import SimpleNamespace
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


def test_prompt_from_sharegpt_conversations_stops_before_answer():
    module = _load_module()

    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return repr(messages)

    prompt = module._prompt_from_record(
        {
            "conversations": [
                {"from": "human", "value": "Question?"},
                {"from": "gpt", "value": "Answer."},
            ],
        },
        tokenizer=Tokenizer(),
        source="sample.jsonl:1",
    )

    assert prompt == "[{'role': 'user', 'content': 'Question?'}]"
    assert "Answer." not in prompt


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
    assert stats.draft_length == 7.0
    assert stats.accepted_draft_length == 3.0


def test_eval_stats_position_accept_rates():
    module = _load_module()
    stats = module.EvalStats()
    stats.add_response(
        SimpleNamespace(
            num_output_tokens=0,
            proposal_lengths=[3, 3, 2],
            accepted_draft_lengths=[3, 1, 0],
        ),
    )

    assert stats.position_proposed_counts == [3, 3, 2]
    assert stats.position_accepted_counts == [2, 1, 1]
    assert stats.position_accept_rates == [2 / 3, 1 / 3, 1 / 2]


def test_eval_stats_position_probability_means():
    module = _load_module()
    stats = module.EvalStats()
    stats.add_response(
        SimpleNamespace(
            num_output_tokens=0,
            proposal_lengths=[2, 2],
            accepted_draft_lengths=[1, 0],
            accept_prob_lists=[[0.8, 0.2], [0.4, 0.1]],
            support_accept_rate_lists=[[0.9, 0.3], [0.7, 0.5]],
        ),
    )

    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(
            stats.position_accept_prob_sums,
            [1.2, 0.3],
            strict=True,
        )
    )
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(
            stats.position_support_accept_rate_sums,
            [1.6, 0.8],
            strict=True,
        )
    )
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(
            stats.position_accept_prob_means,
            [0.6, 0.15],
            strict=True,
        )
    )
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(
            stats.position_support_accept_rate_means,
            [0.8, 0.4],
            strict=True,
        )
    )


def test_aggregate_rows_merges_position_accept_rates():
    module = _load_module()

    row = module._aggregate_rows(
        "sample",
        [
            {
                "num_requests": 1,
                "elapsed_s": 1.0,
                "total_output_tokens": 4,
                "num_proposals": 2,
                "num_proposed_draft_tokens": 5,
                "num_accepted_draft_tokens": 3,
                "position_accepted_counts": "[2, 1, 0]",
                "position_proposed_counts": "[2, 2, 1]",
            },
            {
                "num_requests": 1,
                "elapsed_s": 2.0,
                "total_output_tokens": 5,
                "num_proposals": 1,
                "num_proposed_draft_tokens": 2,
                "num_accepted_draft_tokens": 1,
                "position_accepted_counts": "[1, 0]",
                "position_proposed_counts": "[1, 1]",
            },
        ],
    )

    assert json.loads(row["position_accepted_counts"]) == [3, 1, 0]
    assert json.loads(row["position_proposed_counts"]) == [3, 3, 1]
    assert json.loads(row["position_accept_rates"]) == [1.0, 1 / 3, 0.0]


def test_draft_sample_from_anchor_defaults_to_false():
    module = _load_module()

    assert module._draft_sample_from_anchor(SimpleNamespace()) is False


def test_draft_sample_from_anchor_reads_config():
    module = _load_module()

    draft = SimpleNamespace(config=SimpleNamespace(sample_from_anchor=True))

    assert module._draft_sample_from_anchor(draft) is True


def test_shard_records_round_robin():
    module = _load_module()
    records = [{"prompt": str(i)} for i in range(7)]

    shard = module._shard_records(records, shard_index=1, num_shards=3)

    assert shard == [(2, records[1]), (5, records[4])]


def test_aggregate_rows_recomputes_weighted_lengths():
    module = _load_module()

    row = module._aggregate_rows(
        "sample",
        [
            {
                "num_requests": 2,
                "elapsed_s": 4.0,
                "total_output_tokens": 20,
                "num_proposals": 2,
                "num_proposed_draft_tokens": 8,
                "num_accepted_draft_tokens": 4,
            },
            {
                "num_requests": 3,
                "elapsed_s": 5.0,
                "total_output_tokens": 40,
                "num_proposals": 3,
                "num_proposed_draft_tokens": 18,
                "num_accepted_draft_tokens": 9,
            },
        ],
    )

    assert row["dataset"] == "sample"
    assert row["num_requests"] == 5
    assert row["elapsed_s"] == 5.0
    assert row["output_tokens_per_second"] == 12.0
    assert row["draft_length"] == 5.2
    assert row["acceptance_length"] == 3.6
    assert row["accepted_draft_length"] == 2.6


def test_draft_ids_to_target_ids_uses_d2t_offsets():
    module = _load_module()
    module.torch = __import__("torch")

    class Draft:
        use_draft_vocab = True
        d2t = module.torch.tensor([0, 4, 10])

    assert module._draft_ids_to_target_ids(Draft(), [0, 1, 2]) == [0, 5, 12]


def test_expand_draft_probs_uses_d2t_offsets():
    module = _load_module()
    module.torch = __import__("torch")

    class Draft:
        use_draft_vocab = True
        verifier_vocab_size = 8
        t2d = module.torch.zeros(8, dtype=module.torch.bool)
        d2t = module.torch.tensor([0, 2, 4])

    runner = object.__new__(module.DSparkOfflineRunner)
    runner.draft_model = Draft()
    draft_probs = module.torch.tensor([[[0.2, 0.3, 0.5]]])

    expanded = runner._expand_draft_probs_to_target_vocab(draft_probs)

    assert expanded.shape == (1, 1, 8)
    assert module.torch.allclose(
        expanded[0, 0],
        module.torch.tensor([0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.5, 0.0]),
    )
    assert module.torch.allclose(expanded.sum(), module.torch.tensor(1.0))


def test_ensure_loaded_vocab_mappings_rejects_missing_pruned_mapping(tmp_path: Path):
    module = _load_module()
    module.torch = __import__("torch")

    class Draft:
        use_draft_vocab = True

    args = SimpleNamespace(
        draft_model=str(tmp_path),
        d2t_path=None,
        t2d_path=None,
    )

    try:
        module._ensure_loaded_vocab_mappings(Draft(), args)
    except ValueError as exc:
        assert "no real d2t/t2d mapping" in str(exc)
    else:
        raise AssertionError("expected ValueError")
