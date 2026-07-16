import importlib.util
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
