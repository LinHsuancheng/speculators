"""Tests for CLI arguments."""

from scripts.train import parse_args
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.metrics import ce_loss, kl_div_loss, tv_loss
from speculators.models.peagle.core import PEagleDraftModel


def _parse(monkeypatch, extra: list[str]):
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--verifier-name-or-path", "dummy"] + extra
    )
    return parse_args()


# ---------------------------------------------------------------------------
# Ensure CLI args flow correctly through vars(args) into get_trainer_kwargs
# ---------------------------------------------------------------------------


def test_dflash_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]
    assert train_kw["gamma"] == 4.0
    assert val_kw["gamma"] == 4.0


def test_dflash_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert "ce" in val_kw["loss_config"]
    assert train_kw["gamma"] == 4.0
    assert val_kw["gamma"] == 4.0


def test_dflash_explicit_decay_gamma(monkeypatch):
    args = _parse(monkeypatch, ["--dflash-decay-gamma", "7.0"])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["gamma"] == 7.0
    assert val_kw["gamma"] == 7.0


def test_dflash_decay_gamma_falls_back_when_omitted():
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(loss_fn="kl_div")
    assert train_kw["gamma"] == 4.0
    assert val_kw["gamma"] == 4.0


def test_dflash_default_micro_block_layout(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.block_size == 16
    assert args.anchor_len == 1
    assert args.micro_block_size == 0
    assert args.micro_block_layer_growth is False
    assert args.max_prev_micro_blocks is None


def test_dflash_micro_block_args(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--micro-block-size",
            "3",
            "--anchor-len",
            "1",
            "--micro-block-layer-growth",
            "--max-prev-micro-blocks",
            "4",
        ],
    )
    assert args.micro_block_size == 3
    assert args.anchor_len == 1
    assert args.micro_block_layer_growth is True
    assert args.max_prev_micro_blocks == 4


def test_dflash_compound_loss(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", '{"ce": 0.1, "tv": 0.9}'])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert "tv" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][1] == 0.1
    assert train_kw["loss_config"]["tv"][1] == 0.9
    assert "ce" in val_kw["loss_config"]
    assert "tv" in val_kw["loss_config"]


def test_eagle3_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = Eagle3DraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]


def test_eagle3_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = Eagle3DraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert "ce" in val_kw["loss_config"]


def test_peagle_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = PEagleDraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]


def test_peagle_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = PEagleDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert "ce" in val_kw["loss_config"]


def test_dspark_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]
    assert train_kw["confidence_head_alpha"] == 1.0
    assert val_kw["confidence_head_alpha"] == 1.0


def test_dspark_compound_loss(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", '{"ce": 0.1, "tv": 0.9}'])
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert train_kw["loss_config"]["ce"][1] == 0.1
    assert "tv" in train_kw["loss_config"]
    assert train_kw["loss_config"]["tv"][0] is tv_loss
    assert train_kw["loss_config"]["tv"][1] == 0.9
    assert "ce" in val_kw["loss_config"]
    assert "tv" in val_kw["loss_config"]


def test_dspark_confidence_head_alpha(monkeypatch):
    args = _parse(monkeypatch, ["--confidence-head-alpha", "0.5"])
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["confidence_head_alpha"] == 0.5
    assert val_kw["confidence_head_alpha"] == 0.5
