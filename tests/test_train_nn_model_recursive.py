from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.ml import train_nn_model as mod


def test_recursive_rounds_stop_on_no_auc_improvement(monkeypatch) -> None:
    calls: list[dict] = []
    saved_summary: dict = {}

    def fake_train_nn(**kwargs):
        calls.append(kwargs)
        aucs = [0.60, 0.605, 0.606]
        idx = len(calls) - 1
        return SimpleNamespace(
            rows=100,
            metrics={"roc_auc": aucs[idx]},
            model_path=f"/tmp/model_{idx}.pt",
        )

    def fake_persist(model_dir: Path, summary: dict) -> Path:
        saved_summary["data"] = summary
        return model_dir / "summary.json"

    monkeypatch.setattr(mod, "load_env", lambda: None)
    monkeypatch.setattr(mod, "get_engine", lambda _url=None: object())
    monkeypatch.setattr(mod, "train_nn", fake_train_nn)
    monkeypatch.setattr(mod, "_persist_recursive_summary", fake_persist)
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_nn_model.py",
            "--rounds",
            "3",
            "--round-patience",
            "1",
            "--min-roc-auc-improvement",
            "0.01",
        ],
    )

    mod.main()

    assert len(calls) == 2
    assert saved_summary["data"]["rounds_completed"] == 2
    assert saved_summary["data"]["rounds_requested"] == 3


def test_recursive_rounds_honor_lr_decay(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_train_nn(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            rows=100,
            metrics={"roc_auc": 0.6 + (0.01 * len(calls))},
            model_path=f"/tmp/model_{len(calls)}.pt",
        )

    monkeypatch.setattr(mod, "load_env", lambda: None)
    monkeypatch.setattr(mod, "get_engine", lambda _url=None: object())
    monkeypatch.setattr(mod, "train_nn", fake_train_nn)
    monkeypatch.setattr(mod, "_persist_recursive_summary", lambda model_dir, summary: model_dir / "summary.json")
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_nn_model.py",
            "--rounds",
            "3",
            "--learning-rate",
            "0.001",
            "--lr-decay",
            "0.5",
            "--min-roc-auc-improvement",
            "0.0",
            "--round-patience",
            "5",
        ],
    )

    mod.main()

    assert len(calls) == 3
    lrs = [float(call["learning_rate"]) for call in calls]
    assert lrs == [0.001, 0.0005, 0.00025]


def test_recursive_promotes_best_checkpoint(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    copied: list[tuple[str, str]] = []
    saved_summary: dict = {}

    def fake_train_nn(**kwargs):
        idx = len(calls)
        calls.append(kwargs)
        model_path = tmp_path / f"nn_gru_attention_round{idx + 1}.pt"
        model_path.write_text(f"model-{idx + 1}", encoding="utf-8")
        aucs = [0.62, 0.61]
        return SimpleNamespace(
            rows=100,
            metrics={"roc_auc": aucs[idx]},
            model_path=str(model_path),
        )

    def fake_copy2(src: str | Path, dst: str | Path):
        copied.append((str(src), str(dst)))
        Path(dst).write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")
        return str(dst)

    def fake_persist(model_dir: Path, summary: dict) -> Path:
        saved_summary["data"] = summary
        return model_dir / "summary.json"

    monkeypatch.setattr(mod, "load_env", lambda: None)
    monkeypatch.setattr(mod, "get_engine", lambda _url=None: object())
    monkeypatch.setattr(mod, "train_nn", fake_train_nn)
    monkeypatch.setattr(mod, "_persist_recursive_summary", fake_persist)
    monkeypatch.setattr(mod.shutil, "copy2", fake_copy2)
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_nn_model.py",
            "--model-dir",
            str(tmp_path),
            "--rounds",
            "2",
            "--round-patience",
            "1",
            "--min-roc-auc-improvement",
            "0.001",
        ],
    )

    mod.main()

    assert len(calls) == 2
    assert len(copied) == 1
    src, dst = copied[0]
    assert src.endswith("nn_gru_attention_round1.pt")
    assert Path(dst).name.startswith("nn_gru_attention_")
    assert saved_summary["data"]["selected_model_path"] == dst
