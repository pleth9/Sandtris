from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_ai_train_is_only_a_packaged_launcher():
    legacy_train = (ROOT / "AI" / "train.py").read_text()

    assert "sandtris.ai.train" in legacy_train
    assert "calculate_spanning_potential" not in legacy_train
    assert "pygame" not in legacy_train
