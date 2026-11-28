import json

from scripts.compare_models import compare_candidates, format_markdown, load_candidate


def test_load_ngonsacviet_and_eval_report(tmp_path):
    meta_file = tmp_path / "cand_a" / "ngonsacviet.json"
    meta_file.parent.mkdir()
    meta_file.write_text(
        json.dumps(
            {
                "epochs": 0.4,
                "step": 450,
                "evaluation": {
                    "rad->vi": {"chrf": 27.8, "copy_chrf": 8.5, "long_chrf": 26.5, "repetition_rate": 0.01, "sentences": 993},
                    "tyz->vi": {"chrf": 16.2, "copy_chrf": 9.9, "long_chrf": 21.0, "repetition_rate": 0.0, "sentences": 2181},
                },
            }
        ),
        encoding="utf-8",
    )

    eval_file = tmp_path / "eval_b.json"
    eval_file.write_text(
        json.dumps(
            {
                "checkpoint": "nllb-5lang",
                "metadata": {"epochs": 1.0, "step": 800},
                "results": [
                    {
                        "direction": "rad->vi",
                        "sentences": 993,
                        "model": {"chrf": 31.0, "bleu": 12.1},
                        "copy_baseline": {"chrf": 8.5, "bleu": 1.0},
                        "by_length": {"16+w": {"chrf": 30.0}},
                        "length_ratio": 0.96,
                        "repetition_rate": 0.002,
                    },
                    {
                        "direction": "bdq->vi",
                        "sentences": 1927,
                        "model": {"chrf": 29.0, "bleu": 10.0},
                        "copy_baseline": {"chrf": 11.0, "bleu": 2.0},
                        "by_length": {"16+w": {"chrf": 28.0}},
                        "length_ratio": 0.94,
                        "repetition_rate": 0.0,
                    },
                    {
                        "direction": "tyz->vi",
                        "sentences": 2181,
                        "model": {"chrf": 15.0, "bleu": 6.0},
                        "copy_baseline": {"chrf": 9.9, "bleu": 2.0},
                        "by_length": {"16+w": {"chrf": 18.0}},
                        "length_ratio": 0.90,
                        "repetition_rate": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    a = load_candidate("a", meta_file.parent)
    b = load_candidate("b", eval_file)
    assert a["passed_directions"] == 2
    assert a["languages"] == ["rad", "tyz"]
    assert b["passed_directions"] == 3
    assert b["languages"] == ["bdq", "rad", "tyz"]

    summary = compare_candidates([("a", meta_file), ("b", eval_file)])
    assert summary["winner"] == "b"
    assert summary["per_direction"]["rad->vi"]["winner"] == "b"
    assert summary["per_direction"]["tyz->vi"]["winner"] == "a"
    md = format_markdown(summary)
    assert "### Overall Leaderboard" in md
    assert "### Per-Direction Winners" in md
