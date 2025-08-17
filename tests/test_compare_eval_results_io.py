import csv
from pathlib import Path

from scripts.compare_eval_results import (
    COLUMNS,
    overlap_by_model,
    read_summary_csv,
    write_combined_long,
    write_env_summary,
)


def make_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow(r)


def test_read_and_env_summary_and_overlap(tmp_path: Path):
    local_csv = tmp_path / "local.csv"
    online_csv = tmp_path / "online.csv"
    # Two models; one overlaps
    make_csv(
        local_csv,
        [
            [
                "modelA",
                100,
                0.5,
                0.6,
                120,
                300,
                0.4,
                0.5,
                150,
                350,
                10,
                5,
                20,
                65,
                0.1,
                0.05,
                60,
                30,
                10,
            ],
            [
                "modelB",
                50,
                0.7,
                0.8,
                90,
                200,
                0.6,
                0.7,
                110,
                220,
                8,
                4,
                30,
                8,
                0.2,
                0.1,
                25,
                15,
                10,
            ],
        ],
    )
    make_csv(
        online_csv,
        [
            [
                "modelA",
                120,
                0.55,
                0.65,
                130,
                310,
                0.45,
                0.55,
                160,
                360,
                12,
                6,
                22,
                68,
                0.12,
                0.06,
                62,
                32,
                12,
            ],
        ],
    )

    local = read_summary_csv(str(local_csv))
    online = read_summary_csv(str(online_csv))
    assert set(local.keys()) == {"modelA", "modelB"}
    assert set(online.keys()) == {"modelA"}

    # Env summary CSV is created and dicts returned
    outs = write_env_summary(local, online, str(tmp_path / "summary.csv"))
    assert {"local", "online"} == set(outs.keys())

    # Overlap detection
    overlap = overlap_by_model(local, online)
    assert overlap == ["modelA"]


def test_write_combined_long(tmp_path: Path):
    local = {
        "m": {"n": 10, "em_rag": 0.1, "f1_rag": 0.2, "p50_rag_ms": 5, "em_norag": 0.05, "f1_norag": 0.1},
    }
    online = {
        "m": {"n": 12, "em_rag": 0.2, "f1_rag": 0.3, "p50_rag_ms": 6, "em_norag": 0.06, "f1_norag": 0.11},
    }
    out_csv = tmp_path / "combined.csv"
    write_combined_long(local, online, str(out_csv))
    assert out_csv.exists()
    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("env,model,")
    assert any("local,m" in ln for ln in lines)
    assert any("online,m" in ln for ln in lines)
