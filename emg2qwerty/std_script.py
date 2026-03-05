import subprocess
import json
from pathlib import Path
from datetime import datetime

STD_VALUES = [0.005, 0.01, 0.02, 0.03, 0.04]
METRIC_VAL = "val/CER"
METRIC_TEST = "test/CER"
BASE_CMD = [
    "python", "-m", "emg2qwerty.train",
    "user=single_user",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
]

def newest_results_json(search_root: Path, since_ts: float) -> Path | None:
    candidates = []
    for p in search_root.rglob("results.json"):
        try:
            if p.stat().st_mtime >= since_ts:
                candidates.append(p)
        except OSError:
            pass
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def main():
    project_root = Path.cwd()

    best_std = None
    best_val = float("inf")
    best_ckpt = None
    results_table = {}

    for std in STD_VALUES:
        print("\n")
        print(f"Running std={std}")
        print("\n")

        cmd = BASE_CMD + [f"gaussian_noise.std={std}"]

        start_ts = datetime.now().timestamp()

        proc = subprocess.Popen(cmd)
        proc.wait()

        if proc.returncode != 0:
            print(f"[ERROR] Run failed for std={std}")
            continue

        results_path = newest_results_json(project_root, start_ts)
        if results_path is None:
            print("[ERROR] Could not find results.json")
            continue

        with results_path.open("r", encoding="utf-8") as f:
            results = json.load(f)

        val_metrics = results["val_metrics"][0]
        if METRIC_VAL not in val_metrics:
            print(f"[ERROR] Metric '{METRIC_VAL}' not in results. Keys: {list(val_metrics.keys())}")
            continue

        test_metrics = results["test_metrics"][0]
        if METRIC_TEST not in test_metrics:
            print(f"[ERROR] Metric '{METRIC_TEST}' not in results. Keys: {list(test_metrics.keys())}")
            continue

        val_score = float(val_metrics[METRIC_VAL])
        test_score = float(test_metrics[METRIC_TEST])

        ckpt = results.get("best_checkpoint", None)

        results_table[str(std)] = (val_score, test_score, ckpt)

        print(f"[Complete] {METRIC_VAL}: {val_score}")
        print(f"[Complete] {METRIC_TEST}: {test_score}")
        print(f"[Complete] best_checkpoint: {ckpt}")

        if val_score < best_val:
            best_val = val_score
            best_std = std
            best_ckpt = ckpt

    for std, (v, t, ckpt) in results_table.items():
        print("\n")
        print(f"STD: {std}")
        print(f"{METRIC_VAL}: {v}")
        print(f"{METRIC_TEST}: {t}")
        print(f"ckpt: {ckpt}")

    print("\n")
    print(f"Best std: {best_std}")
    print(f"Best {METRIC_VAL}: {best_val}")
    print(f"Best checkpoint: {best_ckpt}")
    print("\n")

if __name__ == "__main__":
    main()