"""
Multi-seed runner for DSGR Qwen pipeline.

Usage:
  cd SAT
  python run_multiseed_qwen.py --seeds 42,43,44 --variant DSGR_CoT
"""
import os
import json
import argparse
import subprocess
import statistics
from datetime import datetime


def run_one_seed(seed: int, run_id: str, resume: bool):
    env = os.environ.copy()
    env["DSGR_SEED"] = str(seed)
    env["DSGR_RUN_ID"] = run_id
    env["DSGR_RESUME"] = "1" if resume else "0"
    cmd = ["python", "run_full_qwen.py"]
    subprocess.check_call(cmd, env=env)


def safe_stats(vals):
    if not vals:
        return None, None
    mean_v = statistics.mean(vals)
    std_v = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return round(mean_v, 4), round(std_v, 4)


def main():
    parser = argparse.ArgumentParser(description="Run DSGR Qwen pipeline with multiple seeds.")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds.")
    parser.add_argument("--variant", default="DSGR_CoT", help="Variant key in final_results.json")
    parser.add_argument("--run_prefix", default="multiseed", help="Run ID prefix.")
    parser.add_argument("--resume", action="store_true", help="Resume each seed run if checkpoint exists.")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    root = os.path.join("results_qwen", "runs")
    os.makedirs(root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    for seed in seeds:
        run_id = f"{args.run_prefix}_{stamp}_seed{seed}"
        print(f"\n=== Running seed {seed} | run_id={run_id} ===")
        run_one_seed(seed=seed, run_id=run_id, resume=args.resume)
        result_path = os.path.join(root, run_id, "final_results.json")
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Result file not found: {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        if args.variant not in res:
            raise KeyError(f"Variant '{args.variant}' not found in {result_path}. Keys={list(res.keys())}")
        r = res[args.variant]
        rows.append({
            "seed": seed,
            "run_id": run_id,
            "Hit@1": float(r.get("Hit@1", 0.0)),
            "Hit@3": float(r.get("Hit@3", 0.0)),
            "Hit@10": float(r.get("Hit@10", 0.0)),
            "MRR": float(r.get("MRR", 0.0)),
            "total": int(r.get("total", 0)),
            "skipped": int(r.get("skipped", 0)),
        })

    hit1_mean, hit1_std = safe_stats([x["Hit@1"] for x in rows])
    hit3_mean, hit3_std = safe_stats([x["Hit@3"] for x in rows])
    hit10_mean, hit10_std = safe_stats([x["Hit@10"] for x in rows])
    mrr_mean, mrr_std = safe_stats([x["MRR"] for x in rows])

    summary = {
        "variant": args.variant,
        "seeds": seeds,
        "runs": rows,
        "aggregate": {
            "Hit@1_mean": hit1_mean, "Hit@1_std": hit1_std,
            "Hit@3_mean": hit3_mean, "Hit@3_std": hit3_std,
            "Hit@10_mean": hit10_mean, "Hit@10_std": hit10_std,
            "MRR_mean": mrr_mean, "MRR_std": mrr_std,
        }
    }
    out_path = os.path.join(root, f"{args.run_prefix}_{stamp}_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Multi-seed summary ===")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
