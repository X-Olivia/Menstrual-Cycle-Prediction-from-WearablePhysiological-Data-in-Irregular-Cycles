from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run personalization stratified LLCO pipeline")
    parser.add_argument("--cv-threshold", type=float, default=0.15)
    parser.add_argument("--shots", type=str, default="0,1,2,3")
    parser.add_argument("--detector", type=str, default="oracle")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    py = sys.executable

    cmd_eval = [
        py,
        str(base / "eval_llco_kshot.py"),
        "--cv-threshold",
        str(args.cv_threshold),
        "--shots",
        args.shots,
        "--detector",
        args.detector,
    ]
    cmd_analysis = [py, str(base / "analysis_interaction.py")]

    print("[RUN]", " ".join(cmd_eval))
    subprocess.run(cmd_eval, check=True)
    print("[RUN]", " ".join(cmd_analysis))
    subprocess.run(cmd_analysis, check=True)
    print("[DONE] personalization stratified pipeline finished.")


if __name__ == "__main__":
    main()
