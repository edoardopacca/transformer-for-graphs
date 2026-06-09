"""Regenerate the eval_families figures from already-computed families_eval.json
files, WITHOUT re-running the GPU eval. Use this to iterate on the plots (e.g. the
red 'disconnected' bar in the capacity figure) in seconds, locally.

    python plot_families.py                       # every families_eval.json under runs/
    python plot_families.py runs/families_n20      # only under a given root
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from eval_families import _plots   # reuse the exact plotting code


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
    jsons = sorted(root.rglob("families_eval.json"))
    if not jsons:
        print(f"no families_eval.json under {root}"); return
    for jp in jsons:
        results = json.loads(jp.read_text())
        _plots(results, jp.parent, results.get("n", 20))
        print(f"redrew {jp.parent}")
    print(f"done: {len(jsons)} figure sets")


if __name__ == "__main__":
    main()
