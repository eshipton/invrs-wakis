"""Render figures/animations from run directories (cheap, no simulation).

  python scripts/render.py --results <dir> [--runs] [--gallery gallery.png]

--runs re-renders every run directory found under <dir> (directories holding
a summary.json); the cross-challenge gallery (best seed per challenge: final
design on top, field below, official eval metric in the title) is written to
<dir>/gallery.png unless --gallery gives another path.
"""

import argparse
import json
import os
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")

from invrs_wakis.visualize import gallery, render_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--runs", action="store_true", help="re-render each run directory")
    ap.add_argument("--gallery", default=None)
    args = ap.parse_args()
    root = pathlib.Path(args.results)
    if args.runs:
        for s in sorted(root.glob("**/summary.json")):
            name = json.loads(s.read_text()).get("challenge")
            print(s.parent, render_run(s.parent, name), flush=True)
    path = gallery(root, args.gallery or root / "gallery.png")
    print("gallery:", path)


if __name__ == "__main__":
    main()
