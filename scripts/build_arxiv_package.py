#!/usr/bin/env python3
"""Build a minimal deterministic arXiv source archive from paper/current."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "current"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    members = [
        PAPER / "main.tex",
        PAPER / "generated" / "release_macros.tex",
        PAPER / "figures" / "release_evidence.pdf",
    ]
    for path in members:
        if not path.is_file():
            raise RuntimeError(f"missing arXiv source member: {path}")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            info = zipfile.ZipInfo(str(path.relative_to(PAPER)))
            info.date_time = (2026, 8, 9, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    print(output)


if __name__ == "__main__":
    main()
