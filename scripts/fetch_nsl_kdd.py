"""Download NSL-KDD (KDDTrain+ and KDDTest+, about 25 MB) into data/nsl_kdd/.

    python scripts/fetch_nsl_kdd.py

The official copy at the Canadian Institute for Cybersecurity sits behind a
form, so this fetches the widely used GitHub mirror and checks each file's row
and column counts against the published sizes before keeping it.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MIRROR = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {"KDDTrain+.txt": 125_973, "KDDTest+.txt": 22_544}
COLUMNS = 43  # 41 features + label + difficulty level


def fetch(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, rows in FILES.items():
        target = dest / name
        if target.exists():
            print(f"{target} already present")
            continue
        url = MIRROR + urllib.request.quote(name)
        print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        lines = data.decode("utf-8").strip().splitlines()
        if len(lines) != rows:
            sys.exit(f"{name}: expected {rows} rows, got {len(lines)}; not saved")
        if any(len(line.split(",")) != COLUMNS for line in lines[:1000]):
            sys.exit(f"{name}: expected {COLUMNS} comma-separated columns; not saved")
        target.write_bytes(data)
        print(f"saved {target} ({len(data) / 1e6:.1f} MB, {rows} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "nsl_kdd")
    fetch(parser.parse_args().dest)


if __name__ == "__main__":
    main()
