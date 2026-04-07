"""
athena_snapshot.py
------------------
Utility for packing and unpacking the three Athena query-result CSVs
(condition_concepts, drug_concepts, sigs_w_conditions) into a single
tagged pickle bundle.

Bundle format (dict):
    {
        "tag":                 str,          # e.g. "feb2025"
        "condition_concepts":  str,          # raw CSV text
        "drug_concepts":       str,          # raw CSV text
        "sigs_w_conditions":   str,          # raw CSV text
    }

CLI usage
---------
Pack three CSVs into a bundle:
    python -m tools.athena_snapshot pack \\
        --tag feb2025 \\
        --condition  OTHER_REF/condition_concepts.feb2025.csv \\
        --drugs      OTHER_REF/drug_concepts.feb2025.csv \\
        --sigs       OTHER_REF/sigs_w_conditions.feb2025.csv \\
        --out        OTHER_REF/AthenaFeb2025snapshot.pkl

Unpack a bundle into a target directory:
    python -m tools.athena_snapshot unpack \\
        --bundle OTHER_REF/AthenaFeb2025snapshot.pkl \\
        --out    output.next/
"""

from __future__ import annotations

import argparse
import pickle
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Default remote location of the pre-built snapshot bundle
# ---------------------------------------------------------------------------
_SNAPSHOT_URL = (
    "https://github.com/strahinjastevanovic/ETL_HemOnc_regimens"
    "/raw/athena_mirrors/OTHER_REF/AthenaFeb2025snapshot.pkl"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pack(
    tag: str,
    condition_concepts_path: str | Path,
    drug_concepts_path: str | Path,
    sigs_w_conditions_path: str | Path,
    out_path: str | Path,
) -> None:
    """Read three CSV files and write them as a single pickle bundle."""
    bundle = {
        "tag": tag,
        "condition_concepts": Path(condition_concepts_path).read_text(encoding="utf-8"),
        "drug_concepts":       Path(drug_concepts_path).read_text(encoding="utf-8"),
        "sigs_w_conditions":   Path(sigs_w_conditions_path).read_text(encoding="utf-8"),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[athena_snapshot] Packed tag={tag!r} → {out}")


def unpack(bundle_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """
    Unpack a bundle into out_dir, writing:
        condition_concepts.csv
        drug_concepts.csv
        sigs_w_conditions.csv

    Returns a dict mapping key → written Path.
    """
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    tag = bundle.get("tag", "unknown")
    written = unpack_bundle(bundle, out_dir)
    print(f"[athena_snapshot] Unpacked tag={tag!r} from {bundle_path} → {out_dir}/")
    return written


def describe(bundle_path: str | Path) -> None:
    """Print tag and line counts for a bundle without unpacking."""
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    print(f"tag: {bundle.get('tag', 'unknown')}")
    for key in ("condition_concepts", "drug_concepts", "sigs_w_conditions"):
        lines = bundle[key].count("\n")
        print(f"  {key}: {lines:,} lines")


def load_snapshot(out_dir: str | Path, url: str = _SNAPSHOT_URL) -> dict[str, Path]:
    """
    Download the pre-built Athena snapshot bundle from *url* and unpack it
    into *out_dir*, writing:
        condition_concepts.csv
        drug_concepts.csv
        sigs_w_conditions.csv

    Returns a dict mapping key → written Path.

    Used when DB_FRESH is not TRUE — no database credentials required.
    """
    print(f"[athena_snapshot] Fetching snapshot from:\n  {url}")
    with urllib.request.urlopen(url) as resp:
        bundle = pickle.loads(resp.read())

    tag = bundle.get("tag", "unknown")
    print(f"[athena_snapshot] Snapshot tag: {tag!r}")
    return unpack_bundle(bundle, out_dir)


def unpack_bundle(bundle: dict, out_dir: str | Path) -> dict[str, Path]:
    """Write a pre-loaded bundle dict into *out_dir*. Returns key → Path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for key in ("condition_concepts", "drug_concepts", "sigs_w_conditions"):
        dest = out / f"{key}.csv"
        dest.write_text(bundle[key], encoding="utf-8")
        written[key] = dest
    return written


# ---------------------------------------------------------------------------
# CLI entry point  (python -m tools.athena_snapshot …)
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.athena_snapshot",
        description="Pack/unpack Athena query snapshot bundles.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- pack ---
    p = sub.add_parser("pack", help="Pack three CSVs into a .pkl bundle")
    p.add_argument("--tag",       required=True, help="Snapshot label, e.g. feb2025")
    p.add_argument("--condition", required=True, metavar="CSV", help="condition_concepts CSV")
    p.add_argument("--drugs",     required=True, metavar="CSV", help="drug_concepts CSV")
    p.add_argument("--sigs",      required=True, metavar="CSV", help="sigs_w_conditions CSV")
    p.add_argument("--out",       required=True, metavar="PKL", help="Output .pkl path")

    # --- unpack ---
    u = sub.add_parser("unpack", help="Unpack a .pkl bundle into a directory")
    u.add_argument("--bundle", required=True, metavar="PKL", help="Input .pkl bundle")
    u.add_argument("--out",    required=True, metavar="DIR", help="Output directory")

    # --- describe ---
    d = sub.add_parser("describe", help="Print bundle metadata without unpacking")
    d.add_argument("--bundle", required=True, metavar="PKL")

    args = parser.parse_args(argv)

    if args.cmd == "pack":
        pack(args.tag, args.condition, args.drugs, args.sigs, args.out)
    elif args.cmd == "unpack":
        unpack(args.bundle, args.out)
    elif args.cmd == "describe":
        describe(args.bundle)


if __name__ == "__main__":
    _main(sys.argv[1:])
