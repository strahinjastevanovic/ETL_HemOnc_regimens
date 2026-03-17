import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json
from utils import set_path, to_pkl, from_pkl

STAGING_DIR = Path("regression_staging")
STAGING_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_NAME = "baseline"


def create_checkpoint(output_dir, checkpoint_name = None):
    """Create a checkpoint pickle from output directory.
    
    Args:
        output_dir: Path to assembler output directory
    
    Returns:
        Path to created checkpoint file
    """

    output_path = Path(output_dir)
    if not checkpoint_name:
        checkpoint_name = output_path.name
    checkpoint_file = STAGING_DIR / f"{checkpoint_name}.pkl.zip"
    
    # Get TSV files from output directory
    _, tsv_files = set_path(output_dir)
    
    # Pkl + zip the checkpoint
    to_pkl(checkpoint_file, tsv_files)
    
    return checkpoint_file


def compare_pickles(new_pkl, ref_pkl):
    """Compare two pickle checkpoints and generate comparison report."""
    new_data = from_pkl(new_pkl)
    ref_data = from_pkl(ref_pkl)
    
    out = {}
    all_tables = (set(new_data.keys()) | set(ref_data.keys())) - {"__metadata__"}
    
    for table in sorted(all_tables):
        if table not in new_data:
            out[table] = f"skipped, missing in {new_pkl}"
        elif table not in ref_data:
            out[table] = f"skipped, missing in {ref_pkl}"
        else:
            from compare import compare_frames
            out[table] = compare_frames(ref_data[table], new_data[table])
    
    # Save to project root
    report_file = Path("comparison_report.json")
    with open(report_file, "w") as f:
        json.dump(out, f, indent=4)
    
    print(f"* Comparison report saved: {report_file}")
    return report_file


def pkl_mode(output_dir):
    """Create checkpoint from output directory (for testing)."""

    if Path(output_dir).name == BASELINE_NAME:
        checkpoint_name = f"{BASELINE_NAME}-user-input"
    else:
        checkpoint_name = None 
    checkpoint_file = create_checkpoint(output_dir, checkpoint_name)
    
    print(f"* Checkpoint created: {checkpoint_file}")
    print("\nNext steps:")
    print('  git commit -m "feat: your changes [test-regression]"')
    print("  git push")


def baseline_mode(version, output_dir):
    """Create baseline and commit/push automatically."""
    
    if not Path(output_dir).exists():
        print(f"Error: {output_dir} directory not found")
        print(f"Run pipeline first: ./RunScript.sh -out {output_dir}")
        sys.exit(1)
    
    print(f"Creating `baseline` from {output_dir} (version: {version})...")
    checkpoint_file = create_checkpoint(output_dir, BASELINE_NAME)
    
    # Create history entry
    history_file = STAGING_DIR / "history.csv"
    if not history_file.exists():
        with open(history_file, "w") as f:
            f.write("date,version\n")
    
    with open(history_file, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d')},{version}\n")
    
    print(f"* Baseline created: {checkpoint_file}")
    print(f"* History updated: {history_file}")
    
    # Git operations
    print("\nCommitting and pushing baseline...")
    subprocess.run(["git", "add", str(STAGING_DIR)], check=True)
    subprocess.run(["git", "commit", "-m", f"chore: updating baseline [baseline {version}]"], check=True)
    subprocess.run(["git", "push"], check=True)
    
    print(f"\n* Baseline committed and pushed")
    print(f"  GitHub Action will update regression-data branch")


def test_mode():
    """Run regression test using staged checkpoint."""
    checkpoints = list(STAGING_DIR.glob("*.pkl.zip"))
    
    if not checkpoints:
        print(f"Error: No checkpoint found in {STAGING_DIR}")
        print("Create one first: python src/tests/regression/main.py --pkl output")
        sys.exit(1)
    
    if len(checkpoints) > 1:
        # Filter out baseline.pkl.zip
        checkpoints = [c for c in checkpoints if c.name != "baseline.pkl.zip"]
    
    if len(checkpoints) != 1:
        print(f"Error: Expected 1 checkpoint, found {len(checkpoints)}")
        print(f"Files: {[c.name for c in checkpoints]}")
        sys.exit(1)
    
    checkpoint = checkpoints[0]
    baseline_file = STAGING_DIR / "baseline.pkl.zip"  # Fetched by GitHub Action to STAGING_DIR
    
    if not baseline_file.exists():
        print(f"Error: baseline.pkl.zip not found in {STAGING_DIR}")
        print("This should be fetched by GitHub Action from regression-data branch")
        sys.exit(1)
    
    print(f"* Comparing {checkpoint.name} against baseline...")
    report_file = compare_pickles(checkpoint, baseline_file)
    
    print("* Generating HTML report...")
    from create_report import create_interactive_report
    create_interactive_report()
    
    print("* Regression test complete. See html/final_output.html")


def main():
    parser      = argparse.ArgumentParser(description="Regression testing tool")
    subparsers  = parser.add_subparsers(dest="command", required=True)
    
    # --pkl mode
    pkl_parser = subparsers.add_parser("pkl", help="Create checkpoint from output directory")
    pkl_parser.add_argument("output_dir", help="Path to pipeline output directory")
    
    # --baseline mode
    baseline_parser = subparsers.add_parser("baseline", help="Create baseline and commit/push")
    baseline_parser.add_argument("version", help="Version string (e.g., v1.2.0)")
    baseline_parser.add_argument("output_dir", help="Path to pipeline output directory")
    
    # --test mode
    test_parser = subparsers.add_parser("test", help="Run regression test (GitHub Action)")
    
    args = parser.parse_args()
    
    if args.command == "pkl":
        pkl_mode(args.output_dir)
    elif args.command == "baseline":
        baseline_mode(args.version, args.output_dir, args.checkpoint_name)
    elif args.command == "test":
        test_mode()


if __name__ == "__main__":
    main()
