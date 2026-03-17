import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json
from utils import set_path, to_pkl, from_pkl


# Project root staging directory
STAGING_DIR = Path("regression_staging")
BASELINE_NAME = "baseline"
REGRESSION_BRANCH = "regression-data"


def setup_stage():
    """Ensure staging directory exists."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)


def create_checkpoint(output_dir, checkpoint_name=None):
    """Create a checkpoint pickle from output directory.
    
    Args:
        output_dir: Path to pipeline output directory
        checkpoint_name: Name for the checkpoint file (without .pkl.zip extension)
    
    Returns:
        Path to created checkpoint file
    """
    setup_stage()
    
    output_path = Path(output_dir)
    
    if checkpoint_name is None:
        checkpoint_name = output_path.name
    
    checkpoint_file = STAGING_DIR / f"{checkpoint_name}.pkl.zip"
    
    # Get TSV files from output directory
    _, tsv_files = set_path(output_dir)
    
    # Pkl + zip the Checkpoint file
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


def baseline_gh_push(version):
    """Handle git operations for baseline update.
    
    Args:
        version: Version string for commit message
    """
    print(f"\nPushing baseline to {REGRESSION_BRANCH}...")
    
    # Check if remote branch exists
    remote_exists = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", REGRESSION_BRANCH],
        capture_output=True
    ).stdout.strip()
    
    # Stage only regression_staging/
    subprocess.run(["git", "add", "-f", str(STAGING_DIR)], check=True)
    
    # Get tree hash of staged files
    tree_hash = subprocess.run(
        ["git", "write-tree"],
        capture_output=True, text=True, check=True
    ).stdout.strip()
    
    # Create commit object
    commit_msg = f"chore: update baseline [baseline {version}]"
    if remote_exists:
        parent = subprocess.run(
            ["git", "rev-parse", f"origin/{REGRESSION_BRANCH}"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        commit_hash = subprocess.run(
            ["git", "commit-tree", tree_hash, "-p", parent, "-m", commit_msg],
            capture_output=True, text=True, check=True
        ).stdout.strip()
    else:
        commit_hash = subprocess.run(
            ["git", "commit-tree", tree_hash, "-m", commit_msg],
            capture_output=True, text=True, check=True
        ).stdout.strip()
    
    # Push commit to regression-data branch
    subprocess.run(["git", "push", "origin", f"{commit_hash}:refs/heads/{REGRESSION_BRANCH}"], check=True)
    
    # Clean up - unstage everything
    subprocess.run(["git", "reset", "HEAD"], check=True)
    
    print(f"* Baseline pushed to {REGRESSION_BRANCH}")


def baseline_mode(version, output_dir):
    """Create baseline and push to regression branch."""
    
    if not Path(output_dir).exists():
        print(f"Error: {output_dir} directory not found")
        print(f"Run pipeline first: ./RunScript.sh -out {output_dir}")
        sys.exit(1)
    
    print(f"Creating baseline from {output_dir} (version: {version})...")
    checkpoint_file = create_checkpoint(output_dir, BASELINE_NAME)
    
    # Create history entry
    history_file = STAGING_DIR / "history.csv"
    if not history_file.exists():
        with open(history_file, "w") as f:
            f.write("date,version\n")
    
    with open(history_file, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M')},{version}\n")
    
    print(f"* Baseline created: {checkpoint_file}")
    print(f"* History updated: {history_file}")
    
    # Git operations - no branch switching needed
    baseline_gh_push(version)


def test_mode():
    """Run regression test using staged checkpoint."""
    checkpoints = list(STAGING_DIR.glob("*.pkl.zip"))
    
    if not checkpoints:
        print(f"Error: No checkpoint found in {STAGING_DIR}")
        print("Create one first: python src/tests/regression/main.py --pkl output")
        sys.exit(1)
    
    # Filter out baseline.pkl.zip
    checkpoints = [c for c in checkpoints if c.name != f"{BASELINE_NAME}.pkl.zip"]
    
    if len(checkpoints) != 1:
        print(f"Error: Expected 1 checkpoint, found {len(checkpoints)}")
        print(f"Files: {[c.name for c in checkpoints]}")
        sys.exit(1)
    
    checkpoint = checkpoints[0]
    baseline_file = STAGING_DIR / f"{BASELINE_NAME}.pkl.zip"
    
    if not baseline_file.exists():
        print("Error: baseline.pkl.zip not found in staging directory")
        print("This should be fetched by GitHub Action from regression-data branch")
        sys.exit(1)
    
    print(f"* Comparing {checkpoint.name} against baseline...")
    compare_pickles(checkpoint, baseline_file)
    
    print("* Generating HTML report...")
    from create_report import create_interactive_report
    create_interactive_report()
    
    print(f"* Regression test complete")
    print(f"* HTML report: html/final_output.html")
    
    # Cleanup
    Path("comparison_report.json").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Regression testing tool")
    parser.add_argument("--pkl", metavar="OUTPUT_DIR", help="Create checkpoint from output directory")
    parser.add_argument("--baseline", nargs=2, metavar=("VERSION", "OUTPUT_DIR"), 
                       help="Create baseline and commit (e.g., --baseline v1.2.0 output/)")
    parser.add_argument("--test", action="store_true", help="Run regression test (used by GitHub Action)")
    
    args = parser.parse_args()
    
    # Check mutual exclusivity
    modes = sum([bool(args.pkl), bool(args.baseline), args.test])
    if modes != 1:
        parser.print_help()
        sys.exit(1)
    
    if args.pkl:
        pkl_mode(args.pkl)
    elif args.baseline:
        version, output_dir = args.baseline
        baseline_mode(version, output_dir)
    elif args.test:
        test_mode()


if __name__ == "__main__":
    main()
