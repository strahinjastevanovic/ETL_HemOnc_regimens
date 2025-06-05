import os
import re
import json
import logging
import polars as pl


def clean_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", str(text)).strip().lower()


def proc_blist_naive(blist: list) -> list:
    blist = [l for subli in [s.split("(") for s in blist] for l in subli]
    return [s.strip(")").strip().lower() for s in blist]


def get_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def apply_addapters(
    frame_path="s_frame.parquet",
    output_dir="results",
    supplementary=None,
    logs_dir="logs"
):
    """Takes serialized format and outputs final processed TSV path."""

    logger = get_logger(f"{logs_dir}/PROC.addapters.log")
    os.makedirs(output_dir, exist_ok=True)


    # # Step 1: Read input
    dp = pl.read_parquet(frame_path)
    logger.warning(f"Input frame shape: {dp.shape}")

    # Use meta-component not to destruct original component
    dp = dp.with_columns(
        pl.col("component").cast(str).map_elements(clean_text).alias("meta_component")
    )

    # Handle Blacklist logic
    drop_summary_msg = "No blacklist applied."
    drop_components = []
    if supplementary:
        blacklist_json = json.load(open(supplementary, "r"))
        bl1 = proc_blist_naive(blacklist_json.get("custom_2024_sigs_curated", []))
        blacklist_set = set(bl1)

        dp = dp.with_columns(
            pl.col("meta_component")
            .is_in(blacklist_set)
            .alias("tmp_drop_idx")
        )

        # Count how many found in blacklist
        found_count = dp.filter(pl.col("tmp_drop_idx")).select(pl.count()).item()
        drop_components = dp.filter(pl.col("tmp_drop_idx")).select("component").unique().to_series().to_list()
        drop_summary_msg = f"Found {len(blacklist_set)} blacklist items. Will drop {found_count} records."
        logger.info(drop_summary_msg)
        for comp in drop_components:
            logger.info(f"DROPPED component: {comp}")
    else:
        dp = dp.with_columns(pl.lit(False).alias("tmp_drop_idx"))

    # Assert checks before filtering
    assert dp.filter(pl.col("meta_component") == "thalidomide").height > 0, "❌ Thalidomide lost as final check!"
    assert dp.filter(pl.col("meta_component") == "bisophonate").height == 0, "❌ Bisophonate still present!"

    # filter + cleanup
    dp = dp.filter(~pl.col("tmp_drop_idx"))
    dp = dp.with_columns(pl.lit("all").alias("metaCondition"))
    dp = dp.drop(["tmp_drop_idx", "meta_component"])

    # output processed frame
    logger.warning(f"Output frame shape: {dp.shape}")
    out_path = os.path.join(output_dir, f"{os.path.basename(frame_path)}".replace(".parquet", "_cleaned.parquet"))
    dp.write_parquet(out_path)
    
    # tmp
    out_path_tsv = os.path.join(output_dir, f"{os.path.basename(frame_path)}".replace(".parquet", "_cleaned.tsv"))
    dp.write_csv(out_path_tsv, separator="\t")

    return out_path
