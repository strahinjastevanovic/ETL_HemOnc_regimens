"""
Regimen Formatter – Smart regimen output generation.

Extracts logic to create minimal distinguishing regimen format:
- Removes info loss from naive filtering
- Preserves key group identifiers (regCode, shortString, regName, condition)
- Handles undefined conditions with logging
- Prevents silent data loss through detailed statistics
"""

import pandas as pd
import logging


def build_final_regimens(frame, logs_dir="logs"):
    """
    Create final regimens.tsv with minimal distinguishing format.
    
    Generates a curated regimen list preserving:
    - regCode: unique regimen identifier
    - shortString: collapsed dose/component representation
    - regName: human-readable regimen name
    - condition: clinical indication
    
    Removes duplicate rows while preserving all condition-regimen-shortString combinations.
    Handles undefined conditions separately with logging.
    
    Parameters:
    frame (pd.DataFrame): Full transformed regimen frame with columns:
        regCode, shortString, regName, condition, conditionCode, ...
    logs_dir (str): Directory for logging statistics
    
    Returns:
    pd.DataFrame: Curated regimens dataframe with columns:
        [regCode, shortString, regName, condition]
    """
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n[REGIMEN FORMATTER] Building final regimens output")
    logger.info(f"[INFO] Input frame shape: {frame.shape}")
    logger.info(f"[INFO] Input columns: {frame.columns.tolist()}")
    
    # Required columns for output
    required_cols = ['regCode', 'shortString', 'regName', 'condition', 'conditionCode']
    missing = [c for c in required_cols if c not in frame.columns]
    if missing:
        logger.error(f"[ERROR] Missing required columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")
    
    regimens_df = frame # shellow copy no need to take up memory
    
    # Separate undefined from defined conditions
    undefined_mask = regimens_df['conditionCode'] == 'undefined'
    undefined_regimens = regimens_df[undefined_mask].copy()
    defined_regimens = regimens_df[~undefined_mask].copy()
    
    # Log undefined condition statistics
    if not undefined_regimens.empty:
        logger.warning(f"\n[WARNING] Found {len(undefined_regimens)} rows with undefined conditions")
        logger.warning(f"[INFO] Unique condition-unknown regimens: {undefined_regimens['regName'].nunique()}")
        undefined_regs = undefined_regimens['regName'].unique().tolist()
        for reg in sorted(undefined_regs):
            count = len(undefined_regimens[undefined_regimens['regName'] == reg])
            logger.warning(f"  - {reg}: {count} row(s)")
        logger.info(f"[NOTE] Undefined regimens retained in output for reference")
    
    # Remove exact duplicates per condition-regimen-shortString combo
    # Keep first occurrence of each unique combination
    regimens_dedup = regimens_df.drop_duplicates(
        subset=['condition', 'regCode', 'shortString'],
        keep='first'
    )
    
    logger.info(f"\n[INFO] Deduplicated regimens shape: {regimens_dedup.shape}")
    logger.info(f"[INFO] Removed {len(regimens_df) - len(regimens_dedup)} duplicate rows")
    
    # Compute statistics
    unique_shortstrings = regimens_dedup['shortString'].nunique()
    unique_regimens = regimens_dedup['regCode'].nunique()
    unique_conditions = regimens_dedup['condition'].nunique()
    
    logger.info(f"\n[STATISTICS]")
    logger.info(f"[INFO] Unique shortStrings: {unique_shortstrings}")
    logger.info(f"[INFO] Unique regimens: {unique_regimens}")
    logger.info(f"[INFO] Unique conditions: {unique_conditions}")
    
    # Analyze shortString distribution across conditions
    shortstring_condition_counts = regimens_dedup.groupby('shortString')['condition'].nunique()
    avg_conditions_per_shortstring = shortstring_condition_counts.mean()
    max_conditions_per_shortstring = shortstring_condition_counts.max()
    
    logger.info(f"[INFO] Avg conditions per shortString: {avg_conditions_per_shortstring:.2f}")
    logger.info(f"[INFO] Max conditions per shortString: {max_conditions_per_shortstring}")
    
    # Find most common shortStrings
    shortstring_counts = regimens_dedup['shortString'].value_counts()
    logger.info(f"[INFO] Most common shortStrings:")
    for ss, count in shortstring_counts.head(5).items():
        logger.info(f"  - {ss}: {count} regimen-condition pairs")
    
    # Analyze regimen-shortString distribution
    multishortstring_regimens = (
        regimens_dedup.groupby('regCode')['shortString'].nunique()
    )
    regimens_with_multiple_shortstrings = (multishortstring_regimens > 1).sum()
    
    logger.info(f"\n[INFO] Regimens with multiple shortStrings: {regimens_with_multiple_shortstrings}")
    logger.info(f"[INFO] This indicates regimens with conditional variant representations")
    
    final_regimens = regimens_dedup.sort_values(
        by=['condition', 'regCode', 'shortString']
    ).reset_index(drop=True)
    
    logger.info(f"\n[INFO] Final regimens output shape: {final_regimens.shape}")
    logger.info(f"[INFO] Output columns: {final_regimens.columns.tolist()}")
    
    return final_regimens


def analyze_shortstring_regimen_mapping(frame, logs_dir="logs"):
    """
    Analyze many-to-many relationship between shortStrings and regimens.
    Logs detailed mapping statistics for validation.
    
    Parameters:
    frame (pd.DataFrame): Regimen dataframe with shortString and regName columns
    logs_dir (str): Directory for logging
    
    Returns:
    dict: Statistics dictionary with mapping analysis
    """
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n[SHORTSTRING-REGIMEN MAPPING ANALYSIS]")
    
    # ShortStrings per regimen
    shortstrings_per_regimen = frame.groupby('regName')['shortString'].nunique()
    regimens_with_multiple = (shortstrings_per_regimen > 1).sum()
    
    logger.info(f"[INFO] Regimens with multiple shortStrings: {regimens_with_multiple}")
    logger.info(f"[INFO] Avg shortStrings per regimen: {shortstrings_per_regimen.mean():.2f}")
    logger.info(f"[INFO] Max shortStrings per regimen: {shortstrings_per_regimen.max()}")
    
    # Regimens per shortString
    regimens_per_shortstring = frame.groupby('shortString')['regName'].nunique()
    shortstrings_shared = (regimens_per_shortstring > 1).sum()
    
    logger.info(f"[INFO] ShortStrings shared across regimens: {shortstrings_shared}")
    logger.info(f"[INFO] Avg regimens per shortString: {regimens_per_shortstring.mean():.2f}")
    logger.info(f"[INFO] Max regimens per shortString: {regimens_per_shortstring.max()}")
    
    # Most shared shortStrings
    logger.info(f"[INFO] Most shared shortStrings (across regimens):")
    most_shared = regimens_per_shortstring.nlargest(5)
    for ss, count in most_shared.items():
        logger.info(f"  - {ss}: {count} regimens")
    
    # return {
    #     'regimens_with_multiple_shortstrings': regimens_with_multiple,
    #     'shortstrings_shared_across_regimens': shortstrings_shared,
    #     'avg_shortstrings_per_regimen': shortstrings_per_regimen.mean(),
    #     'avg_regimens_per_shortstring': regimens_per_shortstring.mean(),
    # }
    return 1
