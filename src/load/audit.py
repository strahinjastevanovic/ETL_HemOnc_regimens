from load.log import Logger 

from itertools import combinations
import json
import polars as pl 
from tqdm import tqdm
from collections import Counter

class AuditColumnTypes:
    def __init__(self, log_dir, filename, default_file="s_frame.schema.json"):
        self.logger = Logger(log_dir, filename)
        self.schema_json_path = f"{log_dir}/{default_file}"

    def audit(self, data_path: str, sample_rows: int = 10000, int_threshold: float = 0.8):
        df_sample = pl.read_csv(data_path, infer_schema_length=sample_rows)
        field_dtypes = []
        schema = {}

        for col in tqdm(df_sample.columns, desc="Auditing column types"):
            values = df_sample[col].drop_nulls().unique().to_list()
            n = len(values)
            num_int = 0
            for v in values:
                try:
                    int(v)
                    num_int += 1
                except:
                    continue

            int_ratio = num_int / n if n else 0
            dtype = "Int64" if int_ratio >= int_threshold else "Utf8"
            field_dtypes.append(f"{col}\t{int_ratio:.2f}\t{dtype}")
            schema[col] = dtype

        # Log types
        self.logger.info("column\tint_ratio\ttype_inferred\n")
        self.logger.info("\n".join(field_dtypes))
        # Save schema
        with open(self.schema_json_path, "w") as f:
            json.dump(schema, f, indent=2)


class Tracker:
    def __init__(self, logger:object):
        self.logger = logger

    def concat_with_overlap_diagnostics(self, subsets:list, group_keys, remove_ovelaps=False):
        overlap = lambda *args: (
            lambda a, b, name: (
                lambda o: self.logger.warning(f"[AUDIT] {o.height} overlapping group_keys in {name}")
                if o.height > 0 else None
            )(a.select(group_keys).unique().join(b.select(group_keys).unique(), on=group_keys, how="inner"))
        )(*args)

        if len(subsets) > 1:
            pairs = [((a[0], a[1]), (b[0], b[1])) for a, b in combinations(subsets, 2)]
            for pair in pairs:
                overlap(pair[0][1], pair[1][1], f"{pair[0][0]} vs {pair[1][0]}")
        return pl.concat([subset[1] for subset in subsets])
    
    def log_summary(self, standard, all_keys, group_keys):
        """
        Takes cleaned frame (standard)
        and compare to checkpoint frame (all_keys)
        against group_keys
        """
        keys = ["regimen_cui", "variant_key"]

        get_total_variants_of_all_regimens = lambda table: table.select(keys) \
            .unique() \
            .group_by("regimen_cui") \
            .agg(pl.col("variant_key").n_unique().alias("n_variant")) \
            .select(pl.col("n_variant").sum()) \
            .item()

        all_keys_sum = get_total_variants_of_all_regimens(all_keys)
        standard_sum = get_total_variants_of_all_regimens(standard)
        
        assert all_keys_sum == standard_sum, f"Mismatch in group splits! {all_keys_sum} != {standard_sum}"
        
        self.logger.info(f"[AUDIT] Number of variants: {standard.shape[0]} ({standard_sum} unique)")



    def log_summary_outliers(self, standard, funny, all_keys, group_keys):
        """
        Takes cleaned frame (standard)
        all dropped records (funny) 
        and compare to checkpoint frame (all_keys)
        against group_keys
        """
        keys = ["regimen_cui", "variant_key"]

        # shared_keys = (
        #     standard.select(keys).unique()
        #     .join(
        #         funny.select(keys).unique(),
        #         on=keys,
        #         how="inner"
        #     )
        # )

        # if shared_keys.height > 0:
        #     self.logger.warning(f"[AUDIT] Dropping {shared_keys.height} overlapping (regimen, variant) keys from all sets")
        #     self.logger.warning(f"[AUDIT] Shared keys in checkpoint: {all_keys.join(shared_keys, how="inner", on=keys).shape}")
        #     self.logger.warning(f"[AUDIT] Shared keys in standard: {standard.join(shared_keys, how="inner", on=keys).shape}")
        #     self.logger.warning(f"[AUDIT] Shared keys in funny: {funny.join(shared_keys, how="inner", on=keys).shape}")
        #     standard = standard.join(   shared_keys, on=keys, how="anti")
        #     funny    = funny.join(      shared_keys, on=keys, how="anti")
        #     all_keys = all_keys.join(   shared_keys, on=keys, how="anti")

        get_total_variants_of_all_regimens = lambda table: table.select(keys) \
            .unique() \
            .group_by("regimen_cui") \
            .agg(pl.col("variant_key").n_unique().alias("n_variant")) \
            .select(pl.col("n_variant").sum()) \
            .item()

        all_keys_sum = get_total_variants_of_all_regimens(all_keys)
        standard_sum = get_total_variants_of_all_regimens(standard)
        funny_sum    = get_total_variants_of_all_regimens(funny) 
        
        funny_unique = funny.select(group_keys).n_unique()
        
        assert all_keys_sum == standard_sum + funny_sum, f"Mismatch in group splits! {all_keys_sum} != {standard_sum} + {funny_sum}"
        
        self.logger.info(f"[AUDIT] Number of vanilla variants: {standard.shape[0]} ({standard_sum} unique)")
        self.logger.info(f"[AUDIT] Number of funny variants: {funny_unique} ({funny_unique} unique)")
