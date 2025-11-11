from pipeline.log import Logger 

from itertools import combinations
import json
import polars as pl 
from tqdm import tqdm

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


class Sumstats:
    def __init__(self, logger:object):
        self.logger = logger

    def concat_with_overlap_diagnostics(self, subsets:list, group_keys):
        overlap = lambda *args: (
            lambda a, b, name: (
                lambda o: self.logger.debug(f"[OVERLAP] {o.height} overlapping group_keys in {name}")
                if o.height > 0 else None
            )(a.select(group_keys).unique().join(b.select(group_keys).unique(), on=group_keys, how="inner"))
        )(*args)

        if len(subsets) > 1:
            pairs = [((a[0], a[1]), (b[0], b[1])) for a, b in combinations(subsets, 2)]
            for pair in pairs:
                overlap(pair[0][1], pair[1][1], f"{pair[0][0]} vs {pair[1][0]}")
        return pl.concat([subset[1] for subset in subsets])
    
    def log_summary(self, standard, funny, all_keys, group_keys):
        """
        Takes cleaned frame (standard)
        all dropped records (funny) 
        and compare to checkpoint frame (all_keys)
        against group_keys
        """
        all_keys_regimens_unique = (
            all_keys
            .select(["regimen_cui", "variant_cui"])
            .unique()  # filters duplicated variants
            .group_by("regimen_cui")
            .agg(pl.col("variant_cui").n_unique().alias("n_variant"))
            .select(pl.col("n_variant").sum())
            .item()
        )

        standard_regimens_unique = ( 
            standard
            .select(["regimen_cui", "variant_cui"])
            .unique() 
            .group_by("regimen_cui")
            .agg(pl.col("variant_cui").n_unique().alias("n_variant"))
            .select(pl.col("n_variant").sum())
            .item()
        )

        funny_regimens_unique = ( 
            funny
            .select(["regimen_cui", "variant_cui"])
            .unique()
            .group_by("regimen_cui")
            .agg(pl.col("variant_cui").n_unique().alias("n_variant"))
            .select(pl.col("n_variant").sum())
            .item()
        )
        
        funny_unique = funny.select(group_keys).n_unique()
        
        assert all_keys_regimens_unique == standard_regimens_unique + funny_regimens_unique, f"Mismatch in group splits! {all_keys_regimens_unique} != {standard_regimens_unique} + {funny_regimens_unique}"
        
        self.logger.info(f"[AUDIT] Number of vanilla variants: {standard.shape[0]} ({standard_regimens_unique} unique)")
        self.logger.info(f"[AUDIT] Number of funny variants: {funny_unique} ({funny_regimens_unique} unique)")
