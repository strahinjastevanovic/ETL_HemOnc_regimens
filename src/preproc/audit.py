from preproc.log import Logger 
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