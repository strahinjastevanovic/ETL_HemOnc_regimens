import polars as pl

class Resolver:
    def __init__(self, logger:object, reporter:object):
        self.logger = logger
        self.reporter = reporter
        self.f_label  = "resolved"

    def resolve_partial_variants(
        self,
        frame: pl.DataFrame, 
        group_keys=['condition_cui', 'regimen_cui', 'variant_cui'],
        ):
        resolved_groups = []

        for i, group_df in frame.group_by(group_keys, maintain_order=True):

            updated_rows = []

            # Group by component_cui within this group
            for component_val, component_df in group_df.group_by("component_cui", maintain_order=True):
                unique_steps = component_df["step_number"].n_unique()
                unique_timings = component_df["timing_sequence"].n_unique()

                if unique_steps != 1 and unique_timings == 1:
                    # Multiple steps, single timing → keep only first
                    updated_rows.append(component_df[0])
                elif unique_steps == 1 and unique_timings != 1:
                    # Single step, multiple timings → keep all
                    updated_rows.append(component_df)

            # Combine kept rows from this group
            if updated_rows:
                if isinstance(updated_rows[0], pl.DataFrame):
                    group_result = pl.concat(updated_rows, how='vertical')
                else:
                    group_result = pl.DataFrame(updated_rows)
                resolved_groups.append(group_result)

        resolved = pl.concat(resolved_groups, how='vertical')

        self.reporter.to_tsv(resolved, f"multi_part_sigs.{self.f_label}")

        uniq_in = frame.select(group_keys).unique()
        uniq_out = resolved.select(group_keys).unique()
        n_variants_in = uniq_in.height
        n_variants_out = uniq_out.height
        self.logger.info(f"[RESOLVED] Multi-parted variants fixed: {n_variants_out} / {n_variants_in}")

        join_cols = frame.columns
        dropped = frame.join(resolved, on=join_cols, how="anti")
    
        return resolved, dropped

    @staticmethod
    def combine(table_list):
        return pl.concat(table_list, how="vertical")
