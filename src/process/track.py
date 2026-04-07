import polars as pl
from itertools import combinations

class Tracker:
    def __init__(self, logger:object):
        self.logger = logger

    def _get_total_variants_of_all_regimens(self, table, keys=None):
        keys = keys or ["regimen_cui", "variant_key"]
        return table.select(keys) \
            .unique() \
            .group_by("regimen_cui") \
            .agg(pl.col("variant_key").n_unique().alias("n_variant")) \
            .select(pl.col("n_variant").sum()) \
            .item()

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
    
    def build_summary(self, standard, all_keys, group_keys):
        """
        Takes cleaned frame (standard)
        and compare to checkpoint frame (all_keys)
        against group_keys
        """
        standard_sum = self._get_total_variants_of_all_regimens(standard, group_keys)
        all_keys_sum = self._get_total_variants_of_all_regimens(all_keys, group_keys)
        assert all_keys_sum == standard_sum, f"Mismatch in group splits! {all_keys_sum} != {standard_sum}"
        self.logger.info(f"[AUDIT] Number of variants: {standard.shape[0]} ({standard_sum} unique)")

        