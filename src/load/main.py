from load.log import Logger
from load.audit import (
    AuditColumnTypes,
    Tracker
)
from load.handlers import (
    Frame,
    NullValueHandlers,
    RegimenHandler,
    VariantHandler,
    PatternHandlers,
    SupplementaryHandler,
)
from load.report import Reporter, build_reports
from load.resolvers import (
    ResolverAllDays,
    ResolverIndefinite,
    ResolverParted,
    ResolverKey
)
class Preprocessor:
    def __init__(self, sigs_path, output_dir, log_dir=".", supplementary_file=None, sheet_config=None):
       
        self.logger     = Logger(log_dir, )
        self.audits     = AuditColumnTypes(log_dir, "PRE.audit.log")
        self.audits.audit(sigs_path)
        self.report_path = f"{output_dir}/report_tables"
        self.reporter   = Reporter(self.report_path) 
        self.config     = sheet_config
        self.s          = Frame().load_data(sigs_path)
        self.sf         = supplementary_file
        self.sigs_anatomy_essentials = [
            'variant_cui', 
            'allDays',
            "cycle_length_lb", 
            "cycle_length_ub",   
            "cycle_length_unit",
            "timing_sequence"
        ]

    def initialize(self, ):
        self.handle_null       = NullValueHandlers(self.logger, self.reporter)
        self.handle_regimen    = RegimenHandler(self.logger, self.reporter)
        self.handle_variant    = VariantHandler(self.logger, self.reporter)
        self.handle_pattern    = PatternHandlers(self.logger, self.reporter)
        self.handle_supp       = SupplementaryHandler(self.logger, self.reporter)
        self.resolve_days      = ResolverAllDays(self.logger, self.reporter)
        self.resolve_indef     = ResolverIndefinite(self.logger, self.reporter)
        self.resolve_parted    = ResolverParted(self.logger, self.reporter)
        self.track             = Tracker(self.logger)
        # self.resolve           = Resolver(self.logger, self.reporter)
        return self # enables chaining


    def run(self):
        """Process the data (clean NaN and handle duplicates).
        
        Keeps frames in mem. and utilizes polars frame backround
        """

        frame = self.s.clone()
        # ---- add Variant Key TBA Issue ----
        key_reslve = ResolverKey()
        frame = key_reslve.with_variant_key(frame)

        group_keys = ["condition_cui", "regimen_cui", "variant_key"]
        fields = self.sigs_anatomy_essentials
        supplementary_file = self.sf

        # ----------- 1 level subset block -component level dropouts, variants kept ------------
        frame = self.handle_supp.clean_by_role(frame, group_keys) 
        frame = self.handle_supp.clean_by_blacklist(frame, supplementary_file, group_keys) 
        frame = self.handle_supp.clean_by_blacklist_regimen(frame, supplementary_file, group_keys)

        # ----------- 2 level subset block -regimen level dropouts ------------
        frame = self.handle_null.handle_nan_in_condition(frame)
        frame = self.handle_null.handle_nan_in_group_keys(frame, group_keys)
        frame = self.handle_null.handle_null_in_sigs(frame, fields, group_keys)
        self.handle_regimen.log_regimen_level_stats(frame)
        frame = self.handle_regimen.filter_imbalanced(frame, group_keys)
        frame = self.handle_regimen.filter_rt(frame)

        # ----------- 3 - level subset block -variant level drouputs ------------

        # ------------ 3.0 - indefinite -----------
        frame = self.resolve_indef.timing_sequence(frame, group_keys)
        indefinite_df, frame  = self.handle_pattern.cycle_length_indeterminate(frame, fields)
       
        resolved_df  = self.resolve_indef.cycle_bounds(indefinite_df, group_keys)
        # print(resolved_df.filter(pl.col("cycle_length_ub").str.contains("NUB")))
        frame = self.resolve_parted.combine([frame, resolved_df])
        # print(1/0)
        checkpoint_df  = self.handle_variant.save_checkpoint(frame, group_keys) # safeguard
       
        # ----------- 3.1 - parted -----------------
        main_df, multi_df    = self.handle_variant.handle_partial(frame, group_keys)
        mdf_resolved           = self.resolve_parted.resolve_partial(multi_df, group_keys)
        main_df               = self.resolve_parted.combine([main_df, mdf_resolved])

        # ----------- 3.2 - all days pattern handler -----------------
        main_df, invalid_df_1 = self.handle_pattern.all_days_pattern_handler(main_df, group_keys)
        valid_df_1 = self.resolve_days.resolve_ex(invalid_df_1, group_keys)
        main_df = self.resolve_parted.combine([main_df, valid_df_1])
        #  ----------- 4 - Tracker filtered -----------
        # funny_list = [
        #     ("invalid_1", invalid_df_1), 
        # ]
        # funny_df = self.track.concat_with_overlap_diagnostics(subsets=funny_list, group_keys=group_keys)

        # ------------ 5 - Summary report -----------
        self.track.log_summary(main_df, checkpoint_df, group_keys)
        self.reporter.report(main_df, "preproc_cleaned", pattern="No idiosyncracy. Clean", field="-", status="H")

        # ---- << final frame >> ----
        self.processed = main_df.clone()
        return self

    def get_processed(self):
        """Returns the final cleaned frame."""
        if not hasattr(self, "processed"):
            raise RuntimeError("Data has not been processed yet.")
        return self.processed
    
    def build_reports(self):
        build_reports(self.config, self.report_path, self.report_path)