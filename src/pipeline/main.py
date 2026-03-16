from pipeline.log import Logger
from pipeline.audit import (
    AuditColumnTypes,
    Tracker
)
from pipeline.handlers import (
    Frame,
    NullValueHandlers,
    RegimenHandler,
    VariantHandler,
    PatternHandlers,
    SupplementaryHandler,
)
from pipeline.report import Reporter, build_reports
from pipeline.resolvers import Resolver

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
        self.group_keys = ["condition_cui", "regimen_cui", "variant_cui"]
        self.sigs_anatomy_essentials = [
            'variant_cui', 
            'allDays',
            "cycle_length_lb", 
            "cycle_length_ub",   
            "cycle_length_unit",
            "timing_sequence"
        ]

    def initialize(self, ):
        self.null_handlers       = NullValueHandlers(self.logger, self.reporter)
        self.regimen_handler     = RegimenHandler(self.logger, self.reporter)
        self.variant_handler     = VariantHandler(self.logger, self.reporter)
        self.pattern_handlers    = PatternHandlers(self.logger, self.reporter)
        self.supp_handler        = SupplementaryHandler(self.logger, self.reporter)
        self.tracker             = Tracker(self.logger)
        self.resolver            = Resolver(self.logger, self.reporter)
        return self # enables chaining


    def run(self):
        """Process the data (clean NaN and handle duplicates).
        
        Keeps frames in mem. and utilizes polars frame backround
        """

        frame = self.s.clone()
        group_keys = self.group_keys
        fields = self.sigs_anatomy_essentials
        supplementary_file = self.sf

        # ----------- 1 level subset block -component level dropouts, variants kept ------------
        frame = self.supp_handler.clean_by_role(frame, group_keys) 

        # ----------- 2 level subset block -regimen level dropouts ------------
        frame = self.null_handlers.handle_nan_in_condition(frame)
        frame = self.null_handlers.handle_nan_in_group_keys(frame, group_keys)
        frame = self.null_handlers.handle_null_in_sigs(frame, fields, group_keys)
        self.regimen_handler.log_regimen_level_stats(frame)
        frame = self.regimen_handler.filter_imbalanced(frame, group_keys)
        frame = self.regimen_handler.filter_rt(frame)
        self.pattern_handlers.log_indefinite_cycle_length(frame, fields)

        # ----------- 3 - level subset block -variant level drouputs ------------
        checkpoint_df = self.variant_handler.create_checkopoint_frame(frame, group_keys) # safeguard
       
        # ----------- 3.1 - parted -----------------
        single_df, multi_df    = self.variant_handler.handle_partial_variants(frame, group_keys)
        mdf_resolved           = self.resolver.resolve_partial_variants(multi_df, group_keys)
        mixed_df               = self.resolver.combine([single_df, mdf_resolved])

        # ----------- 3.2 - all days pattern handler -----------------
        cleaned_df, invalid_df_1 = self.pattern_handlers.all_days_pattern_handler(mixed_df, group_keys)

        #  ----------- 4 - Tracker filtered -----------
        funny_list = [
            ("invalid_1", invalid_df_1), 
        ]
        funny_df = self.tracker.concat_with_overlap_diagnostics(subsets=funny_list, group_keys=group_keys)

        # ------------ 5 - Summary report -----------
        self.tracker.log_summary(cleaned_df, funny_df, checkpoint_df, group_keys)
        self.reporter.report(cleaned_df, "preproc_cleaned", pattern="No idiosyncracy. Clean", field="-", status="H")
        
        # ---- << final frame >> ----
        self.processed = cleaned_df.clone()
        return self

    def get_processed(self):
        """Returns the final cleaned frame."""
        if not hasattr(self, "processed"):
            raise RuntimeError("Data has not been processed yet.")
        return self.processed
    
    def build_reports(self):
        build_reports(self.config, self.report_path, self.report_path)