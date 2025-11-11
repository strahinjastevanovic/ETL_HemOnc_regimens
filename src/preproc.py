from pipeline.main import Preprocessor

def preprocessing(
    sigs_file=".",
    output_dir="workdir",
    log_dir= "log_dir",
    supplementary_file=".",
    sheet_config=None,
):

    print("[INFO] Starting preprocessing run...")
    proc = Preprocessor(
        sigs_path=sigs_file,
        output_dir=output_dir,
        log_dir=log_dir,
        supplementary_file=supplementary_file,
        sheet_config=sheet_config
    ).initialize().run()

    dp = proc.get_processed()

    proc.build_reports()

    dp.write_parquet(f"{output_dir}/s_frame.parquet") 
    dp.write_csv(f"{output_dir}/s_frame.tsv", separator="\t") 

    print("[INFO] Output files written.")

