import pandas as pd
from sqlalchemy import create_engine
import logging 
import os

from queries.vocab_query import query_valid_drugs as drugs_sql, query_conditions as condition_sql

def get_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)  

    logger = logging.getLogger(log_path)  # use log path as name to allow multiple loggers
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode='w')  # overwrite each time
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def log_column_differences(df, col1, col2, logger):
    logger.info("Different names in regimen (HemOnc) and regimen (Athena-OMOP):")
    found=False
    for i, (v1, v2) in enumerate(zip(df[col1], df[col2])):
        if v1 != v2:
            found=True
            logger.info(f'Row {i} - {col1} value: {v1} --- {col2} value: {v2}')

    if not found:
        logger.info("NONE")

def log_missing(df, col1, logger):
   logger.info(f'Query Size: {df[col1].notna().sum()} - Coverage: {round(df.c1_name.notna().sum() / len(df) * 100, 2)}%')
   logger.info(f'Missing {df[col1].isna().sum()} regimen to condition records.')

def log_shape(df, tag, logger):
   logger.info(f"Shape {tag}: {df.shape}")

def run_query_conditions(engine, vocab_file, input_file, log_dir, output_file):
  if os.path.exists(output_file):
      return 1
  df = pd.read_sql(condition_sql, engine)
  df.to_csv(vocab_file, index=False)

  # read sigs
  sg = pd.read_csv(input_file, low_memory=False)

  print(f"[InputSigsShape] {sg.shape}")

  sg.regimen_cui = sg.regimen_cui.astype(str)

  # merge by regimen_name as id does not match
  sg = sg.merge(
      df[["c1_code", "c1_name", "c2_code", "c2_name"]],
      left_on="regimen_cui",
      right_on="c1_code",
      how="left",
  )
  
  #### LOG + CLEAN
  logger = get_logger(f"{log_dir}/query.log")

  # drop vocab regimen_cui
  log_missing(sg, 'c1_code', logger)
  sg.drop(columns=['c1_code'], inplace=True) 

  # log difference sigs.regimen and vocab.regimen (c1_name) then drop the col
  sg_clean = sg[sg.c1_name.notna()] 
  log_column_differences(sg_clean, "regimen", "c1_name", logger)
  
  sg.drop(columns=['c1_name']) 

  # rename condition columns for downstream
  sg.rename(columns={
    "c2_name" : "condition",
    "c2_code" : "condition_cui"
  }, inplace=True)

  print(f"[ConditionAdded] {sg.shape}")
  sg.to_csv(output_file, index=False)


def run_query_valid_drugs(engine, output_file, input_file, log_dir):
  if os.path.exists(output_file):
      return 1
  df = pd.read_sql(drugs_sql, engine)
  df.to_csv(output_file, index=False)
  
  #### LOG + CLEAN
  logger = get_logger(f"{log_dir}/query.log")
  log_shape(df, "valid drugs query", logger)

def main(
    credentials = {
      "username":"username",
      "password":"password",
      "host":"host",
      # "port":"port",
      "db":"db"
    },
    input_file = "INPUT_FILES_HEMONC/sigs.csv",
    vocab_file_condition = "INPUT_FILES_HEMONC/sigs_conditions.csv", 
    vocab_file_drugs = "INPUT_FILES_HEMONC/sigs_drugs.csv", 
    output_file_conditions ="INPUT_FILES_HEMONC/sigs_w_conditions.csv",
    log_dir = None,
   ):
  
  # engine = create_engine(f"postgresql://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['db']}")
  engine = create_engine(f"postgresql://{credentials['username']}:{credentials['password']}@{credentials['host']}/{credentials['db']}")

  run_query_conditions(engine, vocab_file_condition, input_file, log_dir, output_file_conditions)

  run_query_valid_drugs(engine, vocab_file_drugs, input_file, log_dir)