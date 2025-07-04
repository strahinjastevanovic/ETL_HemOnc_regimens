import pandas as pd
from sqlalchemy import create_engine
import logging 
import os

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

def main(
    credentials = {
      "username":"username",
      "password":"password",
      "host":"host",
      "port":"port",
      "db":"db"
    },
    inupt_file = "INPUT_FILES_HEMONC/sigs.csv",
    vocab_file = "INPUT_FILES_HEMONC/sigs_conditions.csv", 
    output_file="INPUT_FILES_HEMONC/sigs_w_conditions.csv",
    log_dir = None,
   ):
  
  if os.path.exists(output_file):
      return 1
  
  engine = create_engine(f"postgresql://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['db']}")

  query = """
  SELECT DISTINCT ON (c1.concept_id, c2.concept_id)
    c1.concept_id           as c1_id, -- regimen_cui in OMOP
    c1.concept_code         as c1_code, -- regimen_cui in sigs
    c1.concept_name         as c1_name, -- regimen in sigs
    c1.domain_id            as c1_domain, 
    c1.concept_class_id     as c1_class,

    c2.concept_id           as c2_id, -- condition_cui in OMOP 
    c2.concept_code         as c2_code, -- condition_cui in sigs
    c2.concept_name         as c2_name, -- condition in sigs 
    c2.domain_id            as c2_domain,
    c2.concept_class_id     as c2_class

  from devv5.concept c1
  join devv5.concept_relationship r 
    on r.concept_id_1 = c1.concept_id and r.invalid_reason is null
  join devv5.concept c2 
    on c2.concept_id = r.concept_id_2

  where c1.vocabulary_id = 'HemOnc'
    and (
      (c2.concept_class_id = 'Condition' or c2.domain_id = 'Condition')
      and
      (c1.domain_id = 'Regimen' or c1.concept_class_id = 'Regimen')
    )
  -- )
  """
  df = pd.read_sql(query, engine)
  df.to_csv(vocab_file, index=False)

  # read sigs
  sg = pd.read_csv(inupt_file, low_memory=False)

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