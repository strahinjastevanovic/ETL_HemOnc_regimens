# Doc: Description: "athena vocab queries" - pkg requirments - if any

# backend

# mapping:
# input: INPUT_FILES_HEMONC/sigs.csv full-join regimen_cui -> concept_code many to many

# output: sigs_conditions.csv ( optional: total number sumstats )

import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np


engine = create_engine("postgresql://username:password@host:port/postgres")


query = """
SELECT DISTINCT ON (c1.concept_id, c2.concept_id)
  c1.concept_id           as c1_id, -- regimen_cui in sigs
  c1.concept_code         as c1_code, -- regimen_cui in sigs
  c1.concept_name         as c1_name, -- regimen in sigs
  c1.domain_id            as c1_domain, 
  c1.concept_class_id     as c1_class,

  c2.concept_id           as c2_id, -- condition_cui in sigs 
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
df.to_csv("INPUT_FILES_HEMONC/sigs_conditions.csv", index=False)

# read sigs
sg = pd.read_csv("INPUT_FILES_HEMONC/sigs.csv")

sg.regimen_cui = sg.regimen_cui.astype(str)

# merge by regimen_name as id does not match
sg = sg.merge(
    df[["c1_code", "c1_name", "c2_id", "c2_name"]],
    left_on="regimen_cui",
    right_on="c1_code",
    how="left",
)
