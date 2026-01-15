# Doc: Description: "athena vocab queries" - pkg requirments - if any
schema = "prodv5"

query_conditions = f"""
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

from {schema}.concept c1
join {schema}.concept_relationship r 
  on r.concept_id_1 = c1.concept_id and r.invalid_reason is null
join {schema}.concept c2 
  on c2.concept_id = r.concept_id_2

where c1.vocabulary_id = 'HemOnc'
  and (
    (c2.concept_class_id = 'Condition' or c2.domain_id = 'Condition')
    and
    (c1.domain_id = 'Regimen' or c1.concept_class_id = 'Regimen')
  )
-- )
"""

query_valid_drugs = f"""
SELECT DISTINCT
    c.concept_id,
    c.concept_name,
    c.concept_class_id,
    c.vocabulary_id,
    c.domain_id,
    c.concept_code,
    c.invalid_reason,
    cr.concept_id_2 AS valid_concept_id
FROM
    {schema}.concept c
LEFT JOIN
    {schema}.concept_relationship cr
    ON c.concept_id = cr.concept_id_1
    AND cr.relationship_id = 'Maps to'
WHERE
    c.vocabulary_id = 'HemOnc'
    AND c.domain_id = 'Drug'
    AND cr.concept_id_2 IS NOT NULL;
"""
