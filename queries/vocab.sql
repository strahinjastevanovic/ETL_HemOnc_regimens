-- SET search_path TO 'devv5';

-- SELECT * FROM devv5.concept;
-- SELECT * FROM concept WHERE schemaname = 'devv5';

SELECT * FROM pg_indexes WHERE schemaname = 'devv5' AND tablename = 'concept';

SELECT
    c1.concept_id           as c1_id,
    c1.concept_name         as c1_name,
    c1.vocabulary_id        as c1_vocab,
    c1.domain_id            as c1_domain,
    c1.concept_class_id     as c1_class,
    c1.invalid_reason       as c1_ir,

    r.relationship_id       as rel,
    r.invalid_reason        as r_ir,

    c2.concept_id           as c2_id,
    c2.concept_name         as c2_name,
    c2.vocabulary_id        as c2_vocab,
    c2.domain_id            as c2_domain,
    c2.concept_class_id     as c2_class,
    c2.invalid_reason       as c2_ir

from devv5.concept c1
join devv5.concept_relationship r on r.concept_id_1 = c1.concept_id and r.invalid_reason is null
join devv5.concept c2 on c2.concept_id = r.concept_id_2
where c1.vocabulary_id = 'HemOnc';

select distinct 
    c1.concept_class_id, 
    relationship_id, 
    c2.concept_class_id, 
    count(*)
from devv5.concept c1
join devv5.concept_relationship r on r.concept_id_1 = c1.concept_id and r.invalid_reason is null
join devv5.concept c2 on c2.concept_id = r.concept_id_2
where c1.vocabulary_id = 'HemOnc'
group by c1.concept_class_id, relationship_id, c2.concept_class_id;

-- Disabled block
-- SELECT * FROM some_heavy_table;
SELECT COUNT(*) FROM devv5.concept;

-- Provided query - requested 4 condition 2 regimen
SELECT
    c1.concept_id           as c1_id,
    c1.concept_name         as c1_name,
    c1.vocabulary_id        as c1_vocab,
    c1.domain_id            as c1_domain,
    c1.concept_class_id     as c1_class,
    c1.invalid_reason       as c1_ir,

    r.relationship_id       as rel,
    r.invalid_reason        as r_ir,

    c2.concept_id           as c2_id,
    c2.concept_name         as c2_name,
    c2.vocabulary_id        as c2_vocab,
    c2.domain_id            as c2_domain,
    c2.concept_class_id     as c2_class,
    c2.invalid_reason       as c2_ir

from devv5.concept c1
join devv5.concept_relationship r on r.concept_id_1 = c1.concept_id and r.invalid_reason is null
join devv5.concept c2 on c2.concept_id = r.concept_id_2
where c1.vocabulary_id = 'HemOnc' AND (
   (c2.concept_class_id = 'Condition' OR
    c2.domain_id = 'Condition' ) AND (
    c1.domain_id = 'Regimen' OR c1.concept_class_id = 'Regimen' )
)

-- Query regimen 2 condition relationships
SELECT COUNT(*) AS total_relationships
from devv5.concept c1
join devv5.concept_relationship r 
  on r.concept_id_1 = c1.concept_id AND r.invalid_reason IS NULL
join devv5.concept c2 
  on c2.concept_id = r.concept_id_2
where c1.vocabulary_id = 'HemOnc'
  and (
    (c2.concept_class_id = 'Condition' or
    c2.domain_id = 'Condition' ) and (
    c1.domain_id = 'Regimen' or c1.concept_class_id = 'Regimen' )
  );


-- UPDATE - ANNOT.
SELECT COUNT(*) AS total_unique_pairs
FROM ( 
SELECT DISTINCT ON (c1.concept_id, c2.concept_id)
  c1.concept_id           as c1_id, -- regimen_cui in sigs
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
)








-- other
-- check if all pairs
SELECT NOT EXISTS (
  SELECT 1
  FROM devv5.concept c1
  JOIN devv5.concept_relationship r 
    ON r.concept_id_1 = c1.concept_id AND r.invalid_reason IS NULL
  JOIN devv5.concept c2 
    ON c2.concept_id = r.concept_id_2
  WHERE c1.vocabulary_id = 'HemOnc'
    AND (
      c1.concept_class_id IS DISTINCT FROM c1.domain_id OR
      c2.domain_id IS DISTINCT FROM c2.concept_class_id
    )
) AS all_match;

-- print distinct cases
SELECT 
  c1.concept_id AS c1_id,
  c1.concept_name AS c1_name,
  c1.concept_class_id AS c1_class,
  c1.domain_id AS c1_domain,
  
  c2.concept_id AS c2_id,
  c2.concept_name AS c2_name,
  c2.concept_class_id AS c2_class,
  c2.domain_id AS c2_domain

FROM devv5.concept c1
JOIN devv5.concept_relationship r 
  ON r.concept_id_1 = c1.concept_id AND r.invalid_reason IS NULL
JOIN devv5.concept c2 
  ON c2.concept_id = r.concept_id_2
WHERE c1.vocabulary_id = 'HemOnc'
  AND (
    c1.concept_class_id IS DISTINCT FROM c1.domain_id OR
    c2.domain_id IS DISTINCT FROM c2.concept_class_id
  )
LIMIT 5;


-- query to get all concept condition pairs
SELECT DISTINCT ON (c1.concept_id, c2.concept_id)
  c1.concept_id           as c1_id, -- regimen_cui in sigs
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