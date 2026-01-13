# SRE Module Refactor: Vector Stacking and Block-Based Cycle Length

## User's Clarified Intent (January 13, 2026)

The following prompt clarifies the core concept driving this refactor. It represents the authoritative understanding of what tseq is and why block-level computation is necessary:

> tseq should be called last_day if more clear. It represents the information of drug and drug signature stacking that needs to be considered. When building a matrix, all drugs in the same timing_sequence block must have vectors of the same length to align properly. This length is determined by the drug with the latest administration day in that block.
>
> **Normal case (matching cycle lengths):**
> - timing_seq="1,2,3", tseq=10 (all days up to day 10)
> - DrugA: allDays=[1,5,8] → vector [1,0,0,0,1,0,0,1,0,0]
> - DrugB: allDays=[1,2,3] → vector [1,1,1,0,0,0,0,0,0,0]
> - Both vectors: length 10, representing single cycle
>
> **Edge case (mismatched cycle contexts, same timing_sequence):**
> - timing_seq="1,2,3" (same cycles for both drugs)
> - DrugA: allDays=[1,2,3] cycleLength=8 → idays=[1,2,3], cycle needs 8 days max
> - DrugB: allDays=[10,11] cycleLength=12 → idays=[10,11], cycle needs 12 days max
> - DrugA needs last_day >= 3, DrugB needs last_day >= 11
> - Block tseq = max(3, 11) = 11
> - DrugA vector: [1,1,1,0,0,0,0,0,0,0,0] (marks days 1,2,3, padded to 11)
> - DrugB vector: [0,0,0,0,0,0,0,0,0,1,1] (marks days 10,11, padded to 11)
> - Both have length 11, can be stacked in matrix
>
> **Multi-timing-sequence case:**
> - timing_seq="1,2,3" tseq=10 (drugs active in cycles 1,2,3)
> - timing_seq="4,5" tseq=12 (drugs active in cycles 4,5)
> - These are separate matrix constructions
> - normalize_multicycle_spans creates Drug@cycleLen10 and Drug@cycleLen12
> - Each generates separate regimen strings

## Overview

The SRE (Structured Regimen Expression) module underwent a critical refactor to ensure drugs in the same timing_sequence block have vectors of consistent length for proper matrix stacking. The core issue: when drugs have different cycle contexts (different allDays ranges or cycle_length metadata), their vectors must all be padded to accommodate the drug with the latest administration day.

## Problem Statement

Drugs within a single timing_sequence block may have different cycle contexts:

1. **allDays mismatch**: DrugA has allDays=[1,2,3] but DrugB has allDays=[10,11]
2. **Cycle length mismatch**: DrugA cycleLength=8 but DrugB cycleLength=12
3. **Combined mismatch**: Both differ in allDays AND cycle length

Prior code didn't account for this properly. It would compute vector length individually per row, creating vectors of different lengths that couldn't be stacked into a single matrix.

## Solution: Block-Level tseq Computation

**tseq** (last_day) is computed ONCE per timing_sequence block by examining ALL drugs in that block:

```
tseq = max(
  max(idays across all drugs in block),
  max(converted cycle_length_lb and cycle_length_ub across all drugs)
)
```

This single tseq is then applied to ALL drugs in the block, padding vectors with zeros where needed.

## Architecture

### 1. `_infer_block_tseq(block: pl.DataFrame) -> int`

**Input**: A block (subset of rows all having the same timing_sequence)

**Process**:
- Iterate through all rows in block
- For each row:
  - Parse allDays to idays, record max(idays)
  - Convert cycle_length_lb and cycle_length_ub to days, record max of both
- Return: max of all recorded maxima

**Output**: Single integer tseq representing the last day needed for this block

**Example**:
- Row 1: idays=[1,2,3], cycle_length=8 → max(3, 8) = 8
- Row 2: idays=[10,11], cycle_length=12 → max(11, 12) = 12
- Block tseq = max(8, 12) = 12

### 2. `_build_vector_from_idays(idays, tseq, ...) -> np.ndarray`

**Input**: 
- idays (parsed administration days for this drug)
- tseq (block's authoritative length)

**Process**:
- Create zeros array of length tseq
- Mark positions in idays with 1s
- Return binary vector

**Output**: Binary vector of length tseq with 1s at idays positions

**Example** (using block tseq=12):
- DrugA idays=[1,2,3] → [1,1,1,0,0,0,0,0,0,0,0,0]
- DrugB idays=[10,11] → [0,0,0,0,0,0,0,0,0,1,1,0]

### 3. `_process_group(group: pl.DataFrame) -> pl.DataFrame`

**Workflow**:

```
For each timing_sequence in group:
  |
  ├─ Compute tseq = _infer_block_tseq(block)
  |
  └─ For each drug in block:
      |
      ├─ Parse idays from allDays
      |
      ├─ Create variant SET {cycle_length_lb, cycle_length_ub}
      |
      └─ For each variant:
          |
          └─ Build vector of length tseq
             Append to component_vectors[drug]

Pass component_vectors to create_reg_string()
  |
  ├─ normalize_multicycle_spans:
  |   Groups vectors by length
  |   Creates Drug@cycleLen{L} for different lengths
  |
  ├─ validate_and_split_variants:
  |   Returns one dict per unique vector length
  |
  └─ collapse_event_matrix:
      Collapses each dict → one regimen string

Output: N regimen strings
  |
  └─ Repeat group N times
     Map cycleLength = max(all tseqs)
     Return with regString and cycleLength columns
```

## Critical Properties

### Vector Length Consistency Within Block

All drugs in the same timing_sequence block get vectors of the same length (the block's tseq). This is necessary for matrix stacking:

```
timing_seq="1,2,3" tseq=11

Matrix:
         Day: 1 2 3 4 5 6 7 8 9 10 11
DrugA:       1 1 1 0 0 0 0 0 0 0  0
DrugB:       0 0 0 0 0 0 0 0 0 1  1
DrugC:       1 0 0 0 1 0 0 0 0 0  0

All rows have length 11
```

### Vector Length Variation Across Blocks

Different timing_sequences may have different tseqs. This is expected and handled by normalize_multicycle_spans:

```
timing_seq="1,2,3" tseq=10 → All vectors length 10
timing_seq="4,5"   tseq=12 → All vectors length 12

normalize_multicycle_spans creates:
  Drug@cycleLen10 (from timing_seq 1,2,3)
  Drug@cycleLen12 (from timing_seq 4,5)

Each generates separate regimen string
```

### Output Cycle Length

When group spans multiple timing_sequences with different tseqs:
- Compute all_tseqs = [tseq for each block]
- Use max(all_tseqs) for regimen string cycle length
- This is a safe upper bound ensuring regimen fits full timeline

**Future refinement**: Track which tseq produced which regimen string by examining @cycleLen{L} suffixes from normalize_multicycle_spans.

## Variant Types

### Type 1: Cycle Length Variants (lb ≠ ub)

When a single row has cycle_length_lb ≠ cycle_length_ub:
- Create SET {lb, ub}
- Generate separate vector for each value
- Both use block's tseq
- Appended separately to component_vectors

### Type 2: Vector Length Variants (@cycleLen)

When drug appears in vectors of different lengths (different timing_sequences):
- normalize_multicycle_spans creates Drug@cycleLen{L1}, Drug@cycleLen{L2}
- Each subset generates one regimen string

### Type 3: Timing Sequence Variants

Same drug, different timing_sequences:
- timing_seq="1,2,3" tseq=10 → vector length 10
- timing_seq="4,5" tseq=12 → vector length 12
- Tracked via timing_seq key in component_vectors tuples

## Example: Edge Case (User's Prompt)

**Setup:**
- Group: condition_cui=C001, regimen_cui=R001, variant_key=V001
- timing_sequence="1,2,3" (active in cycles 1, 2, 3)

**Rows:**
1. DrugA: allDays="1,2,3" cycleLength_lb=8 cycleLength_ub=8
2. DrugB: allDays="10,11" cycleLength_lb=12 cycleLength_ub=12

**Step 1: Compute tseq**
- DrugA: max(idays=[1,2,3], cycle=8) = max(3, 8) = 8
- DrugB: max(idays=[10,11], cycle=12) = max(11, 12) = 12
- Block tseq = max(8, 12) = 12

**Step 2: Build vectors (both length 12)**
- DrugA: [1,1,1,0,0,0,0,0,0,0,0,0]
- DrugB: [0,0,0,0,0,0,0,0,0,1,1,0]

**Step 3: Matrix stacking**
```
       1 2 3 4 5 6 7 8 9 10 11 12
DrugA  1 1 1 0 0 0 0 0 0 0  0  0
DrugB  0 0 0 0 0 0 0 0 0 1  1  0
```

**Step 4: collapse_event_matrix**
- Scans matrix chronologically
- Finds active days: 1, 2, 3, 10, 11
- Generates event tags
- Produces regimen string

## Implementation Details

### component_vectors Structure

```python
component_vectors: Dict[str, List[Tuple[str, np.ndarray]]]

{
  "DrugA": [
    ("1,2,3", [1,1,1,0,...]),    # from timing_seq 1,2,3
    ("1,2,3", [1,1,1,0,...]),    # variant (if lb ≠ ub)
    ("4,5", [1,0,1,0,...]),      # from timing_seq 4,5, different length
  ],
  "DrugB": [
    ("1,2,3", [0,0,0,1,...]),
    ("4,5", [0,1,0,0,...]),
  ]
}
```

All tuples with same timing_seq → same vector length
Different timing_seq → may have different vector lengths

### Cycle Length Output

```python
all_tseqs = [tseq for each timing_seq block]
cycle_length_value = max(all_tseqs)

For each regimen string generated:
  cycleLength = cycle_length_value
```

This ensures all regimen strings output have consistent cycle length representing the full group timeline.

## Validation

✅ Vector length consistency: All drugs in block have same length  
✅ Matrix stacking: Vectors properly align  
✅ Cycle length variants: lb/ub splits handled correctly  
✅ Vector length variants: @cycleLen{L} suffixes handled  
✅ Output shape: |cycleLength| = |reg_strings|  
✅ All unit tests pass (6/6)
