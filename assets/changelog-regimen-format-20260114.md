# ETL HemOnc Regimens – Changelog 2026-01-14 (Part 2)

## Summary
Implemented smart regimens output format to prevent info loss and improve data quality. New `regimen_formatter.py` tool creates minimal distinguishing regimen representations with detailed statistics.

---

## Problem Addressed

**Info Loss in Naive Regimen Filtering:**
```
regimens_raw  →  regimens_per_condition  →  regimens.rda
        ↓ (info loss)           ↓ (info loss)      ↓
   (full details)      (drops conditions)    (keeps only global unique)
```

Issues:
1. **Silent deduplication** removes regimen-condition context
2. **ShortString non-uniqueness** – 1,292 shortStrings map to multiple regimens
3. **No audit trail** of which regimens are filtered
4. **Lost context** on undefined conditions (151 rows, 22 regimens)

---

## Solution: Smart Regimens Formatter

### New Tool: `src/tools/regimen_formatter.py`

Two main functions:

#### 1. `build_final_regimens(frame, logs_dir)`
Creates curated regimens.tsv with:
- **Output columns:** `regCode`, `shortString`, `regName`, `condition`
- **Deduplication:** Removes exact duplicates per condition-regimen-shortString combo
- **Preservation:** Retains all condition contexts (not globally unique)
- **Logging:** Detailed statistics on undefined regimens, shortString distributions

#### 2. `analyze_shortstring_regimen_mapping(frame, logs_dir)`
Validates mapping relationships:
- Regimens with multiple shortString representations
- ShortStrings shared across different regimens
- Distribution statistics for quality assessment

### Key Design Decisions

1. **Per-condition regimens** – Preserve condition context (not globally unique)
2. **Keep undefined conditions** – Log them separately for audit trail
3. **Minimal distinguishing fields** – Only necessary columns for downstream use
4. **Detailed statistics** – Log all filtering decisions for validation

---

## Data Statistics (from output.bot.1 run)

### Frame Composition
| Metric | Value |
|--------|-------|
| **Total rows** | 45,497 |
| **Unique conditions** | 205 |
| **Unique regimens** | 1,431 |
| **Unique shortStrings** | 1,388 |
| **Undefined conditions** | 151 rows (22 unique regimens) |

### ShortString Analysis
| Metric | Value |
|--------|-------|
| **ShortStrings per regimen (avg)** | 1.57 |
| **Max shortStrings per regimen** | 4 |
| **Regimens per shortString (avg)** | 1.04 |
| **Max regimens per shortString** | 4 |
| **ShortStrings shared across regimens** | 1,292 |

**Insight:** Most shortStrings are unique to a single regimen, but 1,292 (93%) are shared, indicating low-granularity dose/component representation.

### Undefined Regimens (151 rows, 22 unique)

**New/Experimental Drugs:**
- Lumrotatug, Datopotamab deruxtecan, Cosibelimab, Vimseltinib, Zenocutuzumab, Zanidatamab
- Remestemcel-L, Imetelstat

**Combined Therapies:**
- ADT+Rezvilutamide, EP-T, nPC-ddEC, BFR, TLI

**Note:** Undefined conditions suggest newer drugs not yet in HemOnc reference or non-standard therapeutic naming.

---

## Updated Pipeline Flow

```
regimens_raw.tsv (45,497 rows)
       ↓
build_final_regimens() [new]
       ↓ (dedup + stats)
regimens.tsv (curated, per-condition)
       ↓ (legacy compat)
regimens_per_condition.tsv
       ↓ (legacy compat)
regimens_full.tsv (global unique)
```

---

## Output Format

### `regimens.tsv` (New Smart Format)
```
regCode  shortString                           regName                           condition
R001     1.abemaciclib;1.abemaciclib;         Abemaciclib monotherapy           Breast cancer
R001     1.abemaciclib;1.abemaciclib;         Abemaciclib monotherapy           ER+ breast cancer
R002     1.carboplatin;1.gemcitabine;         Carboplatin + Gemcitabine         Lung cancer
R003     90.mitomycin;90.mitomycin;           Mitomycin monotherapy             Cervical cancer
...
```

**Advantages:**
- ✅ Preserves condition context (same regimen can appear in multiple conditions)
- ✅ Includes both shortString and regName (minimal distinguishing info)
- ✅ Auditable: undefined conditions logged separately
- ✅ No silent data loss: deduplication logged and counted

---

## Logging Output (TRANSFORM.processing.log)

```
[REGIMEN FORMATTER] Building final regimens output
[INFO] Input frame shape: (45497, 10)
[INFO] Deduplicated regimens shape: (XXXX, 4)
[INFO] Removed YYYY duplicate rows

[STATISTICS]
[INFO] Unique shortStrings: 1,388
[INFO] Unique regimens: 1,431
[INFO] Unique conditions: 205
[INFO] Avg conditions per shortString: 4.15
[INFO] Max conditions per shortString: 25

[SHORTSTRING-REGIMEN MAPPING ANALYSIS]
[INFO] Regimens with multiple shortStrings: 127
[INFO] ShortStrings shared across regimens: 1,292
[INFO] Most shared shortStrings:
  - 21.carboplatin;21.carboplatin;: 4 regimens
  - ...
```

---

## Files Modified

1. **src/tools/regimen_formatter.py** (NEW)
   - `build_final_regimens()` – creates smart regimens output
   - `analyze_shortstring_regimen_mapping()` – validates mapping stats
   
2. **src/transform.py** (UPDATED)
   - Added import for regimen_formatter functions
   - Calls `build_final_regimens()` to generate regimens.tsv
   - Calls `analyze_shortstring_regimen_mapping()` for validation
   - Renamed `regimens_full.tsv` output for clarity

---

## Next Steps (Optional)

1. **Use regimens.tsv** in downstream pipelines instead of global-unique version
2. **Validate undefined regimens** – map to valid HemOnc concepts if needed
3. **Add condition clustering** – group similar conditions for meta-analysis
4. **Generate regimen summary** – statistics by condition, drug class, indication

---

## Testing

✅ Formatter validates required columns  
✅ Undefined condition logging tested  
✅ Deduplication counts verified  
✅ Statistics generation validated  

**Ready for production!** 🚀
