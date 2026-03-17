# Regression Testing Methodology for Regimens produced by Assembler

## 1. Comparison Strategy: Shallow vs. Deep
- **Shallow Comparison**: Row counts (`size`) and unique value counts (`cardinality`) tracked across versions. This provides a high-level "sanity check" for ingestion errors.
- **Deep Comparison**: Probabilistic drift detection using empirical distributions. Instead of comparing raw data-points, we compare the statistical fingerprint of the data.
- **Support Tracking**: Monitoring unique value sets via Jaccard Similarity. This detects "lost" keys (data dropouts) or "gained" keys (new categories or corrupted strings).

## 2. Drift Detection

### 2.1. The Metrics
- **Stability Score**: A classification metric used to flag regressions.
  - **Stable (1)**: Indicated by a perfect match or a purely additive change (gained keys without any lost keys).
  - **Drift (0)**: Triggered when the support matches but the frequency distribution ($JS$ Divergence) has shifted.
  - **Regression (-1)**: Explicitly triggered whenever "lost keys" are detected (data dropout from reference).

- **Jaccard Similarity ($J$):** Measures the membership parity of unique strings (Support Table Overlap). 
  $$ J(P,Q) = \frac{|P \cap Q|}{|P \cup Q|} $$
- **Jensen-Shannon Divergence ($JS$):** A symmetric and bounded ($0 \le JS \le 1$) version of Kullback-Leibler (KL) divergence. It measures how differently the new dataset models observations compared to the reference.
  $$ M = \frac{1}{2}(P+Q) $$
  $$ JS(P,Q) = \frac{1}{2}KL(P||M) + \frac{1}{2}KL(Q||M) $$

### 2.2. Technical Glossary & Information Theory
- **Empirical Distribution**: Frequency-based probability measure $P(x) = \frac{\text{count}(x)}{N}$.
- **Surprise (Relative Information)**: The ratio $P(x)/Q(x)$ measures mismatch. If $Q$ underestimates a value that is frequent in $P$, the "surprise" increases, driving up total divergence.
- **$KL$ units**: The log-ratio converts probability changes into additive information units. 
- **Schema Drift**: Detection of structural changes, such as unexpected column name changes or missing fields in `report` tables.


## 3. Versioning & Lineage Tracking
- **Baseline (Ref)**: The "Current Standard" output (e.g., `v1.3 d6e82f3` produced by `assembler-v1.1.3 482811f`), representing the last validated state of the truth.
- **Trial (New)**: The experimental output from the updated assembler run. Regression test triggers on PR. Human in loop for reviews.

## 4. Intentional Trade-offs
- **No Schema Version Tracking**: If column types change (e.g., int64 → float64), the current implementation won't detect it since everything is cast to `str`.
- **No Longitudinal Statistics**: We store associative statistics (histograms and cardinalities) for every run. This allows us to track drift over time across multiple iterations.

