# Lineage-aware regression testing

With features:
* deterministic and local
* Keep summary stats for fast checks
* Common columns will be subject of stat comparison goverend by the schema.
* For score <= 0, run exact diff 

## Schema definition
```
{
"sample_size_ref": int,
"sample_size_new": int,
"cardinality_ref": int,
"cardinality_new": int,
"jaccard_unique": float,
"lost_key_count": int,
"gained_key_count": int,
"js_divergence": float,
"score": int
}
```

## Metrics in use:
There are three orthogonal changes resulting in a regression score 
1. **Support change**: lost_keys / gained_keys
2. **Distribution shape change**: JS divergence
3. **Sample size change**: sample size difference
The regression testing is categorized as **coverage integrity** type. This is important for defining 0-score.

### score
custom keys categorical `[-1, 0, 1]`
```
<1>   - additional data entries + no loss, in other words JS == 0
<0>   - mixed scenario.
<-1>  - clean data loss
```
Note on mixed scenario:  
Since this is coverage integrity type of regression,W
JS is used as shape drift detector under fixed sample size.  

### Jaccard   
intersection over union in range [0,1]  

### Jensen-Shannon (JS)  
more stable version of KL,   
More interpretable as drift metric in range [0,1] given log base of 2



**Subsequent Runs:**
* Tracks drift over time (subseq. runs) and visualize trends - cardinality, distriution shift, row_count growth.

For example:    

| Metric       | Run1 | Run2 |
|--------------|------|------|
| sample_size  | 10   | 10   |
| JS           | 0.5  | 0.7  |  

shows ditribution moved further away from the reference by introducing most recent change.
