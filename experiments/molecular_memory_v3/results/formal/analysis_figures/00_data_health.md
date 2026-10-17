# Data health — recovered V3 ablations

Source SHA256: `2c872b448858adfb47f4d9e025e53b1ffff683f7c56d0e753054368a416ba2f8`.

Shape: 36 planned rows × 15 columns; in-memory footprint 10,716 bytes.
32 accepted runs, four incomplete runs. Accepted rows contain no missing or nonfinite displayed metrics. No duplicate rows or duplicate scenario/arm/seed keys.
Missing accepted seeds: known-target M3 seeds 42/2026, known-target A_L seed 42, and new-target M0 seed 42. Their metrics remain missing; no partial or imputed predictions enter figures.

## Column inventory

| column | dtype | missing | unique |
| --- | --- | --- | --- |
| scenario | str | 0 | 2 |
| arm | str | 0 | 6 |
| seed | int64 | 0 | 3 |
| accepted | bool | 0 | 2 |
| queue_status | str | 0 | 2 |
| n_test | int64 | 0 | 2 |
| lm_mae | float64 | 4 | 32 |
| lm_rmse | float64 | 4 | 32 |
| lm_spearman | float64 | 4 | 32 |
| lm_target_macro_mae | float64 | 4 | 32 |
| lm_valid_rate | float64 | 4 | 1 |
| numeric_mae | float64 | 4 | 32 |
| numeric_rmse | float64 | 4 | 32 |
| numeric_spearman | float64 | 4 | 32 |
| numeric_target_macro_mae | float64 | 4 | 32 |

## Descriptive statistics (accepted runs only)

| metric | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lm_mae | 32.0 | 1.648507 | 0.128868 | 1.406458 | 1.56712 | 1.628561 | 1.760303 | 1.955226 |
| lm_rmse | 32.0 | 2.064704 | 0.151781 | 1.839008 | 1.941474 | 2.048161 | 2.181843 | 2.455238 |
| lm_spearman | 32.0 | 0.198045 | 0.148547 | -0.058113 | 0.116583 | 0.185911 | 0.286502 | 0.506451 |
| lm_target_macro_mae | 32.0 | 1.677692 | 0.172482 | 1.380802 | 1.547927 | 1.682557 | 1.818942 | 1.981252 |
| lm_valid_rate | 32.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| numeric_mae | 32.0 | 1.83063 | 0.440909 | 1.317854 | 1.529607 | 1.70605 | 1.937746 | 3.157966 |
| numeric_rmse | 32.0 | 2.246914 | 0.453361 | 1.634568 | 1.93137 | 2.124796 | 2.39775 | 3.529355 |
| numeric_spearman | 32.0 | 0.248011 | 0.137969 | 0.027738 | 0.128228 | 0.239686 | 0.327821 | 0.522005 |
| numeric_target_macro_mae | 32.0 | 1.824849 | 0.376974 | 1.245156 | 1.601201 | 1.759825 | 1.923148 | 2.876707 |

## Categories and coverage

| scenario | arm | channel | k | expected_k |
| --- | --- | --- | --- | --- |
| new_peptide_known_target | M0 | lm | 3 | 3 |
| new_peptide_known_target | M0 | numeric | 3 | 3 |
| new_peptide_known_target | M1 | lm | 3 | 3 |
| new_peptide_known_target | M1 | numeric | 3 | 3 |
| new_peptide_known_target | M2 | lm | 3 | 3 |
| new_peptide_known_target | M2 | numeric | 3 | 3 |
| new_peptide_known_target | M3 | lm | 1 | 3 |
| new_peptide_known_target | M3 | numeric | 1 | 3 |
| new_peptide_known_target | A_L | lm | 2 | 3 |
| new_peptide_known_target | A_L | numeric | 2 | 3 |
| new_peptide_known_target | A_Q | lm | 3 | 3 |
| new_peptide_known_target | A_Q | numeric | 3 | 3 |
| new_peptide_new_target | M0 | lm | 2 | 3 |
| new_peptide_new_target | M0 | numeric | 2 | 3 |
| new_peptide_new_target | M1 | lm | 3 | 3 |
| new_peptide_new_target | M1 | numeric | 3 | 3 |
| new_peptide_new_target | M2 | lm | 3 | 3 |
| new_peptide_new_target | M2 | numeric | 3 | 3 |
| new_peptide_new_target | M3 | lm | 3 | 3 |
| new_peptide_new_target | M3 | numeric | 3 | 3 |
| new_peptide_new_target | A_L | lm | 3 | 3 |
| new_peptide_new_target | A_L | numeric | 3 | 3 |
| new_peptide_new_target | A_Q | lm | 3 | 3 |
| new_peptide_new_target | A_Q | numeric | 3 | 3 |

## Normality, outliers and predictors

No normality or inferential test is justified for only three planned seeds per configuration. These are repeated model metrics, not independent predictor columns; VIF is not applicable and no regression is fitted. Correlations are descriptive and saved separately. IQR flags pooled over configurations are diagnostic only; all accepted values are retained.

| metric | IQR_flags | removed |
| --- | --- | --- |
| lm_mae | 0 | 0 |
| lm_rmse | 0 | 0 |
| lm_spearman | 0 | 0 |
| lm_target_macro_mae | 0 | 0 |
| lm_valid_rate | 0 | 0 |
| numeric_mae | 3 | 0 |
| numeric_rmse | 3 | 0 |
| numeric_spearman | 0 | 0 |
| numeric_target_macro_mae | 3 | 0 |

## Interpretation limits

Only complete 3/3 groups receive a mean and sample SD (ddof=1). Paired differences use only the same accepted seed on both sides and likewise require 3/3 for summaries. No p-values, significance markers, missing-to-zero conversion, or seed selection are used. LM generation and the numeric head are separate channels. The four missing runs are execution failures, not evidence of either model success or failure. No C1 sensitivity value is treated as prediction improvement or biological causation.