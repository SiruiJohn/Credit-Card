# Phase2 Advanced Analysis

- Selected threshold (from main): 0.85
- External metrics at selected threshold: precision=0.5912, recall=0.8455, f1=0.6959

## Best Ablation Variants (Top 3 by external F1)
- dedup_low_pos_weight: F1=0.7218, Recall=0.8110, Precision=0.6502, thr=0.70
- no_dedup_auto_weight: F1=0.6977, Recall=0.8343, Precision=0.5996, thr=0.85
- baseline_dedup_auto_weight: F1=0.6974, Recall=0.8382, Precision=0.5971, thr=0.85