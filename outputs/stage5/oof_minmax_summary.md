# OOF Threshold and MinMax Mainline Review

Thresholds are selected from 5-fold out-of-fold predictions on the main dataset, then transferred to the full external 2023 dataset.

## External performance
- standard: thr=0.85, precision=0.5894, recall=0.8466, f1=0.6949, pr_auc=0.7469, TP=240698, FP=167705, FN=43617, TN=116610
- minmax: thr=0.41, precision=0.9996, recall=0.5695, f1=0.7256, pr_auc=0.9640, TP=161926, FP=59, FN=122389, TN=284256

## Best external F1
- minmax scaling with OOF-selected threshold 0.41.
- External F1 = 0.7256.

## Interpretation
MinMax scaling is promoted from an exploratory Stage4 finding to a strict OOF-threshold review. It improves external precision and F1, but external recall falls below the high-recall operating target. It is therefore a strong low-false-alarm candidate, while the standard-scaling LR remains the safer high-recall baseline.