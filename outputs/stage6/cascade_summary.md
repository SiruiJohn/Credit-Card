# Stage 6 — Two-Stage Cascade Architecture & Advanced Metrics

## Architecture
- Stage 1: MinMax LR, threshold = 0.41 (high-confidence auto-block)
- Stage 2: Standard LR, threshold = 0.85 (recall safety net)

## External Dataset Performance Comparison

| Metric | Single Standard LR | Two-Stage Cascade |
|--------|-------------------|-------------------|
| Precision | 0.5894 | 0.6025 |
| Recall | 0.8466 | 0.8942 |
| F1 | 0.6949 | 0.7200 |
| TP | 240,698 | 254,245 |
| FP | 167,705 | 167,719 |
| FN | 43,617 | 30,070 |

## F-β Scores (External)

| β | Single LR | Cascade |
|----|-----------|---------|
| 0.5 | 0.6275 | 0.6446 |
| 1.0 | 0.6949 | 0.7200 |
| 2.0 | 0.7786 | 0.8153 ← recommended |

## Amount-Weighted Metrics (External)

| Metric | Single LR | Cascade |
|--------|-----------|---------|
| W-Precision | 0.5383 | 0.5400 |
| W-Recall | 0.9650 | 0.9715 |
| W-F1 | 0.6911 | 0.6941 |

## Calibration & Stability
- ECE (main): 0.1725
- ECE (external): 0.3261
- PSI (main → external score drift): 5.9197

## Lift Curve Summary (External, Top Buckets)
- Bucket 1: positive_rate=0.9948, lift=1.99x
- Bucket 2: positive_rate=0.6403, lift=1.28x
- Bucket 3: positive_rate=0.5037, lift=1.01x
- Bucket 4: positive_rate=0.5020, lift=1.00x
- Bucket 5: positive_rate=0.5041, lift=1.01x

## Cumulative Gain (Alert Budget)
- Review 1% (5,686 alerts): captures 2.0% of fraud
- Review 2% (11,372 alerts): captures 4.0% of fraud
- Review 5% (28,431 alerts): captures 10.0% of fraud
- Review 10% (56,863 alerts): captures 19.9% of fraud
- Review 20% (113,726 alerts): captures 32.7% of fraud
- Review 50% (284,315 alerts): captures 62.9% of fraud
- Review 100% (568,630 alerts): captures 100.0% of fraud

## Interpretation
- Cascade **preserves 89.4% recall** vs single LR's 84.7%, with near-identical FP count.
- Stage 1 alone (MinMax LR, auto-block): **161,926 fraud** caught with only **59 false alarms** — near-zero human cost.
- Stage 2 (Standard LR, manual review): additional **92,319 fraud** at cost of **167,660 false alarms**.
- F-β=2 (fraud detection preference): Cascade **0.8153** vs Single LR **0.7786**.
- PSI=5.9 indicates significant score distribution drift between main and external datasets.
- ECE (external)=0.3261 indicates model probabilities are poorly calibrated.