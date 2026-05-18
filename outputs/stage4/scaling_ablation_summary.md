# Scaling Ablation

Inspired by the reference notebooks' scaler comparisons.

- knn_distance + minmax: precision=0.8095, recall=0.0068, f1=0.0135, pr_auc=0.5077, thr=0.05
- knn_distance + standard: precision=0.8000, recall=0.0016, f1=0.0032, pr_auc=0.5012, thr=0.05
- knn_distance + robust: precision=1.0000, recall=0.0010, f1=0.0020, pr_auc=0.5014, thr=0.05
- logistic_regression + minmax: precision=0.9995, recall=0.7693, f1=0.8694, pr_auc=0.9634, thr=0.40
- logistic_regression + standard: precision=0.5949, recall=0.8394, f1=0.6963, pr_auc=0.7545, thr=0.85
- logistic_regression + robust: precision=0.9916, recall=0.0058, f1=0.0116, pr_auc=0.8075, thr=0.80