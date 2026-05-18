# Phase2 Baseline Summary

- quick_mode: False
- common_features: 29

## logistic_regression
- CV Recall(mean): 0.8605
- CV F1(mean): 0.4428
- CV PR-AUC(mean): 0.7116
- External Recall: 0.8466
- External F1: 0.6949
- External PR-AUC: 0.7469
- Note: Full external set evaluation.

## gaussian_nb
- CV Recall(mean): 0.8435
- CV F1(mean): 0.1092
- CV PR-AUC(mean): 0.4090
- External Recall: 0.7974
- External F1: 0.6411
- External PR-AUC: 0.4939
- Note: Full external set evaluation.

## knn
- CV Recall(mean): 0.8068
- CV F1(mean): 0.5811
- CV PR-AUC(mean): 0.6695
- External Recall: 0.0274
- External F1: 0.0518
- External PR-AUC: 0.5051
- Note: Evaluated on stratified external subset (n=10000) for runtime feasibility.
