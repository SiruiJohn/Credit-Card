# Credit Card Fraud Detection Project

AI3013 course project for credit card fraud detection, from-scratch machine learning models, and cross-dataset generalization analysis.

## Current Status

- Stage 1 data preparation has been repaired and regenerated.
- Stage 2 baseline has been rerun in full 5-fold mode.
- Stage 2 threshold scans, confusion matrix, ablation, LR final analysis, and kNN tuning have been refreshed.
- Stage 3 cost-sensitive thresholding, temporal robustness, and error-case analysis have been refreshed.
- Notebook-inspired Stage 4 extensions have been added: external EDA figures, scaling ablation, and mature library reference benchmarks.
- Stage 5 OOF-threshold review has been added to verify MinMax scaling under a stricter threshold-selection protocol.
- Formal LaTeX figures in `latex_bundle/figures/` have been synchronized with the refreshed outputs.
- A 15-slide defense deck has been added at `docs/Credit_Card_Fraud_Detection_Defense.pptx`.

See `docs/Project_Status_and_Fix_Log.md` for the detailed fix log.

## Data

- Main dataset: `Credit Card Fraud Detection/creditcard.csv`
  - Raw rows: 284,807
  - Class distribution: 0:284,315 / 1:492
- External 2023 dataset: `Credit Card Fraud Detection_2023/creditcard_2023.csv`
  - Raw rows: 568,630
  - Class distribution: 0:284,315 / 1:284,315

The raw CSV files and regenerated prepared CSV files are intentionally not committed because they exceed normal GitHub repository size limits. Place the two source CSV files in the paths above before running the reproduction commands.

Generated prepared data:

- `outputs/stage1/main_prepared.csv`
- `outputs/stage1/ext2023_prepared.csv`
- `outputs/stage1/prepare_metadata.json`

## Reproduce

Use the Codex bundled Python runtime or another Python 3.12 environment with the packages in `requirements.txt`.

In this workspace, plotting uses the local dependency folder `.codex_pydeps`:

```powershell
$py = 'C:\Users\siruiJohn\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = '.\.codex_pydeps'

& $py scripts\prepare_stage1_data.py
& $py scripts\train_phase2.py
& $py scripts\plot_phase2_results.py
& $py scripts\phase2_advanced_analysis.py
& $py scripts\lr_final_analysis.py
& $py scripts\tune_knn.py
& $py scripts\tune_knn_advanced.py
& $py scripts\cost_sensitive_threshold.py
& $py scripts\temporal_robustness.py
& $py scripts\error_case_analysis.py
& $py scripts\external_eda_notebook_inspired.py
& $py scripts\scaling_ablation.py
& $py scripts\library_benchmark.py
& $py scripts\minmax_oof_threshold_analysis.py
```

## Main Results

Current refreshed baseline:

- Logistic Regression: CV Recall 0.8605, CV F1 0.4428, CV PR-AUC 0.7116, external F1 0.6949.
- Gaussian Naive Bayes: CV Recall 0.8435, CV F1 0.1092, CV PR-AUC 0.4090, external F1 0.6411.
- kNN: CV Recall 0.8068, CV F1 0.5811, CV PR-AUC 0.6695, external F1 0.0518 on stratified external subset.

The primary model remains Logistic Regression because it has the strongest cross-dataset precision-recall balance among the from-scratch models.

Notebook-inspired extensions are documented in `docs/Notebook_Extensions.md`, with outputs saved under `outputs/stage4/`.

Stage 5 OOF-threshold review is saved under `outputs/stage5/`. It shows that MinMax scaling improves external F1 and sharply reduces false positives, but lowers external recall, so it is best treated as a low-false-alarm candidate rather than a replacement for the high-recall standard-scaling baseline.

## Deliverables

- Integrated report: `docs/Final_Submission_Integrated.md`
- English integrated report: `docs/Final_Submission_Integrated_EN.md`
- LaTeX report bundle: `latex_bundle/main.tex`
- Defense deck: `docs/Credit_Card_Fraud_Detection_Defense.pptx`
