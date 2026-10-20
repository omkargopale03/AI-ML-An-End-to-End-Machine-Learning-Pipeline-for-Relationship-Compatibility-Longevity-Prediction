# Model comparison

## compatibility (classification, target `compatible`)

| model | CV ROC-AUC | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|
| logistic_regression **(selected)** | 0.8765 | 0.7865 | 0.7867 | 0.7747 | 0.7807 | 0.8812 |
| random_forest | 0.8515 | 0.7525 | 0.7500 | 0.7431 | 0.7465 | 0.8395 |
| xgboost | 0.8698 | 0.7775 | 0.7718 | 0.7757 | 0.7738 | 0.8737 |

## longevity_reg (regression, target `longevity_years`)

| model | CV RMSE | mae | rmse | r2 |
|---|---|---|---|---|
| ridge **(selected)** | 2.8724 | 1.8426 | 2.8603 | 0.7185 |
| random_forest | 3.9295 | 2.5701 | 4.0464 | 0.4367 |
| xgboost | 3.0789 | 1.9444 | 3.0794 | 0.6737 |

## longevity_cls (classification, target `lasts_5_years`)

| model | CV ROC-AUC | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|
| logistic_regression **(selected)** | 0.9263 | 0.8375 | 0.8480 | 0.8535 | 0.8507 | 0.9257 |
| random_forest | 0.8916 | 0.7985 | 0.7837 | 0.8682 | 0.8238 | 0.8854 |
| xgboost | 0.9143 | 0.8270 | 0.8250 | 0.8645 | 0.8443 | 0.9150 |
