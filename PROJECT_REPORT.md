# House Price Prediction — Project Report

## 1. Executive Summary
This project builds a regression workflow to estimate residential house sale prices from structured property features. It includes exploratory data analysis, feature engineering, preprocessing, comparison of multiple regressors, cross-validation, XGBoost hyperparameter tuning, validation, and Kaggle submission generation.

**Results status:** The supplied notebook does not contain saved execution outputs. Therefore, no validation metrics or Kaggle leaderboard score are claimed in this report. Run the notebook and record the actual results before publishing them.

## 2. Problem Statement
Predict `SalePrice` for houses using the available property attributes. The task is a supervised regression problem.

## 3. Workflow
1. Load the competition training and test CSV files.
2. Inspect column types, missing values, duplicates, and target distribution.
3. Create additional features such as total square footage, total bathrooms, house age, and remodel age.
4. Impute missing numeric and categorical values; scale numeric features and one-hot encode categorical features.
5. Compare Ridge Regression, Random Forest, Extra Trees, Gradient Boosting, and XGBoost.
6. Use a hold-out validation set and five-fold cross-validation.
7. Tune XGBoost with `GridSearchCV`.
8. Evaluate with RMSE, RMSLE, MAE, and R².
9. Fit the selected pipeline on all labeled data and generate `submission.csv` with `Id` and `SalePrice`.

## 4. Models
- Ridge Regression — regularized linear baseline.
- Random Forest Regressor — bagged tree ensemble.
- Extra Trees Regressor — randomized tree ensemble.
- Gradient Boosting Regressor — sequential boosting model.
- XGBoost Regressor — boosted-tree model selected for hyperparameter tuning.

## 5. Evaluation
- **RMSE:** gives greater weight to large prediction errors.
- **RMSLE:** measures prediction error on a logarithmic scale.
- **MAE:** mean absolute error in original price units.
- **R²:** proportion of target variance explained by the model.

Record the actual values after executing the notebook:

| Metric | Result |
|---|---|
| Validation RMSE | Run notebook to populate |
| Validation RMSLE | Run notebook to populate |
| Validation MAE | Run notebook to populate |
| Validation R² | Run notebook to populate |
| Kaggle public score | Add your actual submission score |

## 6. How to Run
### Google Colab
1. Open `house_price_prediction.ipynb` in Google Colab.
2. Run cells in order.
3. Upload `train.csv`, `test.csv`, and `sample_submission.csv` when requested.
4. Review model comparison and validation metrics.
5. Download the generated `submission.csv`.

### Local
```bash
pip install -r requirements.txt
jupyter notebook house_price_prediction.ipynb
```
Place the three competition CSV files in the working directory when prompted or adjust the notebook's data-loading cell to their location.

## 7. Submission Format
The generated CSV contains:
- `Id` — test record identifier.
- `SalePrice` — predicted house sale price.

Do not fabricate leaderboard results. Add the score returned by Kaggle after submitting the file.

## 8. Limitations and Future Improvements
- Results depend on data quality and validation strategy.
- The notebook's current tuning search is limited to its specified XGBoost grid.
- Potential improvements include broader cross-validated tuning, careful outlier analysis, CatBoost/LightGBM comparison where available, and blending models after validating that the blend generalizes.
- Public leaderboard rank is not guaranteed by a single validation score.

## 9. Repository Contents
- `house_price_prediction.ipynb` — analysis and modeling notebook.
- `README.md` — GitHub landing page and run instructions.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes datasets and generated files.
- `PROJECT_REPORT.md` — detailed project report.
