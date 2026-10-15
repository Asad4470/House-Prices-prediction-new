# House Price Prediction — Machine Learning Regression

A machine-learning project that predicts residential home sale prices using structured housing data. The workflow includes exploratory data analysis, feature engineering, preprocessing for numeric and categorical columns, model comparison, cross-validation, XGBoost hyperparameter tuning, validation, and Kaggle submission generation.

> **Note:** Validation scores and Kaggle leaderboard results are intentionally not listed here because the notebook provided does not contain saved execution outputs. Add your actual results after running the notebook.

## Project Overview

The goal is to estimate `SalePrice` from house characteristics. Since house prices are typically skewed, the notebook trains models on `log1p(SalePrice)` and converts predictions back to the original price scale for interpretation and submission.

## Workflow

1. **Load data** — reads `train.csv`, `test.csv`, and `sample_submission.csv`.
2. **Explore the data** — inspects data types, missing values, duplicates, target statistics, and sale-price distributions.
3. **Feature engineering** — creates:
   - `TotalSF`: basement + first-floor + second-floor square footage.
   - `TotalBathrooms`: full bathrooms plus half-bath equivalents, including basement bathrooms.
   - `HouseAge`: sale year minus year built.
   - `RemodelAge`: sale year minus year remodeled.
4. **Preprocess features** — imputes missing numeric values with the median and scales them; imputes categorical values with the most frequent value and one-hot encodes categories.
5. **Compare regression models** — Ridge Regression, Random Forest, Extra Trees, Gradient Boosting, and XGBoost.
6. **Cross-validation** — uses shuffled 5-fold cross-validation on the training split to compare log-target RMSE.
7. **Tune XGBoost** — uses `GridSearchCV` with a 5-fold CV search over estimator count, tree depth, learning rate, row subsampling, and column subsampling.
8. **Evaluate** — reports validation RMSE, RMSLE, MAE, and R², and plots actual versus predicted prices.
9. **Train final model and submit** — refits the selected XGBoost pipeline on all labeled training data and writes `submission.csv` with `Id` and `SalePrice` columns.

## Models Compared

| Model | Purpose |
|---|---|
| Ridge Regression | Regularized linear baseline |
| Random Forest Regressor | Tree ensemble baseline |
| Extra Trees Regressor | Randomized tree ensemble |
| Gradient Boosting Regressor | Sequential boosting baseline |
| XGBoost Regressor | Boosted-tree model; tuned with cross-validation |

## Evaluation Metrics

- **RMSE** — penalizes larger prediction errors more heavily.
- **RMSLE** — evaluates errors on the logarithmic price scale; relevant to the competition's log-based evaluation.
- **MAE** — average absolute error in the original price units.
- **R²** — proportion of target variance explained on the validation split.

The notebook uses a 80/20 train-validation split with `random_state=42`, plus 5-fold cross-validation. Actual values should be taken from the executed notebook, not estimated in advance.

## Dataset

This notebook expects the following files in the current working directory:

```text
train.csv
 test.csv
 sample_submission.csv
```

The files are not included in this repository by default. Use the dataset supplied for the relevant Kaggle House Prices competition and follow its terms of use.

## How to Run

### Google Colab

1. Upload `house_price_prediction.ipynb` to Google Colab.
2. Run the dependency-installation cell.
3. Upload `train.csv`, `test.csv`, and `sample_submission.csv` when prompted.
4. Run the notebook cells in order.
5. Review the model comparison, cross-validation table, and validation metrics.
6. Download `submission.csv` from the final cell and submit it to Kaggle.

### Local Environment

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

Place the three CSV files beside the notebook and run:

```bash
jupyter notebook house_price_prediction.ipynb
```

**Note:** The notebook currently uses `google.colab.files` for uploading and downloading. For local execution, replace those Colab-specific cells with normal local file access and remove the `files.download(...)` cell.

## Expected Output

The notebook creates:

- Exploratory plots for sale-price distributions.
- A table comparing validation metrics for the candidate models.
- A 5-fold cross-validation comparison table.
- Best XGBoost hyperparameters from grid search.
- Actual-versus-predicted validation plot.
- `submission.csv` with the required `Id` and `SalePrice` columns.

## Repository Structure

```text
house-price-prediction/
├── house_price_prediction.ipynb
├── README.md
└── data/                       # optional; usually keep competition data out of Git
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

## Limitations and Next Steps

- Results depend on the dataset and the executed validation run; no score or rank is guaranteed.
- A single 80/20 split may vary. Compare cross-validation results and keep the validation strategy consistent.
- Consider comparing additional gradient-boosting implementations, testing robust feature transformations, and evaluating a carefully validated blend of strong models.
- Avoid selecting models solely by repeated public-leaderboard submissions; use cross-validation to reduce leaderboard overfitting.

## Skills Demonstrated

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `XGBoost` · `Regression` · `Feature Engineering` · `Missing-Value Imputation` · `One-Hot Encoding` · `Cross-Validation` · `Hyperparameter Tuning` · `Model Evaluation`

## Author

Add your name and profile links here.

- GitHub: [Your GitHub Profile](https://github.com/)
- Kaggle: [Your Kaggle Profile](https://www.kaggle.com/)
