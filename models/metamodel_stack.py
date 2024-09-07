import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from scripts.config import SRC_DIR


def preprocess_data(file_path):
    """
    Create preprocess data
    :param file_path:
    :return:
    """
    train = pd.read_csv(file_path, index_col=[0])
    X = train.drop(columns=["msno", "is_churn"])
    y = train["is_churn"]
    return X, y


xgb_model = pickle.load(open("xgboost_model_gs.pkl", "rb"))
lgbm_model = pickle.load(open("lightgbm_model_gs.pkl", "rb"))

# Load and preprocess the data
X, y = preprocess_data(SRC_DIR / "data/preprocessed/final_train_dataset.csv")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_filled = X_train.fillna(-99)
X_test_filled = X_test.fillna(-99)

# Generate predictions
xgb_train_pred = xgb_model.predict_proba(X_train)[:, 1]
xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]

lgbm_train_pred = lgbm_model.predict_proba(X_train)[:, 1]
lgbm_test_pred = lgbm_model.predict_proba(X_test)[:, 1]


# Stack predictions
X_train_stack = np.column_stack((xgb_train_pred, lgbm_train_pred))
X_test_stack = np.column_stack((xgb_test_pred, lgbm_test_pred))


# Train the meta-model
meta_model = LogisticRegression()
meta_model.fit(X_train_stack, y_train)

# Predict with meta-model
meta_train_pred = meta_model.predict_proba(X_train_stack)[:, 1]
meta_test_pred = meta_model.predict_proba(X_test_stack)[:, 1]

# Compute ROC AUC
roc_auc_train = roc_auc_score(y_train, meta_train_pred)
roc_auc_test = roc_auc_score(y_test, meta_test_pred)

# Compute Log Loss
logloss_train = log_loss(y_train, meta_train_pred)
logloss_test = log_loss(y_test, meta_test_pred)

print(f"ROC AUC (train): {roc_auc_train}")
print(f"ROC AUC (test): {roc_auc_test}")
print(f"Log Loss (train): {logloss_train}")
print(f"Log Loss (test): {logloss_test}")


# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({"Model": ["XGBoost", "LightGBM"], "Importance": meta_model.coef_[0]})

print(importance_df)
