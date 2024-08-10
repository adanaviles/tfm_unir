import itertools
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from scripts.config import SRC_DIR

# Define the models and a single set of hyperparameters for each
LIST_MODELS = [
    "adaboost",
    "random_forest",
    "xgboost",
    "lightgbm",
    "decision_tree",
    # 'logistic_regression',
    "catboost",
]


def preprocess_data(file_path):
    train = pd.read_csv(file_path, index_col=[0])
    X = train.drop(columns=["msno", "is_churn"])
    y = train["is_churn"]
    return X, y


def evaluate_simple_model(model, X_test: pd.DataFrame, y_test):
    loaded_model = pickle.load(open(f"{model}_model_gs.pkl", "rb"))

    # Generate predictions
    y_pred_probs = loaded_model.predict_proba(X_test)[:, 1]
    logloss = log_loss(y_test, y_pred_probs)
    auc_score = roc_auc_score(y_test, y_pred_probs)

    return auc_score, logloss


def evaluate_combination(model_combination, X_test: pd.DataFrame, y_test):
    model_1_name = model_combination[0]
    model_2_name = model_combination[1]

    loaded_model_1 = pickle.load(open(f"{model_1_name}_model_gs.pkl", "rb"))
    loaded_model_2 = pickle.load(open(f"{model_2_name}_model_gs.pkl", "rb"))

    # Generate predictions
    preds_1 = loaded_model_1.predict_proba(X_test)[:, 1]
    preds_2 = loaded_model_2.predict_proba(X_test)[:, 1]

    # Step 4: Average the predictions
    y_pred_probs = (preds_1 + preds_2) / 2

    logloss = log_loss(y_test, y_pred_probs)
    auc_score = roc_auc_score(y_test, y_pred_probs)

    return auc_score, logloss


if __name__ == "__main__":
    # Load and preprocess the data
    X, y = preprocess_data(SRC_DIR / "data/preprocessed/final_train_dataset.csv")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_filled = X_train.fillna(-99)
    X_test_filled = X_test.fillna(-99)
    unique_pairs_models = list(itertools.combinations(LIST_MODELS, 2))

    # Create a DataFrame to store the results
    results = pd.DataFrame(columns=["Model", "AUC Score Test", "Log-loss Test"])

    for model in LIST_MODELS:
        auc_score, logloss = evaluate_simple_model(model, X_test_filled, y_test)
        results_df = pd.DataFrame.from_dict(
            {
                "Model": [model],
                "AUC Score Test": [auc_score],
                "Log-loss Test": [logloss],
            },
        )
        results = pd.concat([results, results_df], axis=0)

    for model_comb in unique_pairs_models:
        auc_score, logloss = evaluate_combination(model_comb, X_test_filled, y_test)
        results_df = pd.DataFrame.from_dict(
            {
                "Model": [model_comb[0] + "-" + model_comb[1]],
                "AUC Score Test": [auc_score],
                "Log-loss Test": [logloss],
            },
        )
        results = pd.concat([results, results_df], axis=0)
    # # Save the results DataFrame
    results.to_csv("model_combination_logloss_results.csv", index=False)
