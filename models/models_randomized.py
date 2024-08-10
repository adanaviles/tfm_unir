import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from scripts.config import SRC_DIR

# Define the models and hyperparameters as you with so you can compare.
MODELS = {
    "adaboost": {
        "model": AdaBoostClassifier(random_state=42),
        "params": {"n_estimators": [100], "learning_rate": [0.1], "algorithm": ["SAMME"]},
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=42, verbose=3),
        "params": {
            "n_estimators": [200],
            "max_features": ["sqrt"],
            "max_depth": [15],
            "min_samples_split": [10],
            "min_samples_leaf": [3],
            "bootstrap": [True],
        },
    },
    "xgboost": {
        "model": XGBClassifier(random_state=42, verbosity=3),
        "params": {
            "n_estimators": [250],
            "max_depth": [30],
            "learning_rate": [0.13],
            "subsample": [0.9],
            "colsample_bytree": [0.7],
        },
    },
    "lightgbm": {
        "model": lgb.LGBMClassifier(random_state=42),
        "params": {"n_estimators": [300], "num_leaves": [127], "learning_rate": [0.1], "subsample": [0.8]},
    },
    "decision_tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {"max_depth": [15], "min_samples_split": [10], "min_samples_leaf": [5]},
    },
    # 'logistic_regression': {
    #     'model': LogisticRegression(random_state=42, max_iter=300, verbose=3),
    #     'params': {
    #         'penalty': ['l1', 'l2', 'elasticnet'],
    #         'C': [1],
    #         'solver': ['liblinear']
    #     }
    # },
    "catboost": {
        "model": CatBoostClassifier(random_state=42, verbose=3),
        "params": {"iterations": [1000], "depth": [6], "learning_rate": [0.1], "l2_leaf_reg": [3]},
    },
}


def preprocess_data(file_path):
    train = pd.read_csv(file_path, index_col=[0])
    X = train.drop(columns=["msno", "is_churn"])
    y = train["is_churn"]
    return X, y


def train_and_evaluate(model_name, X_train, X_test, y_train, y_test, params=None):
    model_info = MODELS.get(model_name)
    if model_info is None:
        raise ValueError(f"Model '{model_name}' is not defined.")

    model = model_info["model"]
    print(f"Computing {model}")
    if params is None:
        params = model_info["params"]

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    if params:
        # rs = RandomizedSearchCV(model, param_distributions=params, n_jobs=3, cv=3)
        # rs.fit(X_train, y_train, sample_weight=sample_weights)
        # best_model = rs.best_estimator_
        # best_params = rs.best_params_
        gs = GridSearchCV(model, param_grid=params, n_jobs=3, cv=3, verbose=3)
        gs.fit(X_train, y_train, sample_weight=sample_weights)
        best_model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        model.fit(X_train, y_train, sample_weight=sample_weights)
        best_model = model
        best_params = None

    y_pred_probs = best_model.predict_proba(X_test)
    logloss = log_loss(y_test, y_pred_probs)
    auc_score = roc_auc_score(y_test, y_pred_probs[:, 1])

    print(f"Model: {model_name}")
    print(f"Log-loss: {logloss:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    if best_params:
        print(f"Best parameters for {model}: {best_params}")

    # Save the model
    with open(f"{model_name}_model_gs.pkl", "wb") as file:
        pickle.dump(best_model, file)

    # # Load the model (for demonstration, loading right after saving)
    # with open(f'{model_name}_model.pkl', 'rb') as file:
    #     loaded_model = pickle.load(file)
    #
    # return loaded_model


if __name__ == "__main__":
    # Load and preprocess the data
    X, y = preprocess_data(SRC_DIR / "data/preprocessed/final_train_dataset.csv")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_filled = X_train.fillna(-99)
    X_test_filled = X_test.fillna(-99)
    # Train and evaluate models
    for model_name in MODELS.keys():
        if (model_name == "logistic_regression") | (model_name == "adaboost"):
            train_and_evaluate(model_name, X_train_filled, X_test_filled, y_train, y_test)
        else:
            train_and_evaluate(model_name, X_train, X_test, y_train, y_test)
