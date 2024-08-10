import pickle
from tabnanny import verbose

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from scripts.config import SRC_DIR

test = pd.read_csv(SRC_DIR / "data/preprocessed/final_test_dataset.csv", index_col=0)

X_test = test.drop(columns=["msno", "is_churn"])
y_test = test["is_churn"]

loaded_model_1 = pickle.load(open("catboost_model_gs.pkl", "rb"))
loaded_model_2 = pickle.load(open(f"xgboost_model_gs.pkl", "rb"))

# Generate predictions
preds_1 = loaded_model_1.predict_proba(X_test)[:, 1]
preds_2 = loaded_model_2.predict_proba(X_test)[:, 1]

y_pred_probs = (preds_1 + preds_2) / 2

msno = pd.DataFrame(test["msno"])
msno["is_churn"] = y_pred_probs
msno.to_csv("test_combination_submission.csv", index=False)
###

import lightgbm as lgb

train = pd.read_csv(SRC_DIR / "data/preprocessed/final_train_dataset.csv", index_col=0)
X = train.drop(columns=["msno", "is_churn"])
y = train["is_churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
model = lgb.LGBMClassifier(random_state=42, n_estimators=300, num_leaves=127, learning_rate=0.1, subsample=0.8)
model.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_probs = model.predict_proba(X_test)
logloss = log_loss(y_test, y_pred_probs)
