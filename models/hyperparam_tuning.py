from tabnanny import verbose

import pandas as pd
import numpy as np
import pickle
from scripts.config import SRC_DIR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, log_loss, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# with open(f'random_forest_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)
#
# test = pd.read_csv(SRC_DIR / 'data/preprocessed/final_test_dataset.csv', index_col=0)
#
# X_test = test.drop(columns=['msno', 'is_churn'])
# y_test = test['is_churn']
#
# y_pred_probs = loaded_model.predict_proba(X_test)[:, 1]
#
# msno = pd.DataFrame(test['msno'])
# msno["is_churn"] = y_pred_probs
# msno.to_csv("test_submission.csv", index=False)
#

from xgboost import XGBClassifier
from xgboost import DMatrix

# test = pd.read_csv(SRC_DIR / 'data/preprocessed/final_test_dataset.csv', index_col=0)
#
# X_test = test.drop(columns=['msno', 'is_churn'])
# y_test = test['is_churn']

train = pd.read_csv(SRC_DIR / 'data/preprocessed/final_train_dataset.csv', index_col=[0])
X_train = train.drop(columns=['msno', 'is_churn'])
y_train = train['is_churn']

from xgboost import XGBClassifier

# Set parameters
params = {'objective': 'binary:logistic',
          'eval_metric': 'logloss'}

# Initialize the XGBClassifier
xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', nthread=-1, n_estimators=12)

# Train the XGBoost model
xgb_model = xgb.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict_proba(X_test)[:,1]

msno = pd.DataFrame(test['msno'])
msno["is_churn"] = y_pred
msno.to_csv("test_submission_xgb_full.csv", index=False)


####

from xgboost import XGBClassifier

# Set parameters
params = {'objective': 'binary:logistic',
          'eval_metric': 'logloss'}

# 'params': {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7, 15],
#     'learning_rate': [0.01, 0.1, 0.3],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0],
# }

# Initialize the XGBClassifier
xgb = XGBClassifier(objective='binary:logistic',
                    eval_metric='logloss',
                    nthread=-1,
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.01,
                    subsample=0.8,
                    verbosity=2)

# Train the XGBoost model
xgb_model = xgb.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict_proba(X_test)[:,1]

msno = pd.DataFrame(test['msno'])
msno["is_churn"] = y_pred
msno.to_csv("test_submission_xgb_full_more_params.csv", index=False)

# retrian less features
feature_importances = xgb_model.feature_importances_
top_50_indices = np.argsort(feature_importances)[-50:][::-1]

X_train_pruned =  X_train[:, top_50_indices]
X_test_pruned =  X_test[:, top_50_indices]

xgb_model = xgb.fit(X_train_pruned, y_train)

# For feature selection
from sklearn.feature_selection import SelectFromModel
