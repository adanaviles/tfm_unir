import pickle

import pandas as pd

from scripts.config import SRC_DIR

test = pd.read_csv(SRC_DIR / "data/preprocessed/final_test_dataset.csv", index_col=0)

X_test = test.drop(columns=["msno", "is_churn"])
y_test = test["is_churn"]

loaded_model_1 = pickle.load(open("lightgbm_model_gs.pkl", "rb"))
loaded_model_2 = pickle.load(open("xgboost_model_gs.pkl", "rb"))

# Generate predictions
preds_1 = loaded_model_1.predict_proba(X_test)[:, 1]
preds_2 = loaded_model_2.predict_proba(X_test)[:, 1]

y_pred_probs = (preds_1 + preds_2) / 2

msno = pd.DataFrame(test["msno"])
msno["is_churn"] = y_pred_probs
msno.to_csv("test_combination_submission_xgb_lgbm.csv", index=False)
