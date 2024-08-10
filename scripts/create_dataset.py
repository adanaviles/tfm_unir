import pandas as pd
from config import SRC_DIR
import numpy as np
import os
from scripts.transform_functions import (feature_eng_logs, feature_eng_members, feature_eng_transactions)
from sklearn.preprocessing import OneHotEncoder

def create_final_dataset(train_or_test: str):

    members = feature_eng_members(train_or_test=train_or_test)
    transactions = feature_eng_transactions(train_or_test=train_or_test)
    logs = feature_eng_logs(train_or_test=train_or_test)
    final_df = pd.merge(members, transactions, on="msno", how='left')
    final_df = pd.merge(final_df, logs, on="msno", how='left')
    time_columns = ["last_login",
                    "first_login",
                    "registration_init_time",
                    "first_transaction",
                    "last_transaction"]
    for col in time_columns:
        final_df[col] = pd.to_datetime(final_df[col], errors='coerce')

    # Create new variables and drop direct dates
    # Diff between first and last transaction
    final_df["diff_first_last_transaction"] = (final_df['first_transaction'] - final_df['last_transaction']).dt.days
    # Diff between first and last login
    final_df["diff_max_min_log"] = (final_df['last_login'] - final_df['first_login']).dt.days
    # Days since registration and first login
    final_df["diff_first_log_registration"] = (final_df['first_login'] - final_df['registration_init_time']).dt.days
    # Days since last transaction
    final_df["days_since_last_transaction"] = (pd.to_datetime("2017-03-31") - final_df['last_transaction']).dt.days

    # Days since registration
    final_df["days_since_registration"] = (pd.to_datetime("2017-03-31") - final_df[
        'registration_init_time']).dt.days

    # Days since last transaction
    final_df["diff_days_last_login_last_transaction"] = (final_df['first_login'] - final_df['last_transaction']).dt.days
    # Logs are up to 2017-02-28 so we can compute days since last login
    final_df["days_since_last_login"] = (pd.to_datetime("2017-03-31") - final_df['last_login']).dt.days
    # Users that has less than 5 logins are rare
    final_df["rare_user"] = final_df['amount_of_logins'].apply(lambda x: 0 if (x > 5) else 1)

    # Registration_init_time_year and month
    final_df["registration_init_time_year"] = final_df["registration_init_time"].dt.year
    final_df["registration_init_time_month"] =final_df["registration_init_time"].dt.month

    # Use OHE on registered_via, city and registration_init_year/month
    ohe = OneHotEncoder(sparse_output=False)
    cols_to_do_ohe = ["registered_via", "city", "registration_init_time_year", "registration_init_time_month"]
    for col in cols_to_do_ohe:
        col_ohe = ohe.fit_transform(final_df[[col]].astype(int))
        # Step 4: Convert the result into a DataFrame with appropriate column names
        col_ohe_df = pd.DataFrame(col_ohe, columns=ohe.get_feature_names_out([col]))
        final_df = pd.concat([final_df, col_ohe_df], axis=1)
        final_df.drop(columns=[col], inplace=True)

    # Drop time columns
    final_df.drop(columns=time_columns, inplace=True)
    final_df.to_csv(f"final_{train_or_test}_dataset.csv")

    print("final dataset done")
    return final_df

 #estudio UrGLhudJ+bt/OdieMmVmNIxHL/8tuOv/8ZjQ4UI2nsA=
 # "0iUI/F38xwvgFrndDx0d3hFr8EbV2ew1gfTpyEyv/fE="
if __name__ == "__main__":
    # df = create_final_dataset(train_or_test="train")
    df = create_final_dataset(train_or_test="test")
