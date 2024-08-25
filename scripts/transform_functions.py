import os

import numpy as np
import pandas as pd
from config import SRC_DIR


def feature_eng_transactions(train_or_test: str):
    """
    Feature eng of transactions
    :param train_or_test:
    :return:
    """
    transactions = pd.read_csv(SRC_DIR / "data/internal/transactions.csv")
    if train_or_test == "test":
        df_to_merge = pd.read_csv(SRC_DIR / "data/external/sample_submission_v2.csv")
        transactions = pd.merge(left=df_to_merge, right=transactions, on="msno", how="left")
    elif train_or_test == "train":
        df_to_merge = pd.read_csv(SRC_DIR / "data/internal/train.csv", low_memory=False)
        transactions = pd.merge(left=df_to_merge, right=transactions, on="msno", how="left")
    else:
        return "Train or test must be provided"
    print(f"Computing transactions for {train_or_test} set")
    # Duplicated data can be here, remove it
    transactions.drop_duplicates(inplace=True)  # Can be duplicates due to new data
    # With date columns
    date_cols = ["transaction_date", "membership_expire_date"]
    for col in date_cols:
        transactions[col] = pd.to_datetime(transactions[col], format="%Y%m%d", errors="coerce")
    # Discounts
    transactions["diff_planned_paid"] = transactions["plan_list_price"] - transactions["actual_amount_paid"]
    # Paid less than expected
    transactions["has_paid_less"] = transactions["diff_planned_paid"].apply(lambda x: 1 if x > 0 else 0)
    transactions["has_paid_more"] = transactions["diff_planned_paid"].apply(lambda x: 1 if x < 0 else 0)

    # If paymant_plan_days is zero we say he paid max per date
    transactions["paid_per_day"] = np.where(
        transactions["payment_plan_days"] != 0,
        (transactions["actual_amount_paid"] / transactions["payment_plan_days"]),
        transactions["actual_amount_paid"],
    )

    transactions = (
        transactions.groupby("msno")
        .agg(
            payment_plan_days_mean=("payment_plan_days", "mean"),
            plan_list_price_mean=("plan_list_price", "mean"),
            actual_amount_paid_mean=("actual_amount_paid", "mean"),
            most_freq_payment_method=("payment_method_id", lambda x: x.mode()[0] if not x.mode().empty else None),
            amount_paid_less=("has_paid_less", "sum"),
            amount_paid_more=("has_paid_more", "sum"),
            amount_cancels=("is_cancel", "sum"),
            mean_paid_per_day=("paid_per_day", "mean"),
            amount_transactions=("transaction_date", "nunique"),
            total_autorenew=("is_auto_renew", "sum"),
            first_transaction=("transaction_date", "min"),
            last_transaction=("transaction_date", "max"),
        )
        .reset_index()
    )
    transactions["percent_of_autorenew"] = transactions["total_autorenew"] / transactions["amount_transactions"]

    return transactions


def feature_eng_members(train_or_test: str):
    """
    Feature eng of members
    :param train_or_test:
    :return:
    """
    members = pd.read_csv(SRC_DIR / "data/internal/members.csv")
    if train_or_test == "test":
        df_to_merge = pd.read_csv(SRC_DIR / "data/external/sample_submission_v2.csv")
        members = pd.merge(left=df_to_merge, right=members, on="msno", how="left")
    elif train_or_test == "train":
        df_to_merge = pd.read_csv(SRC_DIR / "data/internal/train.csv")
        members = pd.merge(left=df_to_merge, right=members, on="msno", how="left")
    else:
        return "Train or test must be provided"
    print(f"Computing members for {train_or_test} set")

    date_cols = ["registration_init_time"]
    for col in date_cols:
        members[col] = pd.to_datetime(members[col], format="%Y%m%d", errors="coerce")

    # Fill outliers for age, replace upper with 80 and lower with 5
    members["bd"] = members["bd"].apply(lambda x: x if (x < 80) and (x > 5) else np.nan)

    members["bd"] = members["bd"].fillna(np.mean(members["bd"]))

    # For city
    members["city"] = members["city"].fillna(0)

    # For registered_via
    members["registered_via"] = members["registered_via"].fillna(0)

    # For registration_init_time we have data up to 2/28/2017 replace outliers
    members["registration_init_time"] = members["registration_init_time"].apply(
        lambda x: x if (x < pd.to_datetime("20170228", format="%Y%m%d")) else np.nan,
    )

    # For registration_init_time we have data up to 2/28/2017 replace outliers
    members["registration_init_time"] = members["registration_init_time"].fillna(
        np.mean(members["registration_init_time"]),
    )
    # Drop gender, too many nans.
    members.drop(columns=["gender"], inplace=True)

    return members


def feature_eng_logs(train_or_test: str):
    """
    Feature eng of logs
    :param train_or_test:
    :return:
    """
    if train_or_test == "test":
        df_to_merge = pd.read_csv(SRC_DIR / "data/external/sample_submission_v2.csv")
        temp_dir = "temp_chunks_monthly_test"
    elif train_or_test == "train":
        df_to_merge = pd.read_csv(SRC_DIR / "data/internal/train.csv")
        temp_dir = "temp_chunks_monthly_train"
    else:
        return "Train or test must be provided"

    chunk_size = 5_000_000  # Adjust based on memory availability
    os.makedirs(temp_dir, exist_ok=True)
    chunk_files = []

    columns_to_clip = [
        "num_25",
        "num_50",
        "num_75",
        "num_985",
        "num_100",
        "num_unq",
        "total_secs",
        "days_between_logins",
    ]

    for i, chunk in enumerate(
        pd.read_csv(SRC_DIR / "data/internal/user_logs.csv", chunksize=chunk_size, parse_dates=["date"]),
        start=1,
    ):
        # Merge with test to only use test data
        chunk = pd.merge(df_to_merge, chunk, on="msno", how="left")
        # We have plenty of data we can drop nans
        chunk = chunk.dropna()

        # Drop duplicates due to new data
        chunk = chunk.drop_duplicates()

        print(f"Processing chunk {i}")
        # Columns year and month
        chunk["year"] = chunk.date.dt.year
        chunk["month"] = chunk.date.dt.month
        # New column being the difference in days between logins
        chunk.sort_values(by=["msno", "date"], ascending=True, inplace=True)
        chunk["days_between_logins"] = chunk.groupby("msno")["date"].diff().dt.days
        chunk["days_between_logins"] = chunk["days_between_logins"].fillna(0)

        chunk[columns_to_clip] = chunk[columns_to_clip].clip(lower=0)

        # Using filtered chunk to do total aggregations
        chunk_agg_total = (
            chunk.groupby("msno")
            .agg(
                num_25_mean=("num_25", "mean"),
                num_50_mean=("num_50", "mean"),
                num_75_mean=("num_75", "mean"),
                num_985_mean=("num_985", "mean"),
                num_100_mean=("num_100", "mean"),
                num_unq_mean=("num_unq", "mean"),
                total_secs_mean=("total_secs", "mean"),
                days_between_logins_mean=("days_between_logins", "mean"),
                num_25_sum=("num_25", "sum"),
                num_50_sum=("num_50", "sum"),
                num_75_sum=("num_75", "sum"),
                num_985_sum=("num_985", "sum"),
                num_100_sum=("num_100", "sum"),
                num_unq_sum=("num_unq", "sum"),
                total_secs_sum=("total_secs", "sum"),
                days_between_logins_sum=("days_between_logins", "sum"),
                # num_25_std=('num_25', 'std'),
                # num_50_std=('num_50', 'std'),
                # num_75_std=('num_75', 'std'),
                # num_985_std=('num_985', 'std'),
                # num_100_std=('num_100', 'std'),
                # num_unq_std=('num_unq', 'std'),
                # total_secs_std=('total_secs', 'std'),
                # days_between_logins_std=('days_between_logins', 'std'),
                first_login=("date", "min"),
                last_login=("date", "max"),
                amount_of_logins=("msno", "count"),
            )
            .reset_index()
        )

        # For montly values means
        monthly_agg = (
            chunk.groupby(["msno", "year", "month"])
            .agg(
                num_25_mean_monthly=("num_25", "mean"),
                num_50_mean_monthly=("num_50", "mean"),
                num_75_mean_monthly=("num_75", "mean"),
                num_985_mean_monthly=("num_985", "mean"),
                num_100_mean_monthly=("num_100", "mean"),
                num_unq_mean_monthly=("num_unq", "mean"),
                total_secs_mean_monthly=("total_secs", "mean"),
                days_between_logins_mean_monthly=("days_between_logins", "mean"),
                num_25_sum_monthly=("num_25", "sum"),
                num_50_sum_monthly=("num_50", "sum"),
                num_75_sum_monthly=("num_75", "sum"),
                num_985_sum_monthly=("num_985", "sum"),
                num_100_sum_monthly=("num_100", "sum"),
                num_unq_sum_monthly=("num_unq", "sum"),
                total_secs_sum_monthly=("total_secs", "sum"),
            )
            .reset_index()
        )

        monthly_agg.sort_values(by=["msno", "year", "month"], ascending=False, inplace=True)
        monthly_agg["month_rank"] = monthly_agg.groupby("msno").cumcount() + 1
        monthly_agg = monthly_agg[monthly_agg["month_rank"] < 4]
        means_monthly_pivot = monthly_agg.pivot(
            index="msno",
            columns="month_rank",
            values=monthly_agg.columns[3:-1],
        )  # not adding year, month and msno

        del monthly_agg, chunk

        # Flatten the MultiIndex columns and reset the index to merge back
        means_monthly_pivot.columns = [f"{col}_{month_rank}" for col, month_rank in means_monthly_pivot.columns]
        means_monthly_pivot.reset_index(inplace=True)

        # Merge back the set
        chunk_df = pd.merge(chunk_agg_total, means_monthly_pivot, on="msno", how="left")
        del means_monthly_pivot
        del chunk_agg_total

        chunk_file = os.path.join(temp_dir, f"chunk_{i}.parquet")
        chunk_df.to_parquet(chunk_file)
        chunk_files.append(chunk_file)
    del df_to_merge

    print("Concatenating all chunks...")
    chunk_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".parquet")]
    final_df = pd.DataFrame()
    for i, file in enumerate(chunk_files):
        print(f"Processing file {i + 1}/{len(chunk_files)}: {file}")

        # Load the Parquet file
        chunk_df = pd.read_parquet(file)
        final_df = pd.concat([final_df, chunk_df], axis=0)
        del chunk_df

        final_df = (
            final_df.groupby("msno")
            .agg(
                num_25_mean=("num_25_mean", "mean"),
                num_50_mean=("num_50_mean", "mean"),
                num_75_mean=("num_75_mean", "mean"),
                num_985_mean=("num_985_mean", "mean"),
                num_100_mean=("num_100_mean", "mean"),
                num_unq_mean=("num_unq_mean", "mean"),
                total_secs_mean=("total_secs_mean", "mean"),
                days_between_logins_mean=("days_between_logins_mean", "mean"),
                num_25_sum=("num_25_sum", "sum"),
                num_50_sum=("num_50_sum", "sum"),
                num_75_sum=("num_75_sum", "sum"),
                num_985_sum=("num_985_sum", "sum"),
                num_100_sum=("num_100_sum", "sum"),
                num_unq_sum=("num_unq_sum", "sum"),
                total_secs_sum=("total_secs_sum", "sum"),
                days_between_logins_sum=("days_between_logins_sum", "sum"),
                first_login=("first_login", "min"),
                last_login=("last_login", "max"),
                amount_of_logins=("amount_of_logins", "sum"),
                num_25_mean_monthly_1=("num_25_mean_monthly_1", "mean"),
                num_25_mean_monthly_2=("num_25_mean_monthly_2", "mean"),
                num_25_mean_monthly_3=("num_25_mean_monthly_3", "mean"),
                num_50_mean_monthly_1=("num_50_mean_monthly_1", "mean"),
                num_50_mean_monthly_2=("num_50_mean_monthly_2", "mean"),
                num_50_mean_monthly_3=("num_50_mean_monthly_3", "mean"),
                num_75_mean_monthly_1=("num_75_mean_monthly_1", "mean"),
                num_75_mean_monthly_2=("num_75_mean_monthly_2", "mean"),
                num_75_mean_monthly_3=("num_75_mean_monthly_3", "mean"),
                num_985_mean_monthly_1=("num_985_mean_monthly_1", "mean"),
                num_985_mean_monthly_2=("num_985_mean_monthly_2", "mean"),
                num_985_mean_monthly_3=("num_985_mean_monthly_3", "mean"),
                num_100_mean_monthly_1=("num_100_mean_monthly_1", "mean"),
                num_100_mean_monthly_2=("num_100_mean_monthly_2", "mean"),
                num_100_mean_monthly_3=("num_100_mean_monthly_3", "mean"),
                num_unq_mean_monthly_1=("num_unq_mean_monthly_1", "mean"),
                num_unq_mean_monthly_2=("num_unq_mean_monthly_2", "mean"),
                num_unq_mean_monthly_3=("num_unq_mean_monthly_3", "mean"),
                total_secs_mean_monthly_1=("total_secs_mean_monthly_1", "mean"),
                total_secs_mean_monthly_2=("total_secs_mean_monthly_2", "mean"),
                total_secs_mean_monthly_3=("total_secs_mean_monthly_3", "mean"),
                days_between_logins_mean_monthly_1=("days_between_logins_mean_monthly_1", "mean"),
                days_between_logins_mean_monthly_2=("days_between_logins_mean_monthly_2", "mean"),
                days_between_logins_mean_monthly_3=("days_between_logins_mean_monthly_3", "mean"),
                num_25_sum_monthly_1=("num_25_sum_monthly_1", "sum"),
                num_25_sum_monthly_2=("num_25_sum_monthly_2", "sum"),
                num_25_sum_monthly_3=("num_25_sum_monthly_3", "sum"),
                num_50_sum_monthly_1=("num_50_sum_monthly_1", "sum"),
                num_50_sum_monthly_2=("num_50_sum_monthly_2", "sum"),
                num_50_sum_monthly_3=("num_50_sum_monthly_3", "sum"),
                num_75_sum_monthly_1=("num_75_sum_monthly_1", "sum"),
                num_75_sum_monthly_2=("num_75_sum_monthly_2", "sum"),
                num_75_sum_monthly_3=("num_75_sum_monthly_3", "sum"),
                num_985_sum_monthly_1=("num_985_sum_monthly_1", "sum"),
                num_985_sum_monthly_2=("num_985_sum_monthly_2", "sum"),
                num_985_sum_monthly_3=("num_985_sum_monthly_3", "sum"),
                num_100_sum_monthly_1=("num_100_sum_monthly_1", "sum"),
                num_100_sum_monthly_2=("num_100_sum_monthly_2", "sum"),
                num_100_sum_monthly_3=("num_100_sum_monthly_3", "sum"),
                num_unq_sum_monthly_1=("num_unq_sum_monthly_1", "sum"),
                num_unq_sum_monthly_2=("num_unq_sum_monthly_2", "sum"),
                num_unq_sum_monthly_3=("num_unq_sum_monthly_3", "sum"),
                total_secs_sum_monthly_1=("total_secs_sum_monthly_1", "sum"),
                total_secs_sum_monthly_2=("total_secs_sum_monthly_2", "sum"),
                total_secs_sum_monthly_3=("total_secs_sum_monthly_3", "sum"),
            )
            .reset_index()
        )

        final_df.sort_values(by=["msno", "last_login"], ascending=False, inplace=True)
        final_df = final_df.drop_duplicates(subset=["msno"], keep="last").copy()

    final_df.to_csv(f"logs_info_{train_or_test}.csv")
    return final_df


if __name__ == "__main__":
    feature_eng_transactions("train")
