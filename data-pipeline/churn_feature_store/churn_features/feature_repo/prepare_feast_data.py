import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd


def prepare_data_for_feast(
    input_path: str,
    output_path: str = "data/processed_churn_data.parquet",
) -> pd.DataFrame:
    """
    Clean, engineer features, and convert raw or processed churn CSV to Feast-compatible Parquet.
    """
    df = pd.read_csv(input_path).dropna().copy()

    current_time = datetime.now()
    df["event_timestamp"] = current_time - pd.to_timedelta(
        np.random.randint(0, 90 * 24 * 60 * 60, size=len(df)),
        unit="s",
    )
    df["created_timestamp"] = current_time
    df["customer_id"] = df["CustomerID"].astype(str)

    column_mapping = {
        "Age": "age",
        "Gender": "gender",
        "Tenure": "tenure_months",
        "Usage Frequency": "usage_frequency",
        "Support Calls": "support_calls",
        "Payment Delay": "payment_delay_days",
        "Subscription Type": "subscription_type",
        "Contract Length": "contract_length",
        "Total Spend": "total_spend",
        "Last Interaction": "last_interaction_days",
        "Churn": "churned",
        "Tenure_Age_Ratio": "tenure_age_ratio",
        "Spend_per_Usage": "spend_per_usage",
        "Support_Calls_per_Tenure": "support_calls_per_tenure",
    }
    df = df.rename(columns=column_mapping)

    for col in ("age", "tenure_months", "support_calls", "last_interaction_days", "churned"):
        if col in df.columns:
            df[col] = df[col].astype("int64")

    for col in ("usage_frequency", "payment_delay_days", "total_spend"):
        if col in df.columns:
            df[col] = df[col].astype("float64")

    if "tenure_age_ratio" not in df.columns:
        df["tenure_age_ratio"] = df["tenure_months"] / (df["age"] + 1)

    if "spend_per_usage" not in df.columns:
        df["spend_per_usage"] = df["total_spend"] / (df["usage_frequency"] + 1)

    if "support_calls_per_tenure" not in df.columns:
        df["support_calls_per_tenure"] = df["support_calls"] / (df["tenure_months"] + 1)

    if "avg_monthly_spend" not in df.columns:
        df["avg_monthly_spend"] = df["total_spend"] / np.maximum(df["tenure_months"], 1)

    if "churn_risk_score" not in df.columns:
        df["churn_risk_score"] = (
            df["payment_delay_days"] * 0.3
            + (df["support_calls"] / np.maximum(df["tenure_months"], 1)) * 0.2
            + (1 - (df["last_interaction_days"] / 30)) * 0.5
        ).clip(0, 1)

    feast_columns = [
        "customer_id",
        "event_timestamp",
        "created_timestamp",
        "age",
        "gender",
        "tenure_months",
        "usage_frequency",
        "support_calls",
        "payment_delay_days",
        "subscription_type",
        "contract_length",
        "total_spend",
        "last_interaction_days",
        "tenure_age_ratio",
        "spend_per_usage",
        "support_calls_per_tenure",
        "avg_monthly_spend",
        "churn_risk_score",
        "churned",
    ]

    df_feast = df[feast_columns].copy()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_feast.to_parquet(output_path, index=False)

    print(f"Data prepared for Feast. Shape: {df_feast.shape}")
    print(f"Saved to: {output_path}")

    return df_feast


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare churn CSV data for Feast/training/evaluation")
    parser.add_argument(
        "--input",
        type=str,
        default="../../../data/processed/df_processed.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed_churn_data.parquet",
        help="Path to output Parquet file",
    )
    args = parser.parse_args()
    prepare_data_for_feast(args.input, args.output)
