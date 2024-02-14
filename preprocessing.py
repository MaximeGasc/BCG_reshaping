import pandas as pd
import numpy as np


def create_transaction_df(df):
    """
    Aggregate original dataset to transaction level instead of product level and
    calculate relevant KPIs.
    """
    # Drop rows with returns
    df = df[df["sales_net"] >= 0]
    # df['date_order'] = pd.to_datetime(df['date_order']).copy()
    df.loc[:, "date_order"] = pd.to_datetime(df["date_order"])

    # Aggregate at the transaction level
    df_transac = (
        df.groupby(["client_id", "date_order"])
        .agg(
            nb_distinct_prod=("product_id", "nunique"),
            total_sales_net=("sales_net", "sum"),
            order_channel=("order_channel", "first"),
            total_quantity=("quantity", "sum"),
        )
        .reset_index()
    )

    return df_transac


def avg_nb_distinct_products_per_order(df_transac):
    df_client = df_transac.groupby("client_id").agg(
        avg_nb_distinct_prod_per_order=("nb_distinct_prod", "mean")
    )
    return df_client


def count_orders(df_transac, df_features):
    df_client = df_transac.groupby("client_id").agg(nb_orders=("date_order", "count"))

    df_features = pd.merge(df_features, df_client, on=["client_id"], how="left")

    return df_features


def avg_frequency_orders(df_transac, df_features):
    df_client = df_transac.groupby("client_id").agg(
        time_delta=("date_order", lambda x: (x.max() - x.min()).days)
    )

    df_features = pd.merge(df_features, df_client, on=["client_id"], how="left")

    df_features["avg_freq_orders"] = np.where(
        df_features["time_delta"] == 0,
        0,
        df_features["nb_orders"] / df_features["time_delta"],
    )

    return df_features


def total_sales(df_transac, df_features):
    df_client = df_transac.groupby("client_id").agg(
        total_sales_net=("total_sales_net", "sum")
    )

    df_features = pd.merge(df_features, df_client, on=["client_id"], how="left")

    return df_features


def average_basket(df_features):
    df_features["avg_basket"] = (
        df_features["total_sales_net"] / df_features["nb_orders"]
    )

    return df_features


def pct_order_channel(df_transac, df_features):
    df_client = (
        df_transac.groupby("client_id")["order_channel"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    df_client.columns = [
        "pct_" + col.lower().replace(" ", "_") for col in df_client.columns
    ]  # Rename columns
    df_client.reset_index(
        inplace=True
    )  # Reset index to make client_id a regular column

    df_features = pd.merge(df_features, df_client, on=["client_id"], how="left")

    return df_features


def time_since_last_order(df_transac, df_features, end_date):
    df_client = df_transac.groupby("client_id").agg(
        time_since_last_order=("date_order", lambda x: (end_date - x.max()).days)
    )

    df_features = pd.merge(df_features, df_client, on=["client_id"], how="left")

    return df_features


def std_order_frequency(df_transac, df_features):
    df_client = df_transac.groupby("client_id").agg(
        std_order_freq=("date_order", "std")
    )

    df_features = pd.merge(df_features, df_client, on=["client_id"], how="left")

    df_features["std_order_freq"] = df_features["std_order_freq"].dt.days

    return df_features


def create_features(df, end_train_date):

    # Create transaction level database
    df_transac = create_transaction_df(df)
    df_transac = df_transac[df_transac["date_order"] <= end_train_date]

    df_features = avg_nb_distinct_products_per_order(df_transac)
    df_features = count_orders(df_transac, df_features)
    df_features = avg_frequency_orders(df_transac, df_features)
    df_features = total_sales(df_transac, df_features)
    df_features = average_basket(df_features)
    df_features = pct_order_channel(df_transac, df_features)
    df_features = time_since_last_order(df_transac, df_features, end_train_date)
    df_features = std_order_frequency(df_transac, df_features)

    return df_features


def def_temp_window(df, n_days=120):
   '''
   Call the feature generation function, define the temporal window of observation and compute the target
   '''
    end_date = df.date_order.max()  - pd.Timedelta(days=120)
   
    df_features = create_features(df, end_date)
   
    client_id_no_churn = df[df.date_order > end_date].client_id.to_numpy()
   
    df_features['is_churn'] = df_features.client_id.isin(client_id_no_churn)
   
    return df_features
