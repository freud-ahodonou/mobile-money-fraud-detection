import numpy as np

def add_features(df, freq_maps=None, fit=False):
    df = df.copy()

    eps = 1e-6

    # =====================
    # Features numériques
    # =====================
    df["amount_log"] = np.log1p(df["amount"])

    df["balance_origin_change"] = (
        df["origin_balance_after"]
        - df["origin_balance_before"]
    )

    df["balance_dest_change"] = (
        df["destination_balance_after"]
        - df["destination_balance_before"]
    )

    df["same_balance_change"] = (
        np.abs(df["balance_origin_change"])
        ==
        np.abs(df["balance_dest_change"])
    ).astype(int)

    df["amount_ratio_origin"] = (
        df["amount"]
        / (np.abs(df["origin_balance_before"]) + eps)
    )

    df["amount_ratio_dest"] = (
        df["amount"]
        / (np.abs(df["destination_balance_before"]) + eps)
    )

    df["balance_gap"] = (
        df["origin_balance_before"]
        - df["destination_balance_before"]
    )

    df["total_balance"] = (
        df["origin_balance_before"]
        + df["destination_balance_before"]
    )

    df["amount_ratio_total"] = (
        df["amount"]
        / (np.abs(df["total_balance"]) + eps)
    )

    # =====================
    # Features par type d'opération
    # =====================
    df["operation_amount_mean"] = (
        df.groupby("operation")["amount"]
        .transform("mean")
    )

    df["operation_amount_std"] = (
        df.groupby("operation")["amount"]
        .transform("std")
    )

    # =====================
    # Nombre d'opérations par compte
    # =====================
    df["origin_account_operation_count"] = (
        df.groupby("origin_account")["operation"]
        .transform("count")
    )

    df["destination_account_operation_count"] = (
        df.groupby("destination_account")["operation"]
        .transform("count")
    )

    # =====================
    # Features temporelles
    # =====================
    if "period" in df.columns:

        df["period"] = (
            df["period"]
            .fillna(0)
            .astype(int)
        )

        df["amount_mean_period"] = (
            df.groupby("period")["amount"]
            .transform("mean")
        )

        df["amount_std_period"] = (
            df.groupby("period")["amount"]
            .transform("std")
        )

        df["amount_percent_period"] = (
            df["amount"]
            / (df["amount_mean_period"] + eps)
        )

    # =====================
    # Frequency encoding
    # =====================
    if fit or freq_maps is None:

        freq_maps = {

            "origin_account":
            df["origin_account"]
            .value_counts(normalize=True)
            .to_dict(),

            "destination_account":
            df["destination_account"]
            .value_counts(normalize=True)
            .to_dict()
        }

    df["origin_account_freq"] = (
        df["origin_account"]
        .map(freq_maps["origin_account"])
        .fillna(0)
    )

    df["destination_account_freq"] = (
        df["destination_account"]
        .map(freq_maps["destination_account"])
        .fillna(0)
    )

    # =====================
    # Suppression des IDs bruts
    # =====================
    df = df.drop(
        columns=[
            "origin_account",
            "destination_account"
        ],
        errors="ignore"
    )

    return df, freq_maps