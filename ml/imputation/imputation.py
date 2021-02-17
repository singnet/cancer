import torch
import pandas as pd
import numpy
from MF import MFImputer
from util import *


def test_imputer(ge_state_outcome_df, state_df, imputer):
    knn_imputer = imputer
    df = ge_state_outcome_df.drop(["patient_ID"], axis=1)
    df_new = knn_imputer.fit_transform(df)
    ge_state_outcome_df_v2 = pd.DataFrame(df_new, columns=df.columns)
    patient_id = state_df["patient_ID"]
    ge_state_outcome_df_v2 = pd.concat([patient_id, ge_state_outcome_df_v2], axis=1)
    X, y = ge_state_outcome_df_v2.drop(["patient_ID", "posOutcome"], axis=1), ge_state_outcome_df_v2["posOutcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                        test_size=0.3, random_state=seed)
    params_all, clf_all, cv_scores_all, test_scores_all = evaluate_ge(X_train, y_train, X_test, y_test, state_df.columns.tolist())

    return cv_scores_all, test_scores_all


def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    ge_df = pd.read_csv("datasets/merged-combat15.csv")
    state_df = pd.read_csv("datasets/state_and_outcome.csv")
    state_df = state_df.drop(["series_id", "channel_count", "RFS", "DFS",
                          "pCR", "posOutcome2"] , axis=1)
    gpl_vals = state_df["gpl"].unique()
    print(gpl_vals)
    pam_subtypes = state_df["pam_coincide"].unique()
    print(pam_subtypes)
    p5_types = state_df["p5"].unique()
    print(p5_types)
    tumor_types = state_df["tumor"].unique()
    print(tumor_types)
    state_df = state_df.dropna(axis=0, subset=["pam_coincide", "p5"])
    state_df = state_df.reset_index(drop=True)
    state_df["tumor"] = state_df["tumor"].astype("category").cat.codes
    state_df["pam_coincide"] = pam_code_df = state_df["pam_coincide"].astype("category").cat.codes
    state_df["p5"] = p5_code_df = state_df["p5"].astype("category").cat.codes
    state_df["gpl"] = gpl_code = state_df["gpl"].astype("category").cat.codes
    ge_state_outcome_df = pd.merge(state_df, ge_df, on="patient_ID")

    train1, test1 = test_imputer(ge_state_outcome_df.copy(), state_df, KNNImputer(n_neighbors=9))
    train2, test2 = test_imputer(ge_state_outcome_df.copy(), state_df, MFImputer(25).to(device))
    print("train KNN vs matrix factorization")
    print(train1)
    print(train2)
    print("test KNN vs matrix factorization")
    print(test1)
    print(test2)


main()
