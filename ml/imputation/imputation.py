import torch
import pandas as pd
import numpy
from MF import MFImputer
from train_util import *


def impute(imputer, data):
    df = data.drop(["patient_ID"], axis=1)
    df_new = imputer.fit_transform(df)
    return df_new


def test_imputer(df_new, df, state_df):
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
    df = ge_state_outcome_df.drop(["patient_ID"], axis=1)
    constant = SimpleImputer(fill_value=0, strategy="constant")
    most_freq = SimpleImputer(strategy='most_frequent')
    mean = SimpleImputer(strategy='mean')
    median = SimpleImputer(strategy='median')
    knn = KNNImputer(n_neighbors=9)
    mf = MFImputer(25).to(device)
    knn_new = impute(knn, ge_state_outcome_df.copy())
    constant_new = impute(constant, ge_state_outcome_df.copy())
    most_freq_new = impute(most_freq, ge_state_outcome_df.copy())
    mean_new = impute(mean, ge_state_outcome_df.copy())
    median_new = impute(median, ge_state_outcome_df.copy())
    mf_new = impute(mf, ge_state_outcome_df.copy())
    U_new = pd.DataFrame(mf.U.detach().cpu().numpy(), index=ge_state_outcome_df.patient_ID)
    V_new = pd.DataFrame(mf.V.detach().cpu().numpy().T, columns=ge_state_outcome_df.columns[
1:])
    U_new.to_csv('patients_embed.csv')
    V_new.to_csv('genes_embed.csv')
    train1, test1 = test_imputer(knn_new, df, state_df)
    train2, test2 = test_imputer(mf_new, df, state_df)
    train3, test3 = test_imputer(constant_new, df, state_df)
    train4, test4 = test_imputer(most_freq_new, df, state_df)
    train5, test5 = test_imputer(mean_new, df, state_df)
    train6, test6 = test_imputer(median_new, df, state_df)
    print("train KNN vs matrix factorization")
    print(train1)
    print(train2)
    print("test KNN vs matrix factorization")
    print(test1)
    print(test2)


main()
