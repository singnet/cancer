import pandas as pd
from pathlib import Path
import glob


study_paths = glob.glob("study*.csv")


home = str(Path.home())

gene_expr_data_list = []
for study_pth in study_paths:
    gene_expr_data = pd.read_csv(study_pth, encoding="utf-8-sig")
    gene_expr_data_list.append(gene_expr_data)

gen_ex_res = pd.concat(gene_expr_data_list, join='inner')
print(gen_ex_res.shape)
col_ck_min = min(list(gene_expr_data['A2M']))
col_ck_max = max(list(gene_expr_data['A2M']))
# print(min(list(gen_ex_res['A2M'])))
# print(max(list(gen_ex_res['A2M'])))
# print(gene_expr_data.shape)
ml_treat_data = pd.read_csv('bmc15mldata1.csv')
print(ml_treat_data.shape)
gene_expr_data_all = pd.read_csv("ex15bmcMerged.csv")
print(gene_expr_data_all.shape)
patient_num = ml_treat_data.shape[0]
gene_set_1 = set(gen_ex_res['ABCF1'])
gene_set_2 = set(gene_expr_data_all['ABCF1'])
#checking for an equality of sets for 1 sample gene
print(gene_set_1 == gene_set_2)

patient_id_col_ml = ml_treat_data['patient_ID']
patient_id_col_ml_0 = ml_treat_data.patient_ID
patient_set_ml = set(ml_treat_data['patient_ID'])
patient_set_gene = set(gene_expr_data_all['patient_ID'])
patient_set_gene_raw = set(gen_ex_res['patient_ID'])
diff_set = patient_set_gene.difference(patient_set_ml)
diff_set2 = patient_set_gene_raw.difference(patient_set_ml)

print(diff_set==diff_set2)

# Preview the first 5 lines of the loaded data


# binary treatment variables


# radio - radiotherapy (442/2225)
#
# surgery - breast sparing vs mastectomy (144/2225)
#
# hormone - hormone therapy (855/2225)
#
# chemo - chemotherapy (1186/2225)

#
# outcome variables
#
#
# pCR - complete remission by pathology (260/1134)
#
# RFS - relapse free survival (736/1030)
#
# DFS - disease free survival (512/656)
#
# posOutcome - any of the above are positive (1386/2225)

# print(ml_treat_data.head())

slice_rad = ml_treat_data[(ml_treat_data.radio == 1)]
rad_num = slice_rad.shape[0]
rad_ratio = rad_num/patient_num
slice_surg = ml_treat_data[(ml_treat_data.surgery == 1)]
surg_num = slice_surg.shape[0]
surg_rat = surg_num/patient_num
treat_intersec_slice = ml_treat_data[(ml_treat_data.radio == 1) & (ml_treat_data.surgery == 1) & (ml_treat_data.hormone == 1) & (ml_treat_data.chemo == 1)]
treat_intersec_slice = ml_treat_data[(ml_treat_data.radio == 1) & (ml_treat_data.hormone == 1) & (ml_treat_data.chemo == 1) & (ml_treat_data.posOutcome == 1)]
print(treat_intersec_slice.shape)
#& (ml_treat_data.index.isin([0, 2, 4]))
print(treat_intersec_slice.shape)


