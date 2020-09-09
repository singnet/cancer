import pandas as pd
import glob
min_num_of_genes = 5000 #change this to whatever filtering you need
study_paths = glob.glob("./noNorm/*.csv.xz") #input your path to csv.xz files of dataset
gene_expr_data_list = []
for study_pth in study_paths:
    gene_expr_data = pd.read_csv(study_pth, encoding="utf-8-sig")
    if (gene_expr_data.shape[1] < min_num_of_genes):
        continue
    gene_expr_data_list.append(gene_expr_data)

gen_ex_res_and = pd.concat(gene_expr_data_list, join='inner')
print(gen_ex_res_and.shape)
gen_ex_res_or = pd.concat(gene_expr_data_list, join='outer')
print(gen_ex_res_or.shape)

gen_ex_res_and.to_csv("noNormMergedAnd.csv", index=False)
gen_ex_res_or.to_csv("noNormMergedOr.csv", index=False)
