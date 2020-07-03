import pandas as pd

path1 = "bmc15mldata1.csv"
path2 = "ex15bmcMerged.csv"

path1_data = pd.read_csv(path1, encoding="utf-8-sig")
path2_data = pd.read_csv(path2, encoding="utf-8-sig")

p_ids_1 = path1_data.patient_ID
p_ids_2 = path2_data.patient_ID

result = path2_data[p_ids_2.isin(p_ids_1)]
print(result.shape)

result.to_csv("bmc15ANDex15bmc.csv", sep='\t')