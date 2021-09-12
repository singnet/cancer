
#### GO List files

- **pln_gen_mrmr_GOs_5610.csv** : This file contains 60 GO terms that are found both in statistically significant list (p_val < 0.05) of GOs generated by Mike (total 203) and 5610 GO terms generated using pln. The input for PLN experiment are 100 gene expressions selected using MRMR



- **pln_gen_diff_GOs_5610.csv**:  This file contains 94 GO terms that are found both in statistically significant (p_val < 0.05) list of GOs generated by Mike (total 203) and 3146 GO terms generated using PLN. The input for PLN experiment are the top differentially expressed 100 genes.



- **pln_gen_mrmr_GOs_100.csv** -  This file contains 3 GO terms that are found both in statistically significant list (p_val < 0.05) of GOs generated by Mike (total 203) and 100 GO terms selected using MRMR from 5610 property vectors generated using pln. The input for PLN experiment are 100 gene expressions selected using MRMR


- **top_100_pln_ge_mrmr.csv** - This file contains 2 GO terms that are found both in statistically significant list (p_val < 0.05) of GOs generated by Mike (total 203) and 100 GO terms selected by ranking them by their weights.The weights are calculcated from the absolute sum of the coefficients of the solution to Ax = b; where A is the 449x6748 input training data, b is the 449x19 PCA components which are the bases of the PCA subspace.The input for PLN experiment are 100 gene expressions selected using MRMR


- **top_100_pln_ge_diff.csv** -  This file contains 1 GO terms that are found both in statistically significant list (p_val < 0.05) of GOs generated by Mike (total 203) and 100 GO terms selected by ranking them by their weights.The weights are calculcated from the absolute sum of the coefficients of the solution to Ax = b; where A is the 449x6748 input training data, b is the 449x19 PCA components which are the bases of the PCA subspace.The input for PLN experiment are the top differentially expressed 100 genes