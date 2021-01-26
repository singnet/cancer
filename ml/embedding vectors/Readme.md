Files description
-----------------

1. Patients embedding vector generated from 50 genes expression data only
   XGBoost selected 50 genes
   - property_vector_xgb50_OnlyGenexpr_2020-12-24.csv

   Moses selected 50 genes
   - property_vector_moses50_OnlyGenexpr_2020-12-24.csv

2. Patients embedding vector generated from 50 genes expression data plus patients state-and-PreTX data
   XGBoost selected 50 genes 
    - property_vector_moses50_withoutplnresult_2020-12-23.csv

   Moses selected 50 genes 
    - property_vector_xgb50_withoutplnresult_2020-12-23.csv

3. Patients embedding vector generated from 50 genes expression data plus patients state-and-PreTX data plus All GO and Pathways
   XGBoost selected 50 genes
   - property_vector_xgb50-allGO-PW_2020-12-25.csv
 
   Moses selected 50 genes 
   - property_vector_moses50-allGO-pW_2020-12-25.csv

4. Patients embedding vector fro 8 genes(hallmark paper) expression data
   - Only Gene expression
	property_vector_8genes_onlyGenexp_2021-01-04.csv

     NB: Patients with the following ID have a zero vector and has been excluded
     "37009","615665","505568","505459","125209","505442","441751","36843","441873","36864"	
 
   - Gene expression plus patients state-and-PreTx data
