### embedding vector check
# dimensions
# filtered_patient_ppty_vector_2020-12-03.csv
# number of patients = 2236
# vector length = 2178
# 
# patients_ppty_vector_unfilterd_2020-12-03.csv
# number of patients = 2631
# vector length = 2568

library(tidyverse)
v1 <- read_tsv("data/curatedBreastData/embeding_vector_result_8genes/patients_ppty_vector_unfilterd_2020-12-03.csv") %>%
  transmute(patient_id = as.character(patient_id), v = str_remove(vector, "^.")) %>%
  mutate(v = str_remove(v, ".$")) %>%
  separate(v, into = paste0("v", 1:2568), sep = ", ", convert = TRUE)

summary(rowMeans(v1[, -1]))
#       Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -1.013e-03 -2.693e-04 -7.158e-06  0.000e+00  2.665e-04  1.150e-03 

summary(colMeans(v1[, -1]))
#       Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -1.069e-10 -1.000e-17  0.000e+00  4.193e-14  1.000e-17  2.425e-10 

v2 <- read_tsv("data/curatedBreastData/embeding_vector_result_8genes/filtered_patient_ppty_vector_2020-12-03.csv") %>%
  transmute(patient_id = as.character(patient_id), v = str_remove(vector, "^.")) %>%
  mutate(v = str_remove(v, ".$")) %>%
  separate(v, into = paste0("v", 1:2178), sep = ", ", convert = TRUE)

ct <- read_csv("data/curatedBreastData/bcClinicalTable.csv", guess_max = 5000) %>%
  mutate(patient_ID = as.character(patient_ID)) %>%
  rename(patient_id = patient_ID) %>%
  mutate(posOutcome2 = coalesce(RFS, DFS))

ct15 <- read_csv("data/curatedBreastData/bmc15mldata1.csv") %>%
  mutate(patient_ID = as.character(patient_ID)) %>%
  rename(patient_id = patient_ID) %>%
  mutate(posOutcome2 = coalesce(RFS, DFS))

filter(ct, patient_ID %in% v1$patient_id) %>%
  pull(series_id) %>%
  table
# GSE12071          GSE12093           GSE1379          GSE16391          GSE16446 GSE16716,GSE20194 
# 46               136                60                48               114               261 
# GSE17705          GSE18728          GSE19615          GSE19697          GSE20181           GSE2034 
# 298                21               115                 3                52               286 
# GSE21974          GSE21997          GSE22226          GSE22358 GSE25055,GSE25066 GSE25065,GSE25066 
# 32                94               149               122               227               198 
# GSE32646          GSE33658           GSE6577           GSE9893 
# 115                11                88               155 

filter(ct, patient_ID %in% v2$patient_id) %>%
  pull(series_id) %>%
  table
# GSE12093           GSE1379          GSE16391          GSE16446 GSE16716,GSE20194          GSE17705 
#      136                60                48               114               261               298 
# GSE19615          GSE20181           GSE2034          GSE22226          GSE22358 GSE25055,GSE25066 
#      115                52               286               128               122               221 
# GSE25065,GSE25066          GSE32646           GSE9893 
#               125               115               155 

setdiff(filter(ct, patient_ID %in% v1$patient_id) %>% pull(series_id),
        filter(ct, patient_ID %in% v2$patient_id) %>% pull(series_id))
# [1] "GSE12071" "GSE18728" "GSE21974" "GSE33658" "GSE21997" "GSE19697" "GSE6577" 

library("factoextra")
fit1 <- princomp(as.matrix(column_to_rownames(v1, "patient_id")), cor=TRUE)
fviz_pca(fit1)

# best clusters ???
km_wss <- fviz_nbclust(v1[, -1], kmeans, method = "wss", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2568 dimensional embedding vector\nwithin kmeans cluster sums of squares")

km_sil <- fviz_nbclust(v1[, -1], kmeans, method = "silhouette", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2568 dimensional embedding vector\naverage silhouette kmeans clusters")

pam_wss <- fviz_nbclust(v1[, -1], cluster::pam, method = "wss", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2568 dimensional embedding vector\nwithin PAM cluster sums of squares")

pam_sil <- fviz_nbclust(v1[, -1], cluster::pam, method = "silhouette", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2568 dimensional embedding vector\naverage silhouette PAM clusters")

gap <- cluster::clusGap(v1[, -1], FUN = kmeans, nstart = 25, K.max = 12, B = 20)
fviz_gap_stat(gap48) + theme_minimal() + ggtitle("kernelPCA 2568 dimensional embedding vector\ngap statistic kmeans clusters")

gap_pam <- cluster::clusGap(v1[, -1], FUN = cluster::pam, K.max = 12, B = 20)
fviz_gap_stat(gap_pam) + theme_minimal() + ggtitle("kernelPCA 2568 dimensional embedding vector\ngap statistic PAM clusters")

# best clusters ???
km_wss2 <- fviz_nbclust(v2[, -1], kmeans, method = "wss", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2178 dimensional embedding vector\nwithin kmeans cluster sums of squares")

km_sil2 <- fviz_nbclust(v2[, -1], kmeans, method = "silhouette", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2178 dimensional embedding vector\naverage silhouette kmeans clusters")

pam_wss2 <- fviz_nbclust(v2[, -1], cluster::pam, method = "wss", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2178 dimensional embedding vector\nwithin PAM cluster sums of squares")

pam_sil2 <- fviz_nbclust(v2[, -1], cluster::pam, method = "silhouette", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2178 dimensional embedding vector\naverage silhouette PAM clusters")

gap2 <- cluster::clusGap(v2[, -1], FUN = kmeans, nstart = 25, K.max = 12, B = 20)
fviz_gap_stat(gap2) + theme_minimal() + ggtitle("kernelPCA 2178 dimensional embedding vector\ngap statistic kmeans clusters")

gap2_pam <- cluster::clusGap(v2[, -1], FUN = cluster::pam, K.max = 12, B = 20)
fviz_gap_stat(gap2_pam) + theme_minimal() + ggtitle("kernelPCA 2178 dimensional embedding vector\ngap statistic PAM clusters")
