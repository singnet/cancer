### embedding vector check2 with scoped subset (??) links in embedding
# 
# unfiltered_property_vector_pickle_2020-12-07.csv
# number of patients = 2631
# vector length = 2503

library(tidyverse)
v1 <- read_tsv("data/curatedBreastData/embeding_vector_result_8genes/embedding_vector_8genes/unfiltered_property_vector_pickle_2020-12-07.csv") %>%
  transmute(patient_id = as.character(patient_id), v = str_remove(vector, "^.")) %>%
  mutate(v = str_remove(v, ".$")) %>%
  separate(v, into = paste0("v", 1:2503), sep = ", ", convert = TRUE)

summary(rowMeans(v1[, -1]))
#       Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -1.669e-03 -2.443e-04  1.039e-05  0.000e+00  2.580e-04  1.560e-03 

summary(colMeans(v1[, -1]))
#       Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -8.568e-11 -2.000e-17  0.000e+00 -5.140e-15  1.000e-17  7.983e-11 

# property_vector_pickle_2020-12-07.csv
# number of patients = 2236
# vector length = 2178
v2 <- read_tsv("data/curatedBreastData/embeding_vector_result_8genes/embedding_vector_8genes/property_vector_pickle_2020-12-07.csv") %>%
  transmute(patient_id = as.character(patient_id), v = str_remove(vector, "^.")) %>%
  mutate(v = str_remove(v, ".$")) %>%
  separate(v, into = paste0("v", 1:2178), sep = ", ", convert = TRUE)

summary(rowMeans(v2[, -1]))
#       Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -1.148e-03 -3.194e-04  1.666e-05  0.000e+00  3.246e-04  1.312e-03 

summary(colMeans(v2[, -1]))
#       Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -5.886e-16 -1.407e-17 -2.160e-19  3.340e-18  1.422e-17  5.124e-15 

# patients_embedding_vector of the 50_genes data
# Vector length: 2233,
# number of patients: 2237
v3 <- read_tsv("data/curatedBreastData/embeding_vector_result_8genes/property_vector_pickle_2020-12-05.csv") %>%
  transmute(patient_id = as.character(patient_id), v = str_remove(vector, "^.")) %>%
  mutate(v = str_remove(v, ".$")) %>%
  separate(v, into = paste0("v", 1:2233), sep = ", ", convert = TRUE)

setdiff(v3$patient_id, v2$patient_id)
# [1] "125209"

ct <- read_csv("data/curatedBreastData/bcClinicalTable.csv", guess_max = 5000) %>%
  mutate(patient_ID = as.character(patient_ID)) %>%
  rename(patient_id = patient_ID) %>%
  mutate(posOutcome2 = coalesce(RFS, DFS))

ctMod <- select(ct, pam50, series_id, patient_id, age, pCR, RFS, DFS, posOutcome2)

pdata15 <- read_csv("reports/coincideClinDat.csv", guess_max = 4000) %>%
  mutate(patient_ID = as.character(patient_ID)) %>%
  rename(patient_id = patient_ID)

pdataMod <- select(pdata15, p5, p7, pam_coincide, coincide_cat, series_id, patient_id, age, pCR, RFS, DFS, posOutcome, posOutcome2, months2)

# filter(ct, patient_ID %in% v1$patient_id) %>%
#   pull(series_id) %>%
#   table
# GSE12071          GSE12093           GSE1379          GSE16391          GSE16446 GSE16716,GSE20194 
# 46               136                60                48               114               261 
# GSE17705          GSE18728          GSE19615          GSE19697          GSE20181           GSE2034 
# 298                21               115                 3                52               286 
# GSE21974          GSE21997          GSE22226          GSE22358 GSE25055,GSE25066 GSE25065,GSE25066 
# 32                94               149               122               227               198 
# GSE32646          GSE33658           GSE6577           GSE9893 
# 115                11                88               155 

# filter(ct, patient_ID %in% v2$patient_id) %>%
#   pull(series_id) %>%
#   table
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
  theme_minimal() + ggtitle("kernelPCA 2503 dimensional embedding vector\nwithin kmeans cluster sums of squares")

km_sil <- fviz_nbclust(v1[, -1], kmeans, method = "silhouette", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2503 dimensional embedding vector\naverage silhouette kmeans clusters")

pam_wss <- fviz_nbclust(v1[, -1], cluster::pam, method = "wss", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2503 dimensional embedding vector\nwithin PAM cluster sums of squares")

pam_sil <- fviz_nbclust(v1[, -1], cluster::pam, method = "silhouette", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2503 dimensional embedding vector\naverage silhouette PAM clusters")

pam2503_5 <- cluster::pam(v1[, -1], k = 5, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
table(pam2503_5$clustering)
#    1    2    3    4    5 
# 1013  534  326  542  216 

pam2503_7 <- cluster::pam(v1[, -1], k = 7, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
table(pam2503_7$clustering)
# 1   2   3   4   5   6   7 
# 430 672 504 309 494 106 116 


pam2503_9 <- cluster::pam(v1[, -1], k = 9, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
table(pam2503_9$clustering)
#   1   2   3   4   5   6   7   8   9 
# 297 562 363 281 284  80 363 115 286 


pam2503_20 <- cluster::pam(v1[, -1], k = 20, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
table(pam2503_20$clustering)
#   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20 
# 172 112 108 215 195 169  80 109  81 272 139  53  86 115  86 115 190 160 107  67 

pam2503 <- bind_cols(patient_id = pull(v1, patient_id), c5 = paste0("c", pam2503_5$clustering), c7 = paste0("c", pam2503_7$clustering), c9 = paste0("c", pam2503_9$clustering), c20 = paste0("c", pam2503_20$clustering))

# best clusters ???
km_wss2 <- fviz_nbclust(v2[, -1], kmeans, method = "wss", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2178 dimensional embedding vector\nwithin kmeans cluster sums of squares")

km_sil2 <- fviz_nbclust(v2[, -1], kmeans, method = "silhouette", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2178 dimensional embedding vector\naverage silhouette kmeans clusters")

pam_wss2 <- fviz_nbclust(v2[, -1], cluster::pam, method = "wss", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2178 dimensional embedding vector\nwithin PAM cluster sums of squares")

pam_sil2 <- fviz_nbclust(v2[, -1], cluster::pam, method = "silhouette", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2178 dimensional embedding vector\naverage silhouette PAM clusters")

pam2178_5 <- cluster::pam(v2[, -1], k = 5, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
table(pam2178_5$clustering)
#   1   2   3   4   5 
# 446 277 488 197 828 

pam2178_7 <- cluster::pam(v2[, -1], k = 7, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
table(pam2178_7$clustering)
#   1   2   3   4   5   6   7 
# 304 277 730 118 228 499  80 


pam2178_9 <- cluster::pam(v2[, -1], k = 9, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
table(pam2178_9$clustering)
#   1   2   3   4   5   6   7   8   9 
# 305 271 116 469 180 149 420  80 246 


pam2178_20 <- cluster::pam(v2[, -1], k = 20, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
table(pam2178_20$clustering)
#   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20 
# 100 144 115 282 111  83 108 112 155 142  73 108  80  62 134 120  75 103  44  85 

pam2178 <- bind_cols(patient_id = pull(v2, patient_id), c5 = paste0("c", pam2178_5$clustering), c7 = paste0("c", pam2178_7$clustering), c9 = paste0("c", pam2178_9$clustering), c20 = paste0("c", pam2178_20$clustering))

# best clusters 50 gene embedding, 2233 dim vector
km_wss3 <- fviz_nbclust(v3[, -1], kmeans, method = "wss", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2233 dimensional embedding vector\nwithin kmeans cluster sums of squares")

km_sil3 <- fviz_nbclust(v3[, -1], kmeans, method = "silhouette", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2233 dimensional embedding vector\naverage silhouette kmeans clusters")

pam_wss3 <- fviz_nbclust(v3[, -1], cluster::pam, method = "wss", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2233 dimensional embedding vector\nwithin PAM cluster sums of squares")

pam_sil3 <- fviz_nbclust(v3[, -1], cluster::pam, method = "silhouette", k.max = 24) +
  theme_minimal() + ggtitle("kernelPCA 2233 dimensional embedding vector\naverage silhouette PAM clusters")

gap <- cluster::clusGap(v1[, -1], FUN = kmeans, nstart = 25, K.max = 12, B = 20)
fviz_gap_stat(gap48) + theme_minimal() + ggtitle("kernelPCA 2568 dimensional embedding vector\ngap statistic kmeans clusters")

gap2 <- cluster::clusGap(v2[, -1], FUN = kmeans, nstart = 25, K.max = 12, B = 20)
fviz_gap_stat(gap2) + theme_minimal() + ggtitle("kernelPCA 2178 dimensional embedding vector\ngap statistic kmeans clusters")

gap3 <- cluster::clusGap(v2[, -1], FUN = kmeans, nstart = 25, K.max = 12, B = 20)
fviz_gap_stat(gap2) + theme_minimal() + ggtitle("kernelPCA 2233 dimensional embedding vector\ngap statistic kmeans clusters")

gap_pam <- cluster::clusGap(v1[, -1], FUN = cluster::pam, K.max = 12, B = 20)
fviz_gap_stat(gap_pam) + theme_minimal() + ggtitle("kernelPCA 2568 dimensional embedding vector\ngap statistic PAM clusters")

gap2_pam <- cluster::clusGap(v2[, -1], FUN = cluster::pam, K.max = 12, B = 20)
fviz_gap_stat(gap2_pam) + theme_minimal() + ggtitle("kernelPCA 2178 dimensional embedding vector\ngap statistic PAM clusters")

gap3_pam <- cluster::clusGap(v3[, -1], FUN = cluster::pam, K.max = 12, B = 20)
fviz_gap_stat(gap3_pam) + theme_minimal() + ggtitle("kernelPCA 2233 dimensional embedding vector\ngap statistic PAM clusters")

pam2233_5 <- cluster::pam(v3[, -1], k = 5, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
table(pam2233_5$clustering)
#   1    2    3    4    5 
# 185  434  195 1170  253 

pam2233_7 <- cluster::pam(v3[, -1], k = 7, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
table(pam2233_7$clustering)
#   1   2   3   4   5   6   7 
# 182 429 115 146 996 116 253 


pam2233_9 <- cluster::pam(v3[, -1], k = 9, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
table(pam2233_9$clustering)
#   1   2   3   4   5   6   7   8   9 
# 188 433 115 157  80 675 259 268  62 


pam2233_20 <- cluster::pam(v3[, -1], k = 20, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
table(pam2233_20$clustering)
#  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20 
# 58 147 115 136  91 174  80 116 138 102 163 107  77 109  96 104 101 200  54  69 

pam2233 <- bind_cols(patient_id = pull(v3, patient_id), c5 = paste0("c", pam2233_5$clustering), c7 = paste0("c", pam2233_7$clustering), c9 = paste0("c", pam2233_9$clustering), c20 = paste0("c", pam2233_20$clustering))

# score clusters
pamClusters <- list(pam2233 = pam2233, pam2178 = pam2178, pam2503 = pam2503) %>%
  map(~ left_join(., pdataMod))
BiocManager::install()


# make dep vars factors
pamClusters <- map(pamClusters, ~ mutate(., across(pCR:posOutcome2, as.factor)))

# compare subtypes in best model
pam50 <- glm(posOutcome2 ~ pam_coincide, family = binomial(link = logit), data = pamClusters$pam2233)
summary(pam50)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.7696  -1.3261   0.6847   0.8236   1.0357  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)         0.90694    0.12776   7.099 1.26e-12 ***
#   pam_coincideHer2    0.07122    0.21354   0.334  0.73874    
# pam_coincideLumA    0.42440    0.15372   2.761  0.00576 ** 
#   pam_coincideLumB   -0.19578    0.17285  -1.133  0.25737    
# pam_coincideNormal -0.56400    0.30665  -1.839  0.06588 .  
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1931.6  on 1685  degrees of freedom
# Residual deviance: 1904.6  on 1681  degrees of freedom
# (551 observations deleted due to missingness)
# AIC: 1914.6
# 
# Number of Fisher Scoring iterations: 4

pam50pred <- predict(pam50, type = "response") >= 0.5
table(pam50pred)
# TRUE 
# 1686 
pam50cm <- caret::confusionMatrix(data = as.factor(pam50pred),
                                  reference = as.factor(pamClusters$posOutcome2[as.numeric(names(pam50pred))] == "1"))
pam50cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction FALSE TRUE
#      FALSE    26    7
#      TRUE     17   83
# 
# Accuracy : 0.8195          
# 95% CI : (0.7435, 0.8808)
# No Information Rate : 0.6767          
# P-Value [Acc > NIR] : 0.0001642       
# 
# Kappa : 0.5609          
# 
# Mcnemar's Test P-Value : 0.0661926       
#                                           
#             Sensitivity : 0.6047          
#             Specificity : 0.9222          
#          Pos Pred Value : 0.7879          
#          Neg Pred Value : 0.8300          
#              Prevalence : 0.3233          
#          Detection Rate : 0.1955          
#    Detection Prevalence : 0.2481          
#       Balanced Accuracy : 0.7634          
#                                           
#        'Positive' Class : FALSE           

library(ggfortify)
autoplot(pam50)

gene50 <- glm(posOutcome2 ~ c5, family = binomial(link = logit), data = pamClusters$pam2233)
summary(gene50)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.8781  -1.5353   0.7299   0.8258   0.8576  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   1.5755     0.2006   7.855 3.99e-15 ***
#   c5c2         -0.1737     0.2443  -0.711  0.47705    
# c5c3         -0.5900     0.2572  -2.294  0.02178 *  
#   c5c4         -0.6749     0.2160  -3.125  0.00178 ** 
#   c5c5         -0.7646     0.2455  -3.114  0.00185 ** 
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1931.6  on 1685  degrees of freedom
# Residual deviance: 1910.8  on 1681  degrees of freedom
# (551 observations deleted due to missingness)
# AIC: 1920.8
# 
# Number of Fisher Scoring iterations: 4

autoplot(gene50)

gene50pred <- predict(gene50, type = "response") >= 0.5
table(gene50pred)
# TRUE 
# 1686 

# igancm <- caret::confusionMatrix(data = as.factor(iganpred),
#                                  reference = as.factor(gse9893p$RFS[as.numeric(names(iganpred))] == "1"))
# igancm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction FALSE TRUE
#      FALSE    28    7
#      TRUE     15   83
# 
# Accuracy : 0.8346          
# 95% CI : (0.7603, 0.8933)
# No Information Rate : 0.6767          
# P-Value [Acc > NIR] : 3.022e-05       
# 
# Kappa : 0.6027          
# 
# Mcnemar's Test P-Value : 0.1356          
#                                           
#             Sensitivity : 0.6512          
#             Specificity : 0.9222          
#          Pos Pred Value : 0.8000          
#          Neg Pred Value : 0.8469          
#              Prevalence : 0.3233          
#          Detection Rate : 0.2105          
#    Detection Prevalence : 0.2632          
#       Balanced Accuracy : 0.7867          
#                                           
#        'Positive' Class : FALSE           

autoplot(igan)

# biplot alternate embeddings for gse20194
pdata20194 <- rownames_to_column(pdata20194, "patient_ID") %>%
  mutate(patient_ID = as.character(patient_ID)) %>% 
  left_join(comClust) %>%
  column_to_rownames("patient_ID")
pData(gse20194es) <- pdata20194

s2_20194 <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5))),
                                     title = "GSE20194 microarray data set\n BMC normalization",
                                     colorVar = "k5",  # color = ggplotColours(5),
                                     shapeVar = "pCR", shape = c(1, 4),
                                     sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                     topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                     topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                     topSamplesJust = c(1, 0), topSamplesCex = 3,
                                     returnAnalysis = TRUE)
print(s2_20194$plot)

s2_20194_34 <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5))),
                                        title = "GSE20194 microarray data set\n BMC normalization",
                                        colorVar = "k5", # color = colorPalette,
                                        shapeVar = "pCR", shape = c(1, 4),
                                        sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                        topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                        topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                        topSamplesJust = c(1, 0), topSamplesCex = 3,
                                        dim = 3:4, returnAnalysis = TRUE)
print(s2_20194_34$plot)

s3_20194 <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5))),
                                     title = "GSE20194 microarray data set\n BMC normalization",
                                     colorVar = "p5", # color = colorPalette,
                                     shapeVar = "pCR", shape = c(1, 4),
                                     sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                     topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                     topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                     topSamplesJust = c(1, 0), topSamplesCex = 3,
                                     returnAnalysis = TRUE)
print(s3_20194$plot)

s3_20194_34 <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5))),
                                        title = "GSE20194 microarray data set\n BMC normalization",
                                        colorVar = "p5", # color = colorPalette,
                                        shapeVar = "pCR", shape = c(1, 4),
                                        sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                        topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                        topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                        topSamplesJust = c(1, 0), topSamplesCex = 3,
                                        dim = 3:4, returnAnalysis = TRUE)
print(s3_20194_34$plot)

s4_20194 <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5))),
                                     title = "GSE20194 microarray data set\n BMC normalization",
                                     colorVar = "k7",  color = ggplotColours(5),
                                     shapeVar = "pCR", shape = c(1, 4),
                                     sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                     topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                     topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                     topSamplesJust = c(1, 0), topSamplesCex = 3,
                                     returnAnalysis = TRUE)
print(s4_20194$plot)

s4_20194_34 <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5))),
                                        title = "GSE20194 microarray data set\n BMC normalization",
                                        colorVar = "k7", # color = colorPalette,
                                        shapeVar = "pCR", shape = c(1, 4),
                                        sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                        topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                        topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                        topSamplesJust = c(1, 0), topSamplesCex = 3,
                                        dim = 3:4, returnAnalysis = TRUE)
print(s4_20194_34$plot)

s5_20194 <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5))),
                                     title = "GSE20194 microarray data set\n BMC normalization",
                                     colorVar = "p7", # color = colorPalette,
                                     shapeVar = "pCR", shape = c(1, 4),
                                     sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                     topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                     topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                     topSamplesJust = c(1, 0), topSamplesCex = 3,
                                     returnAnalysis = TRUE)
print(s5_20194$plot)

s5_20194_34 <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5))),
                                        title = "GSE20194 microarray data set\n BMC normalization",
                                        colorVar = "p7", # color = colorPalette,
                                        shapeVar = "pCR", shape = c(1, 4),
                                        sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                        topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                        topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                        topSamplesJust = c(1, 0), topSamplesCex = 3,
                                        dim = 3:4, returnAnalysis = TRUE)
print(s5_20194_34$plot)

x