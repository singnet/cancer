library(tidyverse)
library(tidyselect)

# get get clinical data for tamoxifen studies
pheno <- read_csv("data/curatedBreastData/bcClinicalTable.csv", guess_max = 4000)
tstudies <- filter(pheno, tamoxifen == 1) %>% pull(series_id) %>% unique()

# drop chemo study but keep letrozole study
tstudies <- tstudies[-6]

tpheno <- filter(pheno, series_id %in% tstudies)

# filter variables with > 20% NA
.2 * 785
# [1] 157

tpheno <- select(tpheno, where(~ sum(is.na(.)) < 157))

# filter out constants
tpheno <- select(tpheno, where(~ length(table(.)) > 1))

# filter out redundant variables
# tpheno <- select(series_id, gpl, patient_ID, )

# add radio
cv <- read_csv("data/curatedBreastData/embedding_vector_state_and_outcome.csv") %>%
  left_join(select(tpheno, patient_ID, radio = radiotherapyClass)) %>%
  filter(patient_ID %in% tpheno$patient_ID) %>%
  filter(!is.na(radio))

write_csv(cv, "data/curatedBreastData/tamoxifenClinData.csv")

# clinVar check
basic <- glm(posOutcome ~ 1 + series_id + ER + radio, family = binomial(link = logit), data = cv)
summary(basic)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.9580  -1.2374   0.7378   0.7378   1.1870  
# 
# Coefficients:
#                   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)         1.3669     0.7355   1.859  0.06309 .  
# series_idGSE1379   -1.6178     0.3546  -4.562 5.07e-06 ***
# series_idGSE17705  -0.5956     0.2777  -2.145  0.03197 *  
# series_idGSE9893   -1.3894     0.4250  -3.269  0.00108 ** 
# ER                  0.3909     0.6945   0.563  0.57347    
# radiotherapyClass   0.3500     0.4022   0.870  0.38421    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 744.21  on 641  degrees of freedom
# Residual deviance: 714.61  on 636  degrees of freedom
# AIC: 726.61
# 
# Number of Fisher Scoring iterations: 4

basicPred <- predict(basic, type = "response") >= 0.5
basicCM <- caret::confusionMatrix(data = as_factor(as.numeric(basicPred)), reference = as_factor(cv$posOutcome))
basicCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction FALSE TRUE
#      FALSE     1    1
#      TRUE    170  470
# 
# Accuracy : 0.7336          
# 95% CI : (0.6976, 0.7675)
# No Information Rate : 0.7336          
# P-Value [Acc > NIR] : 0.5206          
# 
# Kappa : 0.0054          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.005848        
#             Specificity : 0.997877        
#          Pos Pred Value : 0.500000        
#          Neg Pred Value : 0.734375        
#              Prevalence : 0.266355        
#          Detection Rate : 0.001558        
#    Detection Prevalence : 0.003115        
#       Balanced Accuracy : 0.501862        
#                                           
#        'Positive' Class : FALSE           
                                          

basicPam <- glm(posOutcome ~ 1 + series_id + ER + radio + pam_coincide, family = binomial(link = logit), data = cv)
summary(basicPam)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -2.1809  -1.2274   0.7227   0.7227   1.2590  
# 
# Coefficients:
#                    Estimate Std. Error z value Pr(>|z|)    
# (Intercept)          1.7924     1.3608   1.317 0.187791    
# series_idGSE1379    -1.7324     0.3623  -4.782 1.74e-06 ***
# series_idGSE17705   -0.6400     0.2795  -2.290 0.022045 *  
# series_idGSE9893    -1.6474     0.4798  -3.434 0.000595 ***
# ER                   0.3749     0.7060   0.531 0.595385    
# radiotherapyClass    0.3927     0.4097   0.959 0.337758    
# pam_coincideHer2     0.1129     1.1699   0.097 0.923098    
# pam_coincideLumA    -0.3180     1.1340  -0.280 0.779165    
# pam_coincideLumB    -0.7096     1.1453  -0.620 0.535535    
# pam_coincideNormal   0.7534     1.5569   0.484 0.628430    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 744.21  on 641  degrees of freedom
# Residual deviance: 707.05  on 632  degrees of freedom
# AIC: 727.05
# 
# Number of Fisher Scoring iterations: 4

basicPamPred <- predict(basicPam, type = "response") >= 0.5
basicPamCM <- caret::confusionMatrix(data = as_factor(as.numeric(basicPamPred)), reference = as_factor(cv$posOutcome))
basicPamCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
#          0   7   5
#          1 164 466
# 
#            Accuracy : 0.7368          
#              95% CI : (0.7009, 0.7705)
# No Information Rate : 0.7336          
# P-Value [Acc > NIR] : 0.4494          
# 
# Kappa : 0.0431          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.04094         
#             Specificity : 0.98938         
#          Pos Pred Value : 0.58333         
#          Neg Pred Value : 0.73968         
#              Prevalence : 0.26636         
#          Detection Rate : 0.01090         
#    Detection Prevalence : 0.01869         
#       Balanced Accuracy : 0.51516         
#                                           
#        'Positive' Class : 0               


basicP5 <- glm(posOutcome ~ 1 + series_id + ER + radio + p5, family = binomial(link = logit), data = cv)
summary(basicP5)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -2.0705  -1.1111   0.6413   0.8611   1.3115  
# 
# Coefficients:
#                   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)         1.1516     0.7683   1.499  0.13392    
# series_idGSE1379   -1.5570     0.3602  -4.323 1.54e-05 ***
# series_idGSE17705  -0.5418     0.2820  -1.921  0.05473 .  
# series_idGSE9893   -1.4050     0.4441  -3.164  0.00156 ** 
# ER                  0.4835     0.6971   0.694  0.48787    
# radio               0.3344     0.4056   0.824  0.40968    
# p5k2               -0.3880     0.3080  -1.260  0.20767    
# p5k3                0.2367     0.3021   0.784  0.43321    
# p5k4                0.3838     0.3124   1.228  0.21930    
# p5k5                0.1585     0.3572   0.444  0.65730    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 744.21  on 641  degrees of freedom
# Residual deviance: 706.11  on 632  degrees of freedom
# AIC: 726.11
# 
# Number of Fisher Scoring iterations: 4

basicP5Pred <- predict(basicP5, type = "response") >= 0.5
basicP5CM <- caret::confusionMatrix(data = as_factor(as.numeric(basicP5Pred)), reference = as_factor(cv$posOutcome))
basicP5CM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
#          0  13   8
#          1 158 463
# 
#            Accuracy : 0.7414          
#              95% CI : (0.7057, 0.7749)
# No Information Rate : 0.7336          
# P-Value [Acc > NIR] : 0.3461          
# 
# Kappa : 0.0819          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.07602         
#             Specificity : 0.98301         
#          Pos Pred Value : 0.61905         
#          Neg Pred Value : 0.74557         
#              Prevalence : 0.26636         
#          Detection Rate : 0.02025         
#    Detection Prevalence : 0.03271         
#       Balanced Accuracy : 0.52952         
#                                           
#        'Positive' Class : 0               
 
basicBoth <- glm(posOutcome ~ 1 + series_id + ER + radio + p5 + pam_coincide, family = binomial(link = logit), data = cv)
summary(basicBoth)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -2.3022  -1.0940   0.6264   0.7835   1.3221  
# 
# Coefficients:
#                    Estimate Std. Error z value Pr(>|z|)    
# (Intercept)          1.7352     1.3699   1.267  0.20525    
# series_idGSE1379    -1.6861     0.3681  -4.581 4.63e-06 ***
# series_idGSE17705   -0.5933     0.2840  -2.089  0.03668 *  
# series_idGSE9893    -1.5515     0.4893  -3.171  0.00152 ** 
# ER                   0.4446     0.7071   0.629  0.52951    
# radio                0.3585     0.4126   0.869  0.38491    
# p5k2                -0.3763     0.3103  -1.213  0.22529    
# p5k3                 0.2369     0.3091   0.766  0.44339    
# p5k4                 0.3937     0.3154   1.248  0.21198    
# p5k5                 0.2154     0.3632   0.593  0.55314    
# pam_coincideHer2    -0.2000     1.1787  -0.170  0.86527    
# pam_coincideLumA    -0.4513     1.1322  -0.399  0.69020    
# pam_coincideLumB    -0.8958     1.1474  -0.781  0.43498    
# pam_coincideNormal   0.7749     1.5652   0.495  0.62056    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 744.21  on 641  degrees of freedom
# Residual deviance: 698.90  on 628  degrees of freedom
# AIC: 726.9
# 
# Number of Fisher Scoring iterations: 4

basicBothPred <- predict(basicBoth, type = "response") >= 0.5
basicBothCM <- caret::confusionMatrix(data = as_factor(as.numeric(basicBothPred)), reference = as_factor(cv$posOutcome))
basicBothCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0  17  12
# 1 154 459
# 
#            Accuracy : 0.7414          
#              95% CI : (0.7057, 0.7749)
# No Information Rate : 0.7336          
# P-Value [Acc > NIR] : 0.3461          
# 
# Kappa : 0.1005          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.09942         
#             Specificity : 0.97452         
#          Pos Pred Value : 0.58621         
#          Neg Pred Value : 0.74878         
#              Prevalence : 0.26636         
#          Detection Rate : 0.02648         
#    Detection Prevalence : 0.04517         
#       Balanced Accuracy : 0.53697         
#                                           
#        'Positive' Class : 0               

# add node variable
cv2 <- filter(cv, !is.na(node))
basic2 <- glm(posOutcome ~ 1 + series_id + ER + radio + node, family = binomial(link = logit), data = cv2)
summary(basic2)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.9269  -1.0896   0.5829   0.9222   1.3800  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)         0.2192     0.7582   0.289  0.77251    
# series_idGSE17705   1.0268     0.3201   3.208  0.00133 ** 
#   series_idGSE9893    0.2546     0.4672   0.545  0.58576    
# ER                  0.4405     0.7092   0.621  0.53457    
# radio               0.5397     0.4285   1.259  0.20789    
# node               -1.1243     0.2119  -5.304 1.13e-07 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 584.02  on 483  degrees of freedom
# Residual deviance: 540.55  on 478  degrees of freedom
# AIC: 552.55
# 
# Number of Fisher Scoring iterations: 4

basic2Pred <- predict(basic2, type = "response") >= 0.5
basic2CM <- caret::confusionMatrix(data = as_factor(as.numeric(basic2Pred)), reference = as_factor(cv2$posOutcome))
basic2CM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0  25  18
# 1 116 325
# 
# Accuracy : 0.7231         
# 95% CI : (0.681, 0.7626)
# No Information Rate : 0.7087         
# P-Value [Acc > NIR] : 0.2591         
# 
# Kappa : 0.1569         
# 
# Mcnemar's Test P-Value : <2e-16         
#                                          
#             Sensitivity : 0.17730        
#             Specificity : 0.94752        
#          Pos Pred Value : 0.58140        
#          Neg Pred Value : 0.73696        
#              Prevalence : 0.29132        
#          Detection Rate : 0.05165        
#    Detection Prevalence : 0.08884        
#       Balanced Accuracy : 0.56241        
#                                          
#        'Positive' Class : 0              
 

basic2Pam <- glm(posOutcome ~ 1 + series_id + ER + radio + pam_coincide + node, family = binomial(link = logit), data = cv2)
summary(basic2Pam)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.1671  -1.0606   0.5833   0.9181   1.3929  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)         0.508406   1.385529   0.367  0.71366    
# series_idGSE17705   1.039141   0.326315   3.184  0.00145 ** 
#   series_idGSE9893    0.103013   0.540412   0.191  0.84882    
# ER                  0.431707   0.721097   0.599  0.54939    
# radio               0.560587   0.432585   1.296  0.19501    
# pam_coincideHer2    0.008827   1.193624   0.007  0.99410    
# pam_coincideLumA   -0.294063   1.150911  -0.256  0.79833    
# pam_coincideLumB   -0.429321   1.169953  -0.367  0.71365    
# pam_coincideNormal  0.268467   1.583951   0.169  0.86541    
# node               -1.107405   0.212586  -5.209  1.9e-07 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 584.02  on 483  degrees of freedom
# Residual deviance: 538.88  on 474  degrees of freedom
# AIC: 558.88
# 
# Number of Fisher Scoring iterations: 4

basic2PamPred <- predict(basic2Pam, type = "response") >= 0.5
basic2PamCM <- caret::confusionMatrix(data = as_factor(as.numeric(basic2PamPred)), reference = as_factor(cv2$posOutcome))
basic2PamCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
#          0  24  17
#          1 117 326
# 
#            Accuracy : 0.7231         
#              95% CI : (0.681, 0.7626)
# No Information Rate : 0.7087         
# P-Value [Acc > NIR] : 0.2591         
# 
#               Kappa : 0.1525         
# 
# Mcnemar's Test P-Value : <2e-16         
#                                          
#             Sensitivity : 0.17021        
#             Specificity : 0.95044        
#          Pos Pred Value : 0.58537        
#          Neg Pred Value : 0.73589        
#              Prevalence : 0.29132        
#          Detection Rate : 0.04959        
#    Detection Prevalence : 0.08471        
#       Balanced Accuracy : 0.56033        
#                                          
#        'Positive' Class : 0              


basic2P5 <- glm(posOutcome ~ 1 + series_id + ER + radio + p5 + node, family = binomial(link = logit), data = cv2)
summary(basic2P5)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.0552  -1.1189   0.5856   0.8220   1.5639  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)        0.12096    0.79774   0.152  0.87948    
# series_idGSE17705  1.02089    0.32077   3.183  0.00146 ** 
# series_idGSE9893   0.22075    0.47918   0.461  0.64502    
# ER                 0.53457    0.71024   0.753  0.45165    
# radio              0.49843    0.43505   1.146  0.25192    
# p5k2              -0.45833    0.35385  -1.295  0.19523    
# p5k3               0.09209    0.34296   0.269  0.78830    
# p5k4               0.30653    0.37259   0.823  0.41067    
# p5k5               0.08535    0.41812   0.204  0.83826    
# node              -1.07145    0.21585  -4.964 6.91e-07 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 584.02  on 483  degrees of freedom
# Residual deviance: 534.80  on 474  degrees of freedom
# AIC: 554.8
# 
# Number of Fisher Scoring iterations: 4

basic2P5Pred <- predict(basic2P5, type = "response") >= 0.5
basic2P5CM <- caret::confusionMatrix(data = as_factor(as.numeric(basic2P5Pred)), reference = as_factor(cv2$posOutcome))
basic2P5CM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
#          0  29  21
#          1 112 322
# 
#            Accuracy : 0.7252          
#              95% CI : (0.6831, 0.7645)
# No Information Rate : 0.7087          
# P-Value [Acc > NIR] : 0.2275          
# 
# Kappa : 0.1783          
# 
# Mcnemar's Test P-Value : 5.998e-15       
#                                           
#             Sensitivity : 0.20567         
#             Specificity : 0.93878         
#          Pos Pred Value : 0.58000         
#          Neg Pred Value : 0.74194         
#              Prevalence : 0.29132         
#          Detection Rate : 0.05992         
#    Detection Prevalence : 0.10331         
#       Balanced Accuracy : 0.57222         
#                                           
#        'Positive' Class : 0               

basic2Both <- glm(posOutcome ~ 1 + series_id + ER + radio + p5 + pam_coincide + node, family = binomial(link = logit), data = cv2)
summary(basic2Both)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -2.2754  -1.1001   0.5841   0.8244   1.5620  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)         0.49907    1.38345   0.361  0.71829    
# series_idGSE17705   1.03711    0.32742   3.168  0.00154 ** 
# series_idGSE9893    0.17737    0.54678   0.324  0.74564    
# ER                  0.51040    0.71968   0.709  0.47820    
# radio               0.50797    0.43820   1.159  0.24637    
# p5k2               -0.45485    0.35580  -1.278  0.20111    
# p5k3                0.08572    0.35066   0.244  0.80688    
# p5k4                0.31133    0.37658   0.827  0.40838    
# p5k5                0.10122    0.42290   0.239  0.81083    
# pam_coincideHer2   -0.22192    1.20327  -0.184  0.85367    
# pam_coincideLumA   -0.36457    1.14720  -0.318  0.75064    
# pam_coincideLumB   -0.54823    1.17042  -0.468  0.63949    
# pam_coincideNormal  0.36291    1.59733   0.227  0.82027    
# node               -1.05996    0.21625  -4.902  9.5e-07 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 584.02  on 483  degrees of freedom
# Residual deviance: 533.37  on 470  degrees of freedom
# AIC: 561.37
# 
# Number of Fisher Scoring iterations: 4

basic2BothPred <- predict(basic2Both, type = "response") >= 0.5
basic2BothCM <- caret::confusionMatrix(data = as_factor(as.numeric(basic2BothPred)),
                                       reference = as_factor(cv2$posOutcome))
basic2BothCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
#          0  32  25
#          1 109 318
# 
#            Accuracy : 0.7231         
#              95% CI : (0.681, 0.7626)
# No Information Rate : 0.7087         
# P-Value [Acc > NIR] : 0.2591         
# 
# Kappa : 0.1868         
# 
# Mcnemar's Test P-Value : 7.494e-13      
#                                          
#             Sensitivity : 0.22695        
#             Specificity : 0.92711        
#          Pos Pred Value : 0.56140        
#          Neg Pred Value : 0.74473        
#              Prevalence : 0.29132        
#          Detection Rate : 0.06612        
#    Detection Prevalence : 0.11777        
#       Balanced Accuracy : 0.57703        
#                                          
#        'Positive' Class : 0              

# make eset for combat reference
expData <- read_csv("data/curatedBreastData/microarray data/merged-combat15.csv.xz")
tamExp <- filter(expData, patient_ID %in% cv$patient_ID) %>%
  arrange(patient_ID) 

# save data for asmoses validation
left_join(select(cv, patient_ID, RFS, DFS, posOutcome), mutate(tamExp, patient_ID = as.numeric(patient_ID))) %>%
  write_csv("ml/asmoses/tamox4expSet.csv")

tamExp <- column_to_rownames(tamExp, "patient_ID")

ebExp <- filter(expData, patient_ID %in% tpheno$patient_ID) %>%
  arrange(patient_ID) %>%
  column_to_rownames("patient_ID")
tpheno <- filter(tpheno, patient_ID %in% rownames(ebExp))

library(Biobase)
tamCom <- ExpressionSet(t(tamExp), AnnotatedDataFrame(column_to_rownames(arrange(cv, patient_ID), "patient_ID")))
ebCom <- ExpressionSet(t(ebExp), AnnotatedDataFrame(column_to_rownames(arrange(tpheno, patient_ID), "patient_ID")))

saveRDS(ebCom, "data/curatedBreastData/tamoxifenEset.rds")

cbc <- readRDS("data/curatedBreastData/bcProcessedEsetList.rds")
tamox <- set_names(cbc, paste0("GSE", str_extract(names(cbc), "_[0123456789]+_") %>% str_remove_all("_")))[tstudies]
rm(cbc)

# plot exported from rstudio
map2(tamox, names(tamox), ~ broman::manyboxplot(Biobase::exprs(.x), main = .y))

# make validation set from tamoxifen positive metagx samples (GSE58644)
mgxtPheno <- read_csv("data/metaGxBreast/metaGXcovarTable.csv.xz", guess_max = 5000) %>%
filter(tamoxifen == 1) %>%
arrange(sample_name) %>%
dplyr::select(patient_ID = sample_name, series_id = study, gpl, channel_count,
              RFS = dmfs_status, ER = er, HER2 = her2, PR = pgr, node = N, tumor = `T`, pam50)

# make expression matrix
gse58644no <- read_csv("data/metaGxBreast/microarray data/noNorm/GSE58644_noNorm.csv.xz") %>%
  filter(sample_name %in% mgxtPheno$patient_ID)
gse58644genes <- names(gse58644no)[-1]

# find missing genes
missingCBC <- setdiff(names(expData)[-1], names(gse58644no))
length(missingCBC)
# [1] 504

symbolMap <- read_csv("data/curatedBreastData/hgncSymbolMap21aug20long.csv")
cbcMap <- tibble(cbc = missingCBC, ref = symbolMap$`Approved symbol`[match(missingCBC, symbolMap$alt)])
mgxMap <- tibble(mgx = gse58644genes, ref = symbolMap$`Approved symbol`[match(gse58644genes, symbolMap$alt)])
missingMap <- left_join(cbcMap, mgxMap)
noMatch <- filter(missingMap, is.na(ref)) %>%
  pull(cbc) %>%
  unique()

missingMap <- filter(missingMap, !is.na(ref))
matches <- match(missingMap$mgx, gse58644genes)
missingMatches <- is.na(matches)
gse58644genes[matches[!missingMatches]] <- missingMap$cbc[!missingMatches]
length(setdiff(names(expData), gse58644genes))
# [1] 109

names(gse58644no)[-1] <- gse58644genes
gse58644no <- dplyr::select(gse58644no, sample_name, intersect(names(gse58644no), names(expData))) %>%
  arrange(sample_name) %>%
  column_to_rownames("sample_name")

# normalize gse58644 to tamox set with combat
tamox5 <- bind_rows(gse58644no, dplyr::select(tamExp, names(gse58644no))) %>%
  t()
batch <- c(rep(1, dim(gse58644no)[1]), rep(2, dim(tamExp)[1]))

tamox5com <- sva::ComBat(tamox5, batch, prior.plots = TRUE, ref.batch = 2)

summary(colMeans(tamox5[, 1:154] - tamox5com[, 1:154]))
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 2.403   2.458   2.484   2.488   2.507   2.638 

summary(colMeans(tamox5com[, 155:796] - tamox5com[, 155:796]))
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#    0       0       0       0       0       0 

# plot exported from rstudio
map2(list(tamox5, tamox5com), c("GSE58644 and tamoxifen set", "GSE58644 normalized\ntamoxifen set reference"),
     ~ broman::manyboxplot(.x, main = .y))

write_csv(bind_cols(patient_ID = colnames(tamox5com)[1:154], t(tamox5com[, 1:154]), ),
          "data/curatedBreastData/gse58644tamoxifenCombat.csv.xz")

# make clincal data set
gse58644Pheno <- read_csv("data/metaGxBreast/metaGXcovarTable.csv.xz", guess_max = 5000) %>%
  filter(sample_name %in% rownames(gse58644no)) %>%
  arrange(sample_name) %>%
  dplyr::select(patient_ID = sample_name, series_id = study, gpl, channel_count, RFS = dmfs_status,
                ER = er, HER2 = her2, PR = pgr, node = N, tumor = `T`, pam_coincide = pam50) %>%
  mutate(RFS = as.numeric(RFS == "norecurrence"), ER = as.numeric(ER == "positive"),
         HER2 = as.numeric(HER2 == "positive"), PR = as.numeric(PR == "positive"), radio = 0, posOutcome = RFS)

write_csv(gse58644Pheno, "data/curatedBreastData/gse58644tamoxifenClinData.csv.xz")

# genes missing from tomoxifen feature set
missingGenes <- c('TSTA3', 'SPN', 'ZNF434', 'C12orf49', 'HIST1H1C', 'KIAA1609', 'ZNF192', 'PHF15', 'HRASLS', 'IMPAD1')

setdiff(missingGenes, names(gse58644no))
# [1] "ZNF192"

basicVal <- glm(posOutcome ~ 1 + ER + radio, family = binomial(link = logit), data = cv)
summary(basicVal)
# Deviance Residuals: 
#   Min      1Q  Median      3Q     Max  
# -1.662  -1.500   0.761   0.761   1.117  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)
# (Intercept)   0.5041     0.6961   0.724    0.469
# ER            0.5869     0.6921   0.848    0.396
# radio        -0.3592     0.2274  -1.580    0.114
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 744.21  on 641  degrees of freedom
# Residual deviance: 740.44  on 639  degrees of freedom
# AIC: 746.44
# 
# Number of Fisher Scoring iterations: 4

basicValPred <- predict(basicVal, type = "response") >= 0.5
basicValCM <- caret::confusionMatrix(data = as_factor(as.numeric(basicPred)), reference = as_factor(cv$posOutcome))
basicValCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction FALSE TRUE
#      FALSE     1    1
#      TRUE    170  470
# 
# Accuracy : 0.7336          
# 95% CI : (0.6976, 0.7675)
# No Information Rate : 0.7336          
# P-Value [Acc > NIR] : 0.5206          
# 
# Kappa : 0.0054          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.005848        
#             Specificity : 0.997877        
#          Pos Pred Value : 0.500000        
#          Neg Pred Value : 0.734375        
#              Prevalence : 0.266355        
#          Detection Rate : 0.001558        
#    Detection Prevalence : 0.003115        
#       Balanced Accuracy : 0.501862        
#                                           
#        'Positive' Class : FALSE           

basicTestPred <- predict(basicVal, newdata = gse58644Pheno, type = "response") >= 0.5
basicTestCM <- caret::confusionMatrix(data = as_factor(as.numeric(basicTestPred)),
                                       reference = as_factor(gse58644Pheno$posOutcome))
basicTestCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
#          0   0   0
#          1  37 113
# 
#            Accuracy : 0.7533        
#              95% CI : (0.6764, 0.82)
# No Information Rate : 0.7533        
# P-Value [Acc > NIR] : 0.544         
# 
# Kappa : 0             
# 
# Mcnemar's Test P-Value : 3.252e-09     
#                                         
#             Sensitivity : 0.0000        
#             Specificity : 1.0000        
#          Pos Pred Value :    NaN        
#          Neg Pred Value : 0.7533        
#              Prevalence : 0.2467        
#          Detection Rate : 0.0000        
#    Detection Prevalence : 0.0000        
#       Balanced Accuracy : 0.5000        
#                                         
#        'Positive' Class : 0             

basicValPam <- glm(posOutcome ~ 1 + ER + radio + pam_coincide, family = binomial(link = logit), data = cv)
summary(basicValPam)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.1166  -1.5063   0.7495   0.7495   1.2283  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)
# (Intercept)          0.9068     1.3262   0.684    0.494
# ER                   0.5623     0.6987   0.805    0.421
# radio               -0.3794     0.2798  -1.356    0.175
# pam_coincideHer2    -0.2353     1.1542  -0.204    0.838
# pam_coincideLumA    -0.3431     1.1272  -0.304    0.761
# pam_coincideLumB    -0.6463     1.1377  -0.568    0.570
# pam_coincideNormal   0.6583     1.5436   0.426    0.670
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 744.21  on 641  degrees of freedom
# Residual deviance: 736.83  on 635  degrees of freedom
# AIC: 750.83
# 
# Number of Fisher Scoring iterations: 4

basicValPamPred <- predict(basicValPam, type = "response") >= 0.5
basicValPamCM <- caret::confusionMatrix(data = as_factor(as.numeric(basicValPamPred)), reference = as_factor(cv$posOutcome))
basicValPamCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
#          0   2   1
#          1 169 470
# 
#                Accuracy : 0.7352         
#                  95% CI : (0.6993, 0.769)
#     No Information Rate : 0.7336         
#     P-Value [Acc > NIR] : 0.485          
# 
#                   Kappa : 0.0139         
# 
#  Mcnemar's Test P-Value : <2e-16         
#                                          
#             Sensitivity : 0.011696       
#             Specificity : 0.997877       
#          Pos Pred Value : 0.666667       
#          Neg Pred Value : 0.735524       
#              Prevalence : 0.266355       
#          Detection Rate : 0.003115       
#    Detection Prevalence : 0.004673       
#       Balanced Accuracy : 0.504786       
#                                          
#        'Positive' Class : 0              

basicTestPamPred <- predict(basicValPam, newdata = gse58644Pheno, type = "response") >= 0.5
basicTestPamCM <- caret::confusionMatrix(data = as_factor(as.numeric(basicTestPamPred)),
                                      reference = as_factor(gse58644Pheno$posOutcome))
basicTestPamCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0   0   0
# 1  37 113
# 
# Accuracy : 0.7533        
# 95% CI : (0.6764, 0.82)
# No Information Rate : 0.7533        
# P-Value [Acc > NIR] : 0.544         
# 
# Kappa : 0             
# 
# Mcnemar's Test P-Value : 3.252e-09     
#                                         
#             Sensitivity : 0.0000        
#             Specificity : 1.0000        
#          Pos Pred Value :    NaN        
#          Neg Pred Value : 0.7533        
#              Prevalence : 0.2467        
#          Detection Rate : 0.0000        
#    Detection Prevalence : 0.0000        
#       Balanced Accuracy : 0.5000        
#                                         
#        'Positive' Class : 0             

# basicTestP5 <- glm(posOutcome ~ 1 + ER + radio + p5, family = binomial(link = logit), data = cv)
# summary(basicTestP5)
# 
# basicTestP5Pred <- predict(basicTestP5, type = "response") >= 0.5
# basicTestP5CM <- caret::confusionMatrix(data = as_factor(as.numeric(basicTestP5Pred)), reference = as_factor(cv$posOutcome))
# basicTestP5CM
# 
# basicTestBoth <- glm(posOutcome ~ 1 + ER + radio + p5 + pam_coincide, family = binomial(link = logit), data = cv)
# summary(basicTestBoth)
# 
# basicTestBothPred <- predict(basicTestBoth, type = "response") >= 0.5
# basicTestBothCM <- caret::confusionMatrix(data = as_factor(as.numeric(basicTestBothPred)), reference = as_factor(cv$posOutcome))
# basicTestBothCM
# 
# basicTestBothPred <- predict(basicValBoth, newdata = gse58644Pheno, type = "response") >= 0.5
# basicTestBothCM <- caret::confusionMatrix(data = as_factor(as.numeric(basicTestBothPred)),
#                                       reference = as_factor(gse58644Pheno$posOutcome))
# basicTestBothCM
# no infogan variable

basic2val <- glm(posOutcome ~ 1 + ER + radio + node, family = binomial(link = logit), data = cv2)
summary(basic2val)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.8292  -1.3153   0.6513   0.8374   1.2857  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)  0.85070    0.73073   1.164    0.244    
# ER           0.59196    0.71768   0.825    0.409    
# radio        0.02249    0.24882   0.090    0.928    
# node        -1.12429    0.20872  -5.387 7.18e-08 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 584.02  on 483  degrees of freedom
# Residual deviance: 552.75  on 480  degrees of freedom
# AIC: 560.75
# 
# Number of Fisher Scoring iterations: 4

basic2valPred <- predict(basic2val, type = "response") >= 0.5
basic2valCM <- caret::confusionMatrix(data = as_factor(as.numeric(basic2valPred)), reference = as_factor(cv2$posOutcome))
basic2valCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0   3   2
# 1 138 341
# 
# Accuracy : 0.7107          
# 95% CI : (0.6681, 0.7508)
# No Information Rate : 0.7087          
# P-Value [Acc > NIR] : 0.4828          
# 
# Kappa : 0.0216          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.021277        
#             Specificity : 0.994169        
#          Pos Pred Value : 0.600000        
#          Neg Pred Value : 0.711900        
#              Prevalence : 0.291322        
#          Detection Rate : 0.006198        
#    Detection Prevalence : 0.010331        
#       Balanced Accuracy : 0.507723        
#                                           
#        'Positive' Class : 0               

basic2testPred <- predict(basic2val, newdata = gse58644Pheno, type = "response") >= 0.5
basic2testCM <- caret::confusionMatrix(data = as_factor(as.numeric(basic2testPred)), reference = as_factor(gse58644Pheno$posOutcome))
basic2testCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
#          0   0   2
#          1  33 108
# 
# Accuracy : 0.7552          
# 95% CI : (0.6764, 0.8232)
# No Information Rate : 0.7692          
# P-Value [Acc > NIR] : 0.6948          
# 
# Kappa : -0.0271         
# 
# Mcnemar's Test P-Value : 3.959e-07       
#                                           
#             Sensitivity : 0.00000         
#             Specificity : 0.98182         
#          Pos Pred Value : 0.00000         
#          Neg Pred Value : 0.76596         
#              Prevalence : 0.23077         
#          Detection Rate : 0.00000         
#    Detection Prevalence : 0.01399         
#       Balanced Accuracy : 0.49091         
#                                           
#        'Positive' Class : 0               

basic2valPam <- glm(posOutcome ~ 1  + ER + radio + pam_coincide + node, family = binomial(link = logit), data = cv2)
summary(basic2valPam)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -2.1488  -1.3150   0.6538   0.7761   1.3237  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)         1.34545    1.36405   0.986    0.324    
# ER                  0.57566    0.72348   0.796    0.426    
# radio               0.01083    0.30269   0.036    0.971    
# pam_coincideHer2   -0.43333    1.18595  -0.365    0.715    
# pam_coincideLumA   -0.48694    1.15447  -0.422    0.673    
# pam_coincideLumB   -0.57743    1.17390  -0.492    0.623    
# pam_coincideNormal  0.28300    1.59046   0.178    0.859    
# node               -1.11629    0.20928  -5.334  9.6e-08 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 584.02  on 483  degrees of freedom
# Residual deviance: 551.80  on 476  degrees of freedom
# AIC: 567.8
# 
# Number of Fisher Scoring iterations: 4

basic2valPamPred <- predict(basic2valPam, type = "response") >= 0.5
basic2valPamCM <- caret::confusionMatrix(data = as_factor(as.numeric(basic2valPamPred)), reference = as_factor(cv2$posOutcome))
basic2valPamCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
#          0   3   2
#          1 138 341
# 
#               Accuracy : 0.7107          
#                 95% CI : (0.6681, 0.7508)
#    No Information Rate : 0.7087          
#    P-Value [Acc > NIR] : 0.4828          
# 
#                  Kappa : 0.0216          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.021277        
#             Specificity : 0.994169        
#          Pos Pred Value : 0.600000        
#          Neg Pred Value : 0.711900        
#              Prevalence : 0.291322        
#          Detection Rate : 0.006198        
#    Detection Prevalence : 0.010331        
#       Balanced Accuracy : 0.507723        
#                                           
#        'Positive' Class : 0               

basic2testPam <- glm(posOutcome ~ 1 + ER + radio + pam_coincide + node, family = binomial(link = logit), data = cv2)
summary(basic2testPam)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.1488  -1.3150   0.6538   0.7761   1.3237  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)         1.34545    1.36405   0.986    0.324    
# ER                  0.57566    0.72348   0.796    0.426    
# radio               0.01083    0.30269   0.036    0.971    
# pam_coincideHer2   -0.43333    1.18595  -0.365    0.715    
# pam_coincideLumA   -0.48694    1.15447  -0.422    0.673    
# pam_coincideLumB   -0.57743    1.17390  -0.492    0.623    
# pam_coincideNormal  0.28300    1.59046   0.178    0.859    
# node               -1.11629    0.20928  -5.334  9.6e-08 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 584.02  on 483  degrees of freedom
# Residual deviance: 551.80  on 476  degrees of freedom
# AIC: 567.8
# 
# Number of Fisher Scoring iterations: 4

basic2testPamPred <- predict(basic2valPam, newdata = gse58644Pheno, type = "response") >= 0.5
basic2testPamCM <- caret::confusionMatrix(data = as_factor(as.numeric(basic2testPamPred)), reference = as_factor(gse58644Pheno$posOutcome))
basic2testPamCM
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0   0   0
# 1  33 110
# 
# Accuracy : 0.7692          
# 95% CI : (0.6915, 0.8355)
# No Information Rate : 0.7692          
# P-Value [Acc > NIR] : 0.5465          
# 
# Kappa : 0               
# 
# Mcnemar's Test P-Value : 2.54e-08        
#                                           
#             Sensitivity : 0.0000          
#             Specificity : 1.0000          
#          Pos Pred Value :    NaN          
#          Neg Pred Value : 0.7692          
#              Prevalence : 0.2308          
#          Detection Rate : 0.0000          
#    Detection Prevalence : 0.0000          
#       Balanced Accuracy : 0.5000          
#                                           
#        'Positive' Class : 0               

# get imputed data
impKnn <- read_csv("ml/embedding vectors/imputation_results/knn_imputed_tamox.csv") %>%
  mutate(across(c(patient_ID, gpl, pam_coincide, p5), as.character))
impMfi <- read_csv("ml/embedding vectors/imputation_results/MFI_imputed_tamox.csv") %>%
  mutate(across(c(patient_ID, gpl, pam_coincide, p5), as.character))

# check tumor inputation - no difference but how to interpret "-1" ?
table(impKnn$tumor)
#  -1   0   1   2   3 
# 434  88 114   4   2 

table(impMfi$tumor)
#  -1   0   1   2   3 
# 434  88 114   4   2 

# how different are they?
impDiff <- impKnn[, 3:7] - impMfi[, 3:7]
summary(impDiff)
#       ER         HER2                PR              node            tumor  
# Min.   :0   Min.   :0.000000   Min.   :0.0000   Min.   :0.0000   Min.   :0  
# 1st Qu.:0   1st Qu.:0.000000   1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.:0  
# Median :0   Median :0.000000   Median :0.7778   Median :0.0000   Median :0  
# Mean   :0   Mean   :0.007615   Mean   :0.5189   Mean   :0.1265   Mean   :0  
# 3rd Qu.:0   3rd Qu.:0.000000   3rd Qu.:0.7778   3rd Qu.:0.0000   3rd Qu.:0  
# Max.   :0   Max.   :0.333333   Max.   :1.0000   Max.   :1.0000   Max.   :0  

table(impKnn$HER2)
#   0 0.111111111111111 0.222222222222222 0.333333333333333                 1 
# 608                20                 6                 4                 4 

table(impMfi$HER2)
#   0   1 
# 638   4 

table(impKnn$PR)
#  0 0.666666666666667 0.777777777777778 0.888888888888889                 1 
# 38                68               341                22               173 

table(impMfi$PR)
#   0   1 
# 472 170 

lrKnn <- glmulti::glmulti(posOutcome ~ ER + HER2 + PR + pam_coincide + p5, family = binomial(link = logit), data = impKnn[, 1:10], crit = "bic", level = 1)
print(lrKnn)
# Method: h / Fitting: glm / IC used: bic
# Level: 1 / Marginality: FALSE
# From 32 models:
#   Best IC: 748.029449591274
# Best model:
#   [1] "posOutcome ~ 1 + PR"
# Evidence weight: 0.66241027648207
# Worst IC: 797.504901575045
# 1 models within 2 IC units.
# 3 models to reach 95% of evidence weight.

glmAll <- function(obj) tibble(formula = obj@formulas, ic = obj@crits, K = obj@K) %>%
  mutate(formula_string = map(formula, ~ str_c(str_trim(deparse(.)), collapse = " "))) %>%
  unnest(formula_string)

lrKnnSum <- glmAll(lrKnn) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = impKnn[, 1:10]))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$posOutcome == "1")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

select(lrKnnSum, -4:-8) %>%
  filter(Accuracy == max(Accuracy))
#     formula      ic     K Accuracy  Kappa AccuracyLower AccuracyUpper AccuracyNull AccuracyPValue McnemarPValue
#   1 <formula>  787.    11    0.740 0.0788         0.704         0.773        0.734          0.380      2.28e-30
filter(lrKnnSum, Accuracy == max(Accuracy)) %>%
  pull(formula)
# [[1]]
# posOutcome ~ 1 + ER + PR + pam_coincide + p5
# <environment: 0x556c10f60df0>
  

lrKnnNode <- glmulti::glmulti(posOutcome ~ ER + HER2 + PR + pam_coincide + p5 + node, family = binomial(link = logit), data = impKnn[, 1:10], crit = "bic", level = 1)

lrKnnNodeSum <- glmAll(lrKnnNode) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = impKnn[, 1:10]))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$posOutcome == "1")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

select(lrKnnNodeSum, -4:-8) %>%
  filter(Accuracy == max(Accuracy))
#   formula      ic     K Accuracy Kappa AccuracyLower AccuracyUpper AccuracyNull AccuracyPValue McnemarPValue
# 1 <formula>  730.     4    0.757 0.163         0.722         0.790        0.734         0.0969      5.40e-28
# 2 <formula>  736.     5    0.757 0.163         0.722         0.790        0.734         0.0969      5.40e-28
# 3 <formula>  753.     8    0.757 0.163         0.722         0.790        0.734         0.0969      5.40e-28

filter(lrKnnNodeSum, Accuracy == max(Accuracy)) %>%
  pull(formula)
# [[1]]
# posOutcome ~ 1 + ER + PR + node
# <environment: 0x556be7502278>
#  
# [[2]]
# posOutcome ~ 1 + ER + HER2 + PR + node
# <environment: 0x556be7502278>
#   
# [[3]]
# posOutcome ~ 1 + ER + PR + pam_coincide + node
# <environment: 0x556be7502278>
  
lrKnnNT <- glmulti::glmulti(posOutcome ~ ER + HER2 + PR + pam_coincide + p5 + node + tumor + node:tumor, family = binomial(link = logit), data = impKnn[, 1:10], crit = "bic", level = 1)
lrKnnNTSum <- glmAll(lrKnnNT) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = impKnn[, 1:10]))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$posOutcome == "1")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

select(lrKnnNTSum, -4:-8) %>%
  filter(Accuracy == max(Accuracy))
#   formula      ic     K Accuracy Kappa AccuracyLower AccuracyUpper AccuracyNull AccuracyPValue McnemarPValue
# 1 <formula>  735.     9    0.762 0.231         0.727         0.794        0.734         0.0578      5.95e-19
# 2 <formula>  742.    10    0.762 0.235         0.727         0.794        0.734         0.0578      2.52e-18
# 3 <formula>  748.     8    0.762 0.228         0.727         0.794        0.734         0.0578      1.37e-19
# 4 <formula>  754.    13    0.762 0.231         0.727         0.794        0.734         0.0578      5.95e-19
# 5 <formula>  755.     9    0.762 0.228         0.727         0.794        0.734         0.0578      1.37e-19

# filter(lrKnnNTSum, Accuracy == max(Accuracy)) %>%
#   pull(formula)
# [[1]]
# posOutcome ~ 1 + HER2 + PR + pam_coincide + node + tumor
# <environment: 0x556be4f92160>
#   
# [[2]]
# posOutcome ~ 1 + ER + HER2 + PR + pam_coincide + node + tumor
# <environment: 0x556be4f92160>
#   
# [[3]]
# posOutcome ~ 1 + ER + PR + p5 + tumor
# <environment: 0x556be4f92160>
#   
# [[4]]
# posOutcome ~ 1 + ER + PR + pam_coincide + p5 + node + tumor
# <environment: 0x556be4f92160>
#   
# [[5]]
# posOutcome ~ 1 + ER + HER2 + PR + p5 + tumor
# <environment: 0x556be4f92160>
  

lrKnnGpl <- glmulti::glmulti(posOutcome ~ gpl + ER + HER2 + PR + pam_coincide + p5, family = binomial(link = logit), data = impKnn[, 1:10], crit = "bic", level = 1)
lrKnnGplSum <- glmAll(lrKnnGpl) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = impKnn[, 1:10]))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$posOutcome == "1")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

select(lrKnnGplSum, -4:-8) %>%
  filter(Accuracy == max(Accuracy))
#   formula      ic     K Accuracy Kappa AccuracyLower AccuracyUpper AccuracyNull AccuracyPValue McnemarPValue
# 1 <formula>  735.     9    0.762 0.231         0.727         0.794        0.734         0.0578      5.95e-19
# 2 <formula>  742.    10    0.762 0.235         0.727         0.794        0.734         0.0578      2.52e-18

filter(lrKnnGplSum, Accuracy == max(Accuracy)) %>%
  pull(formula)
# [[1]]
# posOutcome ~ 1 + gpl + HER2 + PR
# <environment: 0x556be887f4d0>
#   
# [[2]]
# posOutcome ~ 1 + gpl + ER + PR + pam_coincide
# <environment: 0x556be887f4d0>
#   
# [[3]]
# posOutcome ~ 1 + gpl + ER + HER2 + PR + pam_coincide
# <environment: 0x556be887f4d0>
  
lrKnnGplNode <- glmulti::glmulti(posOutcome ~ gpl + ER + HER2 + PR + pam_coincide + p5 + node, family = binomial(link = logit), data = impKnn[, 1:10], crit = "bic", level = 1)
lrKnnGplNodeSum <- glmAll(lrKnnGplNode) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = impKnn[, 1:10]))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$posOutcome == "1")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

select(lrKnnGplNodeSum, -4:-8) %>%
  filter(Accuracy == max(Accuracy))
#   formula      ic     K Accuracy Kappa AccuracyLower AccuracyUpper AccuracyNull AccuracyPValue McnemarPValue
# 1 <formula>  730.     4    0.757 0.163         0.722         0.790        0.734         0.0969      5.40e-28
# 2 <formula>  736.     5    0.757 0.163         0.722         0.790        0.734         0.0969      5.40e-28
# 3 <formula>  753.     8    0.757 0.163         0.722         0.790        0.734         0.0969      5.40e-28

filter(lrKnnGplSum, Accuracy == max(Accuracy)) %>%
  pull(formula)
# [[1]]
# posOutcome ~ 1 + gpl + HER2 + PR
# <environment: 0x556be887f4d0>
#   
# [[2]]
# posOutcome ~ 1 + gpl + ER + PR + pam_coincide
# <environment: 0x556be887f4d0>
#   
# [[3]]
# posOutcome ~ 1 + gpl + ER + HER2 + PR + pam_coincide
# <environment: 0x556be887f4d0>
  
lrKnnGplNT <- glmulti::glmulti(posOutcome ~ gpl + ER + HER2 + PR + pam_coincide + p5 + node + tumor, family = binomial(link = logit), data = impKnn[, 1:10], crit = "bic", level = 1)
lrKnnGplNTSum <- glmAll(lrKnnGplNT) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = impKnn[, 1:10]))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$posOutcome == "1")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

select(lrKnnGplNTSum, -4:-8) %>%
  filter(Accuracy == max(Accuracy))
#   formula      ic     K Accuracy Kappa AccuracyLower AccuracyUpper AccuracyNull AccuracyPValue McnemarPValue
# 1 <formula>  742.     5    0.752 0.162         0.717         0.785        0.734          0.152      6.37e-25
# 2 <formula>  761.     9    0.752 0.166         0.717         0.785        0.734          0.152      3.28e-24
# 3 <formula>  767.    10    0.752 0.166         0.717         0.785        0.734          0.152      3.28e-24

filter(lrKnnGplSum, Accuracy == max(Accuracy)) %>%
  pull(formula)
# [[1]]
# posOutcome ~ 1 + gpl + HER2 + PR
# <environment: 0x556be887f4d0>
#   
# [[2]]
# posOutcome ~ 1 + gpl + ER + PR + pam_coincide
# <environment: 0x556be887f4d0>
#   
# [[3]]
# posOutcome ~ 1 + gpl + ER + HER2 + PR + pam_coincide
# <environment: 0x556be887f4d0>


