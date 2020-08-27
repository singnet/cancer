## logistic regression models
library(tidyverse)
library(glmulti)
pheno <- read_csv("data/metaGxBreast/metaGXcovarTable.csv.xz", guess_max = 4000)
dim(pheno)
# [1] 9717   31

#samplename: Unique identifier for tumor samples
#dataset: Acronym of the dataset
#series: Batch or subcohorts of tumor samples
#id: Identifier used in the original publication
#age: Age at diagnosis (years)
#er: Estrogen receptor status
#pgr: Progesterone receptor status
#her2: Human epidermal growth factor 2 status
#grade: Histological grade
#node: Nodal status
#size: Tumor size (cm)
#t.rfs: Time for relapse-free survival (days)
#e.rfs: Event for relapse-free survival
#t.dmfs: Time for distant metastasis-free survival (days)
#e.dmfs: Event for distant metastasis-free survival
#t.os: Time for overall survival (days)
#e.os: Event for overall survival
#treatment: Treatment (0=untreated; 1=treated)
#MAMMAPRINT: Risk classification (Low/HighRisk) computed using the published algorithm of the prognostic gene signature published by van't Veer et al. (13)
#ONCOTYPE: Risk classification (Low/Intermediate/HighRisk) computed using the published algorithm of the prognostic gene signature published by Paik et al. (15)
#GGI: Risk classification (Low/HighRisk) computed using the published algorithm of the prognostic gene signature published by Sotiriou et al. (16)
#SCMGENE: Subtype classification published in the present work.
#SCMOD2: Subtype classification published by Wirapati et al. (8)
#SCMOD1: Subtype classification published by Desmedt et al. (1)
#PAM50: Subtype classification published by Parker et al. (3)
#SSP2006: Subtype classification published by Hu et al. (2)
#SSP2003: Subtype classification published by Sorlie et al. (6)
clusters <- read_csv("data/metaGxBreast/jnci-JNCI-11-0924-s02.csv", comment = "#")
dim(clusters)
# [1] 5715   27

table(clusters$dataset)

# note extra grade level in package data
table(clusters$grade)
#   1    2    3 
# 544 1440 1695 
table(pheno$grade)
#   1    2    3    4 
# 701 2299 2903   15 

covars <- left_join(pheno, clusters, by = c("sample_name" = "samplename"), suffix = c("_mgx", "_3g"))
setdiff(covars$study, covars$dataset)
# [1] "GSE25066" "GSE58644" "GSE32646" "GSE48091" "DUKE"     "EXPO"     "METABRIC" "TCGA"

# extra grade level is from new metagx data sets
table(covars$grade_mgx)
#   1    2    3 
# 379  940 1292 

# three gene data set removes some extra samples
covars <- filter(covars, sample_name %in% clusters$samplename)
dim(covars)
# [1] 4158  57

dim(clusters)[1] - dim(covars)[1]
# 1557

# three studies aren't included in metagx, the rest are from DUKE, EXPO, MAQC2, and STNO2
table(clusters$dataset[!(clusters$samplename %in% covars$sample_name)])
# DUKE      DUKE2 EORTC10994       EXPO        KOO      MAQC2       MDA4       MDA5        MSK 
#  171          6          2        353         53        186          4        298          1 
# PNC      STNO2        TAM       UCSF       VDX3 
#   1        103        242          1        136 

setdiff(clusters$dataset, pheno$study)
# [1] "TAM"  "MDA5" "VDX3"

setdiff(clusters$series, pheno$batch)
# [1] "OXFT"  "KIT"   "IGRT"  "AUST"  "VDX3"  "GUYT"  "GUYT2"

table(clusters$dataset[clusters$series %in% setdiff(clusters$series, pheno$batch)])
# MDA5  TAM VDX3 
# 298  242  136 

dim(clusters)[1] - 298 + 242 + 136
# [1] 5795

dim(clusters)[1] - (298 + 242 + 136)
# [1] 5039

dim(clusters)[1] - dim(covars)[1] - (298 + 242 + 136)
# [1] 881
171 + 353 + 186 + 103
# [1] 813

# generate models on metagx samples
library(glmulti)
table(pheno$sample_type)
# healthy   tumor 
#     133    9584 

table(pheno$replicate)
# FALSE  TRUE 
#  9708     9 

table(pheno$treatment)
# chemo.plus.hormono       chemotherapy     hormonotherapy          untreated 
#                378               1313               1640               2188 

table(pheno$treatment)
pbasic <- filter(pheno, replicate != TRUE) %>%
  select(study, batch, sample_name, age = age_at_initial_pathologic_diagnosis, grade, N, er, pgr, her2, Platform, vital_status, recurrence_status, treatment) %>%
  mutate(chemo = treatment == "chemotherapy" | treatment == "chemo.plus.hormono",
         hormone = treatment == "hormonotherapy" | treatment == "chemo.plus.hormono") %>%
  select(- treatment) %>%
  mutate(across(where( ~ !is.numeric(.)), as.factor))

xtabs(~ chemo + hormone, data = pbasic)
#        hormone
# chemo   FALSE TRUE
#   FALSE  2188 1640
#   TRUE   1304  378

table(pbasic$pgr)
# negative positive 
# 1629     2092 
table(pbasic$her2
# negative positive 
# 2897      803 
table(pbasic$er)
# negative positive 
# 2367     5479

# death model
deathPheno <- glmulti(vital_status ~ study + batch + Platform + age + grade + N + er + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 250 models:
# Best model: vital_status~1+study+batch+er+age+N:grade
# Best model: vital_status~1+batch+Platform+er+age+N:grade
# Best model: vital_status~1+batch+er+age+N:grade
# Crit= 3726.42538386574
# Mean crit= 3811.81503386729
print(deathPheno)
# ...
# From 100 models:
#   Best IC: 3726.42538386574
# Best model:
# [1] "vital_status ~ 1 + study + batch + Platform + er + age + N:grade"
# [1] "vital_status ~ 1 + study + batch + er + age + N:grade"
# [1] "vital_status ~ 1 + batch + Platform + er + age + N:grade"
# [1] "vital_status ~ 1 + batch + er + age + N:grade"
# Evidence weight: 0.130061042428855
# Worst IC: 3865.81056565321
# 8 models within 2 IC units.
# 10 models to reach 95% of evidence weight.

deathPhenoTx <- glmulti(vital_status ~ study + batch + Platform + age + grade + N + er + chemo + hormone + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 1050 models:
# Best model: vital_status~1+study+batch+Platform+er+chemo+age+N:grade
# Best model: vital_status~1+batch+er+chemo+age+N:grade
# Best model: vital_status~1+study+batch+er+chemo+age+N:grade
# Best model: vital_status~1+batch+Platform+er+chemo+age+N:grade
# Crit= 3420.29247363732
# Mean crit= 3445.7886385185
print(deathPhenoTx)
# ...
# From 100 models:
# Best IC: 3420.29247363732
# Best model:
# [1] "vital_status ~ 1 + study + batch + Platform + er + chemo + age + N:grade"                         # [1] "vital_status ~ 1 + batch + er + chemo + age + N:grade"
# [1] "vital_status ~ 1 + study + batch + er + chemo + age + N:grade"
# [1] "vital_status ~ 1 + batch + Platform + er + chemo + age + N:grade"
# Evidence weight: 0.0912790452963583
# Worst IC: 3476.30706988082
# 8 models within 2 IC units.
# 23 models to reach 95% of evidence weight.
>
recurPheno <- glmulti(recurrence_status ~ study + batch + Platform + age + grade + N + er + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 250 models:
# Best model: recurrence_status~1+er+age+grade+N:grade
# Crit= 2004.37782988452
# Mean crit= 2033.4708502359
print(recurPheno)
# ...
# From 100 models:
# Best IC: 2004.37782988452
# Best model:
# [1] "recurrence_status ~ 1 + er + age + grade + N:grade"
# Evidence weight: 0.685708296575585
# Worst IC: 2049.10376655078
# 1 models within 2 IC units.
# 4 models to reach 95% of evidence weight.

recurPhenoTx <- glmulti(recurrence_status ~ study + batch + Platform + age + grade + N + er + chemo + hormone + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 1050 models:
#   Best model: recurrence_status~1+er+chemo+age+grade+N:grade
# Crit= 1622.95267387007
# Mean crit= 1633.50245918406
print(recurPhenoTx)
# ...
# From 100 models:
# Best IC: 1622.95267387007
# Best model:
# [1] "recurrence_status ~ 1 + er + chemo + age + grade + N:grade"
# Evidence weight: 0.19147047890309
# Worst IC: 1638.88105348655
# 2 models within 2 IC units.
# 25 models to reach 95% of evidence weight.

# turn into tibbles with top 8 models
dir.create("data/metaGxBreast/mods")
modList <- c("deathPheno", "recurPheno", "deathPhenoTx", "recurPhenoTx")
walk(modList, ~ write(eval(sym(.)), paste0("data/metaGxBreast/mods/", ., ".csv"), sep = ","))
modsPheno <- map(modList, ~ read_csv(paste0("data/metaGxBreast/mods/", ., ".csv"), col_names = c("rank", "K", "IC", "model"), skip = 1)) %>%
  map(~ .[1:8,])
names(modsPheno) <- modList
modsPheno <- list(noTx = bind_rows(modsPheno[[1]], modsPheno[[2]], .id = "outcome"), tx = bind_rows(modsPheno[[3]], modsPheno[[4]], .id = "outcome"))
modsPheno <- map(modsPheno, ~ mutate(., outcome = factor(outcome, labels = c("death", "recurrence")))) %>%
  map(~ select(., - rank))

# evaluate models and get confusion matrix & scores
# weighted coefficients for consesus models
# defaults: unconditional variance weighting "Buckland", ci method "Lukacs"
coefPheno <- map(modList, ~ coef(eval(sym(.)), select = 8))
names(coefPheno) <- modList

# rerun for broom
# not controling for study shows up in risidual vs leverage plot
bestRecur <- glm(recurrence_status ~ 1 + er + age + grade + N:grade,
             family = binomial(link = logit), data = pbasic)
summary(bestRecur)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.3614  -0.9447  -0.7421   1.2729   1.9240  
# 
# Coefficients:
#                Estimate Std. Error z value Pr(>|z|)    
#   (Intercept) -1.006482   0.319829  -3.147 0.001650 ** 
#   erpositive  -0.219357   0.124480  -1.762 0.078036 .  
#   age         -0.010374   0.004127  -2.514 0.011953 *  
#   grade        0.396459   0.085315   4.647 3.37e-06 ***
#   grade:N      0.166370   0.044629   3.728 0.000193 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 2043.0  on 1580  degrees of freedom
# Residual deviance: 1967.5  on 1576  degrees of freedom
# (8127 observations deleted due to missingness)
# AIC: 1977.5
# 
# Number of Fisher Scoring iterations: 4

caret::confusionMatrix(table(predict(bestRecur, type = "response") >= 0.5,
                             bestRecur$model$recurrence_status == "recurrence"),
                       positive = "TRUE", mode = "everything")
#       FALSE TRUE
# FALSE   950  478
# TRUE     81   72
# 
# Accuracy : 0.6464        
# 95% CI : (0.6223, 0.67)
# No Information Rate : 0.6521        
# P-Value [Acc > NIR] : 0.6927        
# 
# Kappa : 0.0629        
# 
# Mcnemar's Test P-Value : <2e-16        
#                                         
#             Sensitivity : 0.13091       
#             Specificity : 0.92144       
#          Pos Pred Value : 0.47059       
#          Neg Pred Value : 0.66527       
#               Precision : 0.47059       
#                  Recall : 0.13091       
#                      F1 : 0.20484       
#              Prevalence : 0.34788       
#          Detection Rate : 0.04554       
#    Detection Prevalence : 0.09677       
#       Balanced Accuracy : 0.52617       
#                                         
#        'Positive' Class : TRUE          

broom::tidy(bestRecur)
#   term        estimate std.error statistic   p.value
# 1 (Intercept)  -1.01     0.320       -3.15 0.00165   
# 2 erpositive   -0.219    0.124       -1.76 0.0780    
# 3 age          -0.0104   0.00413     -2.51 0.0120    
# 4 grade         0.396    0.0853       4.65 0.00000337
# 5 grade:N       0.166    0.0446       3.73 0.000193  

bestRecur2 <- glm(recurrence_status ~ 1 + study + er + age + grade + N:grade,
                 family = binomial(link = logit), data = pbasic)
summary(bestRecur2)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.6094  -0.9420  -0.6758   1.2155   2.1475  
# 
# Coefficients:
#                  Estimate Std. Error z value Pr(>|z|)    
#   (Intercept)   -1.621073   0.420251  -3.857 0.000115 ***
#   studyNCI       0.747004   0.297524   2.511 0.012048 *  
#   studyNKI       0.320918   0.250920   1.279 0.200909    
#   studyPNC       0.460926   0.317150   1.453 0.146131    
#   studySTNO2     1.000573   0.289633   3.455 0.000551 ***
#   studyTRANSBIG  0.897614   0.272504   3.294 0.000988 ***
#   studyUCSF     -0.377587   0.297422  -1.270 0.204251    
#   studyUNC4     -0.154611   0.263597  -0.587 0.557511    
#   studyUNT       0.913850   0.305512   2.991 0.002779 ** 
#   studyUPP       0.094510   0.269771   0.350 0.726088    
#   erpositive    -0.219303   0.128323  -1.709 0.087452 .  
#   age           -0.007452   0.004797  -1.554 0.120300    
#   grade          0.405094   0.088682   4.568 4.92e-06 ***
#   grade:N        0.245961   0.050894   4.833 1.35e-06 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 2043  on 1580  degrees of freedom
# Residual deviance: 1907  on 1567  degrees of freedom
# (8127 observations deleted due to missingness)
# AIC: 1935
# 
# Number of Fisher Scoring iterations: 4

caret::confusionMatrix(table(predict(bestRecur2, type = "response") >= 0.5,
                             bestRecur2$model$recurrence_status == "recurrence"),
                       positive = "TRUE", mode = "everything")
#       FALSE TRUE
# FALSE   917  425
# TRUE    114  125
# 
# Accuracy : 0.6591          
# 95% CI : (0.6351, 0.6824)
# No Information Rate : 0.6521          
# P-Value [Acc > NIR] : 0.2903          
# 
# Kappa : 0.1344          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.22727         
#             Specificity : 0.88943         
#          Pos Pred Value : 0.52301         
#          Neg Pred Value : 0.68331         
#               Precision : 0.52301         
#                  Recall : 0.22727         
#                      F1 : 0.31686         
#              Prevalence : 0.34788         
#          Detection Rate : 0.07906         
#    Detection Prevalence : 0.15117         
#       Balanced Accuracy : 0.55835         
#                                           
#        'Positive' Class : TRUE            

broom::tidy(bestRecur2)
#   term        estimate std.error statistic   p.value
#  1 (Intercept)   -1.62      0.420      -3.86  0.000115  
#  2 studyNCI       0.747     0.298       2.51  0.0120    
#  3 studyNKI       0.321     0.251       1.28  0.201     
#  4 studyPNC       0.461     0.317       1.45  0.146     
#  5 studySTNO2     1.00      0.290       3.45  0.000551  
#  6 studyTRANSBIG  0.898     0.273       3.29  0.000988  
#  7 studyUCSF     -0.378     0.297      -1.27  0.204     
#  8 studyUNC4     -0.155     0.264      -0.587 0.558     
#  9 studyUNT       0.914     0.306       2.99  0.00278   
# 10 studyUPP       0.0945    0.270       0.350 0.726     
# 11 erpositive    -0.219     0.128      -1.71  0.0875    
# 12 age           -0.00745   0.00480    -1.55  0.120     
# 13 grade          0.405     0.0887      4.57  0.00000492
# 14 grade:N        0.246     0.0509      4.83  0.00000135

bestRecur2.1 <- glm(recurrence_status ~ 1 + batch + er + age + grade + N:grade,
                  family = binomial(link = logit), data = pbasic)
summary(bestRecur2.1)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.6288  -0.9156  -0.6422   1.1683   2.1448  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
#   (Intercept)  -1.433989   0.427055  -3.358 0.000785 ***
#   batchKIU      0.725317   0.359497   2.018 0.043634 *  
#   batchNCI      0.755595   0.298798   2.529 0.011446 *  
#   batchNKI      1.084272   0.312665   3.468 0.000525 ***
#   batchNKI2    -0.059698   0.268868  -0.222 0.824287    
#   batchOXFU     1.318252   0.391109   3.371 0.000750 ***
#   batchPNC      0.507413   0.319183   1.590 0.111897    
#   batchSTNO2    0.993873   0.290825   3.417 0.000632 ***
#   batchUCSF    -0.376066   0.298704  -1.259 0.208033    
#   batchUNC4    -0.140387   0.264894  -0.530 0.596131    
#   batchUPPT     0.349660   0.339150   1.031 0.302546    
#   batchUPPU    -0.041122   0.304614  -0.135 0.892615    
#   batchVDXGUYU  1.470395   0.411095   3.577 0.000348 ***
#   batchVDXIGRU  1.061463   0.372736   2.848 0.004403 ** 
#   batchVDXKIU   0.359897   0.395839   0.909 0.363245    
#   batchVDXOXFU  1.397699   0.476078   2.936 0.003326 ** 
#   batchVDXRHU   0.848753   0.411451   2.063 0.039129 *  
#   erpositive   -0.216640   0.130545  -1.660 0.097015 .  
#   age          -0.008774   0.004896  -1.792 0.073132 .  
#   grade         0.315891   0.091880   3.438 0.000586 ***
#   grade:N       0.310567   0.055556   5.590 2.27e-08 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 2043.0  on 1580  degrees of freedom
# Residual deviance: 1878.5  on 1560  degrees of freedom
# (8127 observations deleted due to missingness)
# AIC: 1920.5
# 
# Number of Fisher Scoring iterations: 4

caret::confusionMatrix(table(predict(bestRecur2.1, type = "response") >= 0.5,
                             bestRecur2.1$model$recurrence_status == "recurrence"),
                       positive = "TRUE", mode = "everything")
#       FALSE TRUE
# FALSE   913  387
# TRUE    118  163
# 
# Accuracy : 0.6806         
# 95% CI : (0.657, 0.7035)
# No Information Rate : 0.6521         
# P-Value [Acc > NIR] : 0.009073       
# 
# Kappa : 0.2053         
# 
# Mcnemar's Test P-Value : < 2.2e-16      
#                                          
#             Sensitivity : 0.2964         
#             Specificity : 0.8855         
#          Pos Pred Value : 0.5801         
#          Neg Pred Value : 0.7023         
#               Precision : 0.5801         
#                  Recall : 0.2964         
#                      F1 : 0.3923         
#              Prevalence : 0.3479         
#          Detection Rate : 0.1031         
#    Detection Prevalence : 0.1777         
#       Balanced Accuracy : 0.5910         
#                                          
#        'Positive' Class : TRUE           

bestRecur2.2 <- glm(recurrence_status ~ 1 + batch + age + grade + N:grade,
                    family = binomial(link = logit), data = pbasic)
summary(bestRecur2.2)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.6357  -0.9152  -0.6423   1.1740   2.1543  
# 
# Coefficients:
#                 Estimate Std. Error z value Pr(>|z|)    
#   (Intercept)  -1.623811   0.409875  -3.962 7.44e-05 ***
#   batchKIU      0.696846   0.358769   1.942 0.052098 .  
#   batchNCI      0.768252   0.299135   2.568 0.010221 *  
#   batchNKI      1.058143   0.311641   3.395 0.000685 ***
#   batchNKI2    -0.080565   0.269009  -0.299 0.764567    
#   batchOXFU     1.233241   0.381717   3.231 0.001235 ** 
#   batchPNC      0.542526   0.316567   1.714 0.086569 .  
#   batchSTNO2    0.987880   0.289951   3.407 0.000657 ***
#   batchUCSF    -0.404146   0.296024  -1.365 0.172175    
#   batchUNC4    -0.134603   0.265122  -0.508 0.611662    
#   batchUPPT     0.306133   0.337930   0.906 0.364985    
#   batchUPPU    -0.061009   0.304740  -0.200 0.841323    
#   batchVDXGUYU  1.472029   0.410762   3.584 0.000339 ***
#   batchVDXIGRU  1.043644   0.372098   2.805 0.005035 ** 
#   batchVDXKIU   0.348791   0.395235   0.882 0.377512    
#   batchVDXOXFU  1.442471   0.475115   3.036 0.002397 ** 
#   batchVDXRHU   0.849707   0.410866   2.068 0.038632 *  
#   age          -0.009999   0.004829  -2.071 0.038387 *  
#   grade         0.361237   0.086809   4.161 3.16e-05 ***
#   grade:N       0.311878   0.054954   5.675 1.39e-08 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 2063.0  on 1599  degrees of freedom
# Residual deviance: 1897.1  on 1580  degrees of freedom
# (8108 observations deleted due to missingness)
# AIC: 1937.1
# 
# Number of Fisher Scoring iterations: 4

caret::confusionMatrix(table(predict(bestRecur2.2, type = "response") >= 0.5,
                             bestRecur2.2$model$recurrence_status == "recurrence"),
                       positive = "TRUE", mode = "everything")
#      FALSE TRUE
# FALSE   930  392
# TRUE    117  161
# 
# Accuracy : 0.6819          
# 95% CI : (0.6584, 0.7047)
# No Information Rate : 0.6544          
# P-Value [Acc > NIR] : 0.01076         
# 
# Kappa : 0.2032          
# 
# Mcnemar's Test P-Value : < 2e-16         
#                                           
#             Sensitivity : 0.2911          
#             Specificity : 0.8883          
#          Pos Pred Value : 0.5791          
#          Neg Pred Value : 0.7035          
#               Precision : 0.5791          
#                  Recall : 0.2911          
#                      F1 : 0.3875          
#              Prevalence : 0.3456          
#          Detection Rate : 0.1006          
#    Detection Prevalence : 0.1737          
#       Balanced Accuracy : 0.5897          
#                                           
#        'Positive' Class : TRUE            

bestRecurTx <- glm(recurrence_status ~ 1 + er + chemo + age + grade + N:grade,
                 family = binomial(link = logit), data = pbasic)
summary(bestRecurTx)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.4001  -0.9426  -0.7443   1.2780   1.9134  
# 
# Coefficients:
#               Estimate Std. Error z value Pr(>|z|)    
#   (Intercept) -0.732315   0.373864  -1.959  0.05014 .  
#   erpositive  -0.255943   0.144144  -1.776  0.07580 .  
#   chemoTRUE   -0.366302   0.177601  -2.062  0.03916 *  
#   age         -0.013519   0.004904  -2.757  0.00584 ** 
#   grade        0.413920   0.094267   4.391 1.13e-05 ***
#   grade:N      0.180357   0.058968   3.059  0.00222 ** 
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1641.2  on 1263  degrees of freedom
# Residual deviance: 1580.1  on 1258  degrees of freedom
# (8444 observations deleted due to missingness)
# AIC: 1592.1
# 
# Number of Fisher Scoring iterations: 4

caret::confusionMatrix(table(predict(bestRecurTx, type = "response") >= 0.5,
                             bestRecurTx$model$recurrence_status == "recurrence"),
                       positive = "TRUE", mode = "everything")
#       FALSE TRUE
# FALSE   762  383
# TRUE     56   63
# 
# Accuracy : 0.6527         
# 95% CI : (0.6257, 0.679)
# No Information Rate : 0.6472         
# P-Value [Acc > NIR] : 0.3519         
# 
# Kappa : 0.0874         
# 
# Mcnemar's Test P-Value : <2e-16         
#                                          
#             Sensitivity : 0.14126        
#             Specificity : 0.93154        
#          Pos Pred Value : 0.52941        
#          Neg Pred Value : 0.66550        
#               Precision : 0.52941        
#                  Recall : 0.14126        
#                      F1 : 0.22301        
#              Prevalence : 0.35285        
#          Detection Rate : 0.04984        
#    Detection Prevalence : 0.09415        
#       Balanced Accuracy : 0.53640        
#                                          
#        'Positive' Class : TRUE           

bestRecurTx2 <- glm(recurrence_status ~ 1 + study + er + hormone + grade + N:grade,
                  family = binomial(link = logit), data = pbasic)
summary(bestRecurTx2)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.5077  -0.9533  -0.6791   1.2009   2.1212  
# 
# Coefficients:
#                 Estimate Std. Error z value Pr(>|z|)    
#   (Intercept)   -2.00119    0.35700  -5.606 2.07e-08 ***
#   studyNCI       0.66321    0.29762   2.228  0.02586 *  
#   studyNKI       0.37298    0.25691   1.452  0.14656    
#   studySTNO2     0.93505    0.28675   3.261  0.00111 ** 
#   studyTRANSBIG  0.87078    0.28535   3.052  0.00228 ** 
#   studyUCSF     -0.43881    0.29738  -1.476  0.14006    
#   studyUNT       0.85864    0.31974   2.685  0.00724 ** 
#   studyUPP      -0.08329    0.28178  -0.296  0.76755    
#   erpositive    -0.22391    0.14784  -1.515  0.12988    
#   hormoneTRUE    0.09262    0.17447   0.531  0.59550    
#   grade          0.43303    0.09640   4.492 7.05e-06 ***
#   grade:N        0.17218    0.06183   2.785  0.00536 ** 
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1641.2  on 1263  degrees of freedom
# Residual deviance: 1539.6  on 1252  degrees of freedom
# (8444 observations deleted due to missingness)
# AIC: 1563.6
# 
# Number of Fisher Scoring iterations: 4

caret::confusionMatrix(table(predict(bestRecurTx2, type = "response") >= 0.5,
                             bestRecurTx2$model$recurrence_status == "recurrence"),
                       positive = "TRUE", mode = "everything")
#       FALSE TRUE
# FALSE   725  338
# TRUE     93  108
# 
# Accuracy : 0.659           
# 95% CI : (0.6321, 0.6852)
# No Information Rate : 0.6472          
# P-Value [Acc > NIR] : 0.1969          
# 
# Kappa : 0.1468          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.24215         
#             Specificity : 0.88631         
#          Pos Pred Value : 0.53731         
#          Neg Pred Value : 0.68203         
#               Precision : 0.53731         
#                  Recall : 0.24215         
#                      F1 : 0.33385         
#              Prevalence : 0.35285         
#          Detection Rate : 0.08544         
#    Detection Prevalence : 0.15902         
#       Balanced Accuracy : 0.56423         
#                                           
#        'Positive' Class : TRUE            

broom::tidy(bestRecurTx2)
#      term        estimate std.error statistic   p.value
#  1 (Intercept)    -2.00      0.357     -5.61  0.0000000207
#  2 studyNCI        0.663     0.298      2.23  0.0259      
#  3 studyNKI        0.373     0.257      1.45  0.147       
#  4 studySTNO2      0.935     0.287      3.26  0.00111     
#  5 studyTRANSBIG   0.871     0.285      3.05  0.00228     
#  6 studyUCSF      -0.439     0.297     -1.48  0.140       
#  7 studyUNT        0.859     0.320      2.69  0.00724     
#  8 studyUPP       -0.0833    0.282     -0.296 0.768       
#  9 erpositive     -0.224     0.148     -1.51  0.130       
# 10 hormoneTRUE     0.0926    0.174      0.531 0.596       
# 11 grade           0.433     0.0964     4.49  0.00000705  
# 12 grade:N         0.172     0.0618     2.78  0.00536     

bestRecurTx2.1 <- glm(recurrence_status ~ 1 + batch + er + hormone + grade + N:grade,
                    family = binomial(link = logit), data = pbasic)
summary(bestRecurTx2.1)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.5570  -0.9117  -0.6500   1.1691   2.1310  
# 
# Coefficients:
#                 Estimate Std. Error z value Pr(>|z|)    
#   (Intercept)  -1.80969    0.36601  -4.944 7.64e-07 ***
#   batchKIU      0.61187    0.38121   1.605 0.108479    
#   batchNCI      0.69341    0.29925   2.317 0.020493 *  
#   batchNKI      1.01297    0.33056   3.064 0.002181 ** 
#   batchNKI2    -0.01957    0.27665  -0.071 0.943616    
#   batchOXFU     1.18601    0.40812   2.906 0.003661 ** 
#   batchSTNO2    0.92233    0.28786   3.204 0.001355 ** 
#   batchUCSF    -0.44144    0.29855  -1.479 0.139235    
#   batchUPPT     0.23757    0.34119   0.696 0.486243    
#   batchUPPU    -0.36227    0.36490  -0.993 0.320809    
#   batchVDXGUYU  1.36697    0.42784   3.195 0.001398 ** 
#   batchVDXIGRU  0.99415    0.38995   2.549 0.010790 *  
#   batchVDXKIU   0.27533    0.41257   0.667 0.504542    
#   batchVDXOXFU  1.32008    0.48854   2.702 0.006891 ** 
#   batchVDXRHU   0.76306    0.42743   1.785 0.074228 .  
#   erpositive   -0.20002    0.15188  -1.317 0.187875    
#   hormoneTRUE  -0.04429    0.19786  -0.224 0.822876    
#   grade         0.33375    0.10070   3.314 0.000919 ***
#   grade:N       0.24832    0.06938   3.579 0.000345 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1641.2  on 1263  degrees of freedom
# Residual deviance: 1513.6  on 1245  degrees of freedom
# (8444 observations deleted due to missingness)
# AIC: 1551.6
# 
# Number of Fisher Scoring iterations: 4

caret::confusionMatrix(table(predict(bestRecurTx2.1, type = "response") >= 0.5,
                             bestRecurTx2.1$model$recurrence_status == "recurrence"),
                       positive = "TRUE", mode = "everything")
#       FALSE TRUE
# FALSE   714  289
# TRUE    104  157
# 
# Accuracy : 0.6891          
# 95% CI : (0.6628, 0.7145)
# No Information Rate : 0.6472          
# P-Value [Acc > NIR] : 0.0009124       
# 
# Kappa : 0.2483          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.3520          
#             Specificity : 0.8729          
#          Pos Pred Value : 0.6015          
#          Neg Pred Value : 0.7119          
#               Precision : 0.6015          
#                  Recall : 0.3520          
#                      F1 : 0.4441          
#              Prevalence : 0.3528          
#          Detection Rate : 0.1242          
#    Detection Prevalence : 0.2065          
#       Balanced Accuracy : 0.6124          
#                                           
#        'Positive' Class : TRUE            

bestRecurTx2.2 <- glm(recurrence_status ~ 1 + batch + chemo + hormone + grade + N:grade,
                      family = binomial(link = logit), data = pbasic)
summary(bestRecurTx2.2)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.5701  -0.9101  -0.6558   1.1477   2.1774  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
#   (Intercept)  -2.145205   0.370754  -5.786 7.21e-09 ***
#   batchKIU      0.683484   0.403039   1.696  0.08992 .  
#   batchNCI      0.789479   0.307499   2.567  0.01025 *  
#   batchNKI      1.089780   0.351100   3.104  0.00191 ** 
#   batchNKI2    -0.004696   0.282261  -0.017  0.98673    
#   batchOXFU     1.194536   0.422325   2.828  0.00468 ** 
#   batchSTNO2    0.983836   0.295479   3.330  0.00087 ***
#   batchUCSF    -0.467527   0.296171  -1.579  0.11443    
#   batchUPPT     0.292362   0.349241   0.837  0.40252    
#   batchUPPU    -0.284016   0.387642  -0.733  0.46376    
#   batchVDXGUYU  1.469644   0.447225   3.286  0.00102 ** 
#   batchVDXIGRU  1.083582   0.410834   2.638  0.00835 ** 
#   batchVDXKIU   0.368615   0.432337   0.853  0.39388    
#   batchVDXOXFU  1.465629   0.506740   2.892  0.00382 ** 
#   batchVDXRHU   0.867430   0.446812   1.941  0.05221 .  
#   chemoTRUE     0.215528   0.215650   0.999  0.31758    
#   hormoneTRUE  -0.037080   0.213091  -0.174  0.86186    
#   grade         0.377347   0.095648   3.945 7.98e-05 ***
#   grade:N       0.233927   0.072320   3.235  0.00122 ** 
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1657.5  on 1279  degrees of freedom
# Residual deviance: 1527.5  on 1261  degrees of freedom
# (8428 observations deleted due to missingness)
# AIC: 1565.5
# 
# Number of Fisher Scoring iterations: 4

caret::confusionMatrix(table(predict(bestRecurTx2.2, type = "response") >= 0.5,
                             bestRecurTx2.2$model$recurrence_status == "recurrence"),
                       positive = "TRUE", mode = "everything")
#       FALSE TRUE
# FALSE   725  300
# TRUE    107  148
# 
# Accuracy : 0.682           
# 95% CI : (0.6557, 0.7075)
# No Information Rate : 0.65            
# P-Value [Acc > NIR] : 0.008472        
# 
# Kappa : 0.224           
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.3304          
#             Specificity : 0.8714          
#          Pos Pred Value : 0.5804          
#          Neg Pred Value : 0.7073          
#               Precision : 0.5804          
#                  Recall : 0.3304          
#                      F1 : 0.4211          
#              Prevalence : 0.3500          
#          Detection Rate : 0.1156          
#    Detection Prevalence : 0.1992          
#       Balanced Accuracy : 0.6009          
#                                           
#        'Positive' Class : TRUE            


# add pam50.  TODO:  pam50_batch, PAM50_prediction, GGI_prediction

pb50 <- left_join(pbasic, select(pheno, sample_name, pam50, pam50_batch)) %>%
  mutate(pam50 = as.factor(pam50), pam50_batch = as.factor(pam50_batch))



bestRecurP50_2.2 <- glm(recurrence_status ~ 1 + batch + age + grade + N:grade + pam50,
                    family = binomial(link = logit), data = pb50)
summary(bestRecurP50_2.2)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.7931  -0.9093  -0.6243   1.1258   2.2118  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
#   (Intercept)  -1.687250   0.443138  -3.808 0.000140 ***
#   batchKIU      0.654207   0.363400   1.800 0.071823 .  
#   batchNCI      0.766571   0.306235   2.503 0.012307 *  
#   batchNKI      1.076245   0.316541   3.400 0.000674 ***
#   batchNKI2    -0.092945   0.272979  -0.340 0.733492    
#   batchOXFU     1.213789   0.386859   3.138 0.001704 ** 
#   batchPNC      0.532607   0.322280   1.653 0.098408 .  
#   batchSTNO2    1.058873   0.296245   3.574 0.000351 ***
#   batchUCSF    -0.298133   0.309458  -0.963 0.335344    
#   batchUNC4    -0.067815   0.273352  -0.248 0.804067    
#   batchUPPT     0.235944   0.343329   0.687 0.491942    
#   batchUPPU    -0.036281   0.308826  -0.117 0.906480    
#   batchVDXGUYU  1.481570   0.416499   3.557 0.000375 ***
#   batchVDXIGRU  1.171006   0.378552   3.093 0.001979 ** 
#   batchVDXKIU   0.440780   0.401035   1.099 0.271721    
#   batchVDXOXFU  1.548313   0.480372   3.223 0.001268 ** 
#   batchVDXRHU   0.844204   0.418024   2.020 0.043434 *  
#   age          -0.010386   0.004883  -2.127 0.033432 *  
#   grade         0.286161   0.094982   3.013 0.002588 ** 
#   pam50Her2     0.518091   0.186070   2.784 0.005363 ** 
#   pam50LumA    -0.087156   0.184510  -0.472 0.636667    
#   pam50LumB     0.631777   0.164085   3.850 0.000118 ***
#   pam50Normal   0.166290   0.237541   0.700 0.483898    
#   grade:N       0.295207   0.055679   5.302 1.15e-07 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 2063.0  on 1599  degrees of freedom
# Residual deviance: 1868.1  on 1576  degrees of freedom
# (8108 observations deleted due to missingness)
# AIC: 1916.1
# 
# Number of Fisher Scoring iterations: 4

caret::confusionMatrix(table(predict(bestRecurP50_2.2, type = "response") >= 0.5,
                             bestRecurP50_2.2$model$recurrence_status == "recurrence"),
                       positive = "TRUE", mode = "everything")
#       FALSE TRUE
# FALSE   916  361
# TRUE    131  192
# 
# Accuracy : 0.6925          
# 95% CI : (0.6692, 0.7151)
# No Information Rate : 0.6544          
# P-Value [Acc > NIR] : 0.0006701       
# 
# Kappa : 0.2462          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.3472          
#             Specificity : 0.8749          
#          Pos Pred Value : 0.5944          
#          Neg Pred Value : 0.7173          
#               Precision : 0.5944          
#                  Recall : 0.3472          
#                      F1 : 0.4384          
#              Prevalence : 0.3456          
#          Detection Rate : 0.1200          
#    Detection Prevalence : 0.2019          
#       Balanced Accuracy : 0.6110          
#                                           
#        'Positive' Class : TRUE            

bestRecurP50b_2.2 <- glm(recurrence_status ~ 1 + batch + age + grade + N:grade + pam50_batch,
                        family = binomial(link = logit), data = pb50)
summary(bestRecurP50b_2.2)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.7931  -0.9093  -0.6243   1.1258   2.2118  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
#   (Intercept)  -1.687250   0.443138  -3.808 0.000140 ***
#   batchKIU      0.654207   0.363400   1.800 0.071823 .  
#   batchNCI      0.766571   0.306235   2.503 0.012307 *  
#   batchNKI      1.076245   0.316541   3.400 0.000674 ***
#   batchNKI2    -0.092945   0.272979  -0.340 0.733492    
#   batchOXFU     1.213789   0.386859   3.138 0.001704 ** 
#   batchPNC      0.532607   0.322280   1.653 0.098408 .  
#   batchSTNO2    1.058873   0.296245   3.574 0.000351 ***
#   batchUCSF    -0.298133   0.309458  -0.963 0.335344    
#   batchUNC4    -0.067815   0.273352  -0.248 0.804067    
#   batchUPPT     0.235944   0.343329   0.687 0.491942    
#   batchUPPU    -0.036281   0.308826  -0.117 0.906480    
#   batchVDXGUYU  1.481570   0.416499   3.557 0.000375 ***
#   batchVDXIGRU  1.171006   0.378552   3.093 0.001979 ** 
#   batchVDXKIU   0.440780   0.401035   1.099 0.271721    
#   batchVDXOXFU  1.548313   0.480372   3.223 0.001268 ** 
#   batchVDXRHU   0.844204   0.418024   2.020 0.043434 *  
#   age          -0.010386   0.004883  -2.127 0.033432 *  
#   grade         0.286161   0.094982   3.013 0.002588 ** 
#   pam50Her2     0.518091   0.186070   2.784 0.005363 ** 
#   pam50LumA    -0.087156   0.184510  -0.472 0.636667    
#   pam50LumB     0.631777   0.164085   3.850 0.000118 ***
#   pam50Normal   0.166290   0.237541   0.700 0.483898    
#   grade:N       0.295207   0.055679   5.302 1.15e-07 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 2063.0  on 1599  degrees of freedom
# Residual deviance: 1868.1  on 1576  degrees of freedom
# (8108 observations deleted due to missingness)
# AIC: 1916.1
# 
# Number of Fisher Scoring iterations: 4

caret::confusionMatrix(table(predict(bestRecurP50b_2.2, type = "response") >= 0.5,
                             bestRecurP50b_2.2$model$recurrence_status == "recurrence"),
                       positive = "TRUE", mode = "everything")
#       FALSE TRUE
# FALSE   907  364
# TRUE    140  189
# 
# Accuracy : 0.685           
# 95% CI : (0.6616, 0.7077)
# No Information Rate : 0.6544          
# P-Value [Acc > NIR] : 0.005151        
# 
# Kappa : 0.23            
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.3418          
#             Specificity : 0.8663          
#          Pos Pred Value : 0.5745          
#          Neg Pred Value : 0.7136          
#               Precision : 0.5745          
#                  Recall : 0.3418          
#                      F1 : 0.4286          
#              Prevalence : 0.3456          
#          Detection Rate : 0.1181          
#    Detection Prevalence : 0.2056          
#       Balanced Accuracy : 0.6040          
#                                           
#        'Positive' Class : TRUE            

bestRecurTxP50_2.2 <- glm(recurrence_status ~ 1 + batch + chemo + hormone + grade + N:grade + pam50,
                      family = binomial(link = logit), data = pb50)
summary(bestRecurTxP50_2.2)
Deviance Residuals: 
  Min       1Q   Median       3Q      Max  
-1.6444  -0.9192  -0.6149   1.1227   2.1849  

Coefficients:
  Estimate Std. Error z value Pr(>|z|)    
(Intercept)  -2.05868    0.43235  -4.762 1.92e-06 ***
  batchKIU      0.57877    0.40855   1.417 0.156587    
batchNCI      0.76213    0.31639   2.409 0.016006 *  
  batchNKI      1.06866    0.35631   2.999 0.002707 ** 
  batchNKI2    -0.05007    0.28636  -0.175 0.861201    
batchOXFU     1.11631    0.42860   2.605 0.009199 ** 
  batchSTNO2    1.03049    0.30212   3.411 0.000647 ***
  batchUCSF    -0.38850    0.31257  -1.243 0.213906    
batchUPPT     0.21125    0.35499   0.595 0.551787    
batchUPPU    -0.32140    0.39210  -0.820 0.412397    
batchVDXGUYU  1.42998    0.45339   3.154 0.001611 ** 
  batchVDXIGRU  1.16948    0.41706   2.804 0.005046 ** 
  batchVDXKIU   0.41790    0.43831   0.953 0.340372    
batchVDXOXFU  1.51947    0.51310   2.961 0.003063 ** 
  batchVDXRHU   0.81216    0.45415   1.788 0.073727 .  
chemoTRUE     0.19491    0.21752   0.896 0.370209    
hormoneTRUE  -0.08615    0.21505  -0.401 0.688697    
grade         0.26726    0.10753   2.486 0.012936 *  
  pam50Her2     0.54062    0.21300   2.538 0.011144 *  
  pam50LumA    -0.17766    0.20792  -0.854 0.392831    
pam50LumB     0.59708    0.18553   3.218 0.001290 ** 
  pam50Normal   0.12280    0.28742   0.427 0.669200    
grade:N       0.21460    0.07328   2.928 0.003407 ** 
  ---
  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

Null deviance: 1657.5  on 1279  degrees of freedom
Residual deviance: 1501.2  on 1257  degrees of freedom
(8428 observations deleted due to missingness)
AIC: 1547.2

Number of Fisher Scoring iterations: 4

caret::confusionMatrix(table(predict(bestRecurTxP50_2.2, type = "response") >= 0.5,
                             bestRecurTxP50_2.2$model$recurrence_status == "recurrence"),
                       positive = "TRUE", mode = "everything")
Confusion Matrix and Statistics


FALSE TRUE
FALSE   721  288
TRUE    111  160

Accuracy : 0.6883          
95% CI : (0.6621, 0.7136)
No Information Rate : 0.65            
P-Value [Acc > NIR] : 0.002087        

Kappa : 0.2462          

Mcnemar's Test P-Value : < 2.2e-16       
                                          
            Sensitivity : 0.3571          
            Specificity : 0.8666          
         Pos Pred Value : 0.5904          
         Neg Pred Value : 0.7146          
              Precision : 0.5904          
                 Recall : 0.3571          
                     F1 : 0.4451          
             Prevalence : 0.3500          
         Detection Rate : 0.1250          
   Detection Prevalence : 0.2117          
      Balanced Accuracy : 0.6119          
                                          
       'Positive' Class : TRUE            
