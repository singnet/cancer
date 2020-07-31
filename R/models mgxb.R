## metaGx time to event models
library(tidyverse)
library(broom)

# load metaGx study functions

pheno <- read_csv("data/metaGxBreast/metaGXcovarTable.csv.xz", guess_max = 10000)
dim(pheno)
# [1] 9717   29

summary(pheno$age_at_initial_pathologic_diagnosis)
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 21.00   46.00   56.00   56.78   66.39   96.29    2095 

summary(pheno$tumor_size)
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 0.000   1.550   2.100   2.458   3.000  18.200    4417 

table(pheno$treatment, exclude = NULL)
# chemo.plus.hormono       chemotherapy     hormonotherapy          untreated               <NA> 
#                378               1313               1640               2188               4198 

table(pheno$grade, exclude = NULL)
#   1    2    3    4 <NA> 
# 701 2299 2903   15 3799 

table(pheno$T)
#  1   2   3   4  T0  T1  T2  T3  T4 
# 48 145  31   6   3  30 255 145  75 

table(pheno$N)
#    0    1 
# 3534 2843 

table(pheno$recurrence_status)
# norecurrence   recurrence 
#         1228          628 

table(pheno$vital_status, exclude = NULL)
# deceased   living     <NA> 
#     1423     2987     5307 

xtabs(~ treatment + vital_status, data = pheno, exclude = NULL)
#                        vital_status
# treatment            deceased living
#   chemo.plus.hormono      103    143
#   chemotherapy            195    211
#   hormonotherapy          552    684
#   untreated               350    600

xtabs(~ vital_status + recurrence_status, data = pheno, exclude = NULL)
#                 recurrence_status
# vital_status norecurrence recurrence
#     deceased          122        264
#     living            657        160

control <- filter(pheno, treatment == "untreated",
                  !is.na(vital_status) | !is.na(recurrence_status)) %>%
  select(study, batch, Platform, sample_name, age_dx = age_at_initial_pathologic_diagnosis,
         grade, N, er, her2, pgr, vital_status, recurrence_status)

write_csv(control, "data/metaGxBreast/controlModel.csv.xz")

# baseline model
control <- mutate(control, status = as.factor(recurrence_status))

# not controling for study shows up in risidual vs leverage plot
recur <- glm(status ~ age_dx + grade + N,
             family = binomial(link = logit), data = control)
summary(recur)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.3224  -0.9252  -0.7173   1.2310   2.1671  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -0.063798   0.500720  -0.127    0.899    
# age_dx      -0.032956   0.007978  -4.131 3.61e-05 ***
# grade        0.452494   0.113974   3.970 7.18e-05 ***
# N            0.392096   0.309317   1.268    0.205    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 874.03  on 682  degrees of freedom
# Residual deviance: 829.45  on 679  degrees of freedom
# (583 observations deleted due to missingness)
# AIC: 837.45
# 
# Number of Fisher Scoring iterations: 4

1 - pchisq(829.45, 682)
# [1] 8.699928e-05

caret::confusionMatrix(table(predict(recur, type = "response") >= 0.5,
                             recur$model$status == "recurrence"),
                       positive = "TRUE", mode = "everything")
# FALSE TRUE
# FALSE   416  192
# TRUE     36   39
# 
# Accuracy : 0.6662          
# 95% CI : (0.6294, 0.7015)
# No Information Rate : 0.6618          
# P-Value [Acc > NIR] : 0.4215          
# 
# Kappa : 0.1068          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.1688          
#             Specificity : 0.9204          
#          Pos Pred Value : 0.5200          
#          Neg Pred Value : 0.6842          
#               Precision : 0.5200          
#                  Recall : 0.1688          
#                      F1 : 0.2549          
#              Prevalence : 0.3382          
#          Detection Rate : 0.0571          
#    Detection Prevalence : 0.1098          
#       Balanced Accuracy : 0.5446          
#                                           
#        'Positive' Class : TRUE

tidy(recur)
#   term        estimate std.error statistic   p.value
# 1 (Intercept)  -0.0638   0.501      -0.127 0.899    
# 2 age_dx       -0.0330   0.00798    -4.13  0.0000361
# 3 grade         0.452    0.114       3.97  0.0000718
# 4 N             0.392    0.309       1.27  0.205    

recur.a <- augment(recur, data = recur$model)

recurE <- glm(status ~ age_dx + grade + N + er,
              family = binomial(link = logit), data = control)

summary(recurE)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.2939  -0.9342  -0.7235   1.2456   2.1497  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)  0.016838   0.546258   0.031 0.975409    
# age_dx      -0.032241   0.007992  -4.034 5.48e-05 ***
#   grade        0.428246   0.123691   3.462 0.000536 ***
#   N            0.335451   0.313705   1.069 0.284925    
# erpositive  -0.071949   0.198868  -0.362 0.717508    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 866.89  on 675  degrees of freedom
# Residual deviance: 824.19  on 671  degrees of freedom
# (590 observations deleted due to missingness)
# AIC: 834.19
# 
# Number of Fisher Scoring iterations: 4

1 - pchisq(824.19, 671)
# [1] 4.464349e-05

caret::confusionMatrix(table(predict(recurE, type = "response") >= 0.5,
                             recurE$model$status == "recurrence"),
                       positive = "TRUE", mode = "everything")
# FALSE TRUE
# FALSE   406  194
# TRUE     40   36
# 
# Accuracy : 0.6538          
# 95% CI : (0.6166, 0.6897)
# No Information Rate : 0.6598          
# P-Value [Acc > NIR] : 0.6439          
# 
# Kappa : 0.0798          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.15652         
#             Specificity : 0.91031         
#          Pos Pred Value : 0.47368         
#          Neg Pred Value : 0.67667         
#               Precision : 0.47368         
#                  Recall : 0.15652         
#                      F1 : 0.23529         
#              Prevalence : 0.34024         
#          Detection Rate : 0.05325         
#    Detection Prevalence : 0.11243         
#       Balanced Accuracy : 0.53342         
#                                           
#        'Positive' Class : TRUE            
                                          
tidy(recurE)
#   term        estimate std.error statistic   p.value
# 1 (Intercept)   0.0168   0.546      0.0308 0.975    
# 2 age_dx       -0.0322   0.00799   -4.03   0.0000548
# 3 grade         0.428    0.124      3.46   0.000536 
# 4 N             0.335    0.314      1.07   0.285    
# 5 erpositive   -0.0719   0.199     -0.362  0.718    

recurE.a <- augment(recurE, recurE$model)

recurS <- glm(status ~ study + age_dx + grade + N,
             family = binomial(link = logit), data = control)
summary(recurS)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.4296  -0.9469  -0.6512   1.2395   2.3129  
# 
# Coefficients:
#                Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   -0.667811   0.862085  -0.775 0.438549    
# studyNCI      -0.146340   0.916412  -0.160 0.873127    
# studyNKI       0.062170   0.627117   0.099 0.921030    
# studySTNO2    -0.603182   0.803381  -0.751 0.452771    
# studyTRANSBIG  0.468679   0.633463   0.740 0.459380    
# studyUCSF     -0.980669   0.822299  -1.193 0.233028    
# studyUNT       0.518788   0.649714   0.798 0.424588    
# studyUPP      -0.441049   0.670579  -0.658 0.510722    
# age_dx        -0.024354   0.009401  -2.591 0.009581 ** 
# grade          0.453939   0.118126   3.843 0.000122 ***
# N              0.818022   0.368150   2.222 0.026285 *  
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 874.03  on 682  degrees of freedom
# Residual deviance: 811.25  on 672  degrees of freedom
# (583 observations deleted due to missingness)
# AIC: 833.25
# 
# Number of Fisher Scoring iterations: 4

1 - pchisq(811.25, 672)
# [1] 0.000170718
 
caret::confusionMatrix(table(predict(recurS, type = "response") >= 0.5,
                             recurS$model$status == "recurrence"),
                       positive = "TRUE", mode = "everything")
#       FALSE TRUE
# FALSE   401  189
# TRUE     51   42
# 
# Accuracy : 0.6486          
# 95% CI : (0.6115, 0.6844)
# No Information Rate : 0.6618          
# P-Value [Acc > NIR] : 0.7794          
# 
# Kappa : 0.0808          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.18182         
#             Specificity : 0.88717         
#          Pos Pred Value : 0.45161         
#          Neg Pred Value : 0.67966         
#               Precision : 0.45161         
#                  Recall : 0.18182         
#                      F1 : 0.25926         
#              Prevalence : 0.33821         
#          Detection Rate : 0.06149         
#    Detection Prevalence : 0.13616         
#       Balanced Accuracy : 0.53449         
#                                           
#        'Positive' Class : TRUE            

tidy(recurS)
#   term          estimate std.error statistic  p.value
# 1 (Intercept)    -0.668    0.862     -0.775  0.439   
# 2 studyNCI       -0.146    0.916     -0.160  0.873   
# 3 studyNKI        0.0622   0.627      0.0991 0.921   
# 4 studySTNO2     -0.603    0.803     -0.751  0.453   
# 5 studyTRANSBIG   0.469    0.633      0.740  0.459   
# 6 studyUCSF      -0.981    0.822     -1.19   0.233   
# 7 studyUNT        0.519    0.650      0.798  0.425   
# 8 studyUPP       -0.441    0.671     -0.658  0.511   
# 9 age_dx         -0.0244   0.00940   -2.59   0.00958 
# 10 grade           0.454    0.118      3.84   0.000122
# 11 N               0.818    0.368      2.22   0.0263  

recurS.a <- augment(recurS, recurS$model)

recurSE <- glm(status ~ study + age_dx + grade + N + er,
             family = binomial(link = logit), data = control)
summary(recurSE)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.4038  -0.9491  -0.6496   1.2312   2.3012  
# 
# Coefficients:
#                Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   -0.637859   0.886063  -0.720 0.471599    
# studyNCI      -0.169438   0.915901  -0.185 0.853233    
# studyNKI       0.058939   0.625833   0.094 0.924968    
# studySTNO2    -0.813724   0.832597  -0.977 0.328404    
# studyTRANSBIG  0.454631   0.631964   0.719 0.471898    
# studyUCSF     -0.946221   0.822510  -1.150 0.249976    
# studyUNT       0.555776   0.649010   0.856 0.391807    
# studyUPP      -0.455479   0.669227  -0.681 0.496122    
# age_dx        -0.023511   0.009479  -2.480 0.013128 *  
# grade          0.442354   0.127919   3.458 0.000544 ***
# N              0.776224   0.371111   2.092 0.036472 *  
# erpositive    -0.042524   0.203216  -0.209 0.834248    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 866.89  on 675  degrees of freedom
# Residual deviance: 805.00  on 664  degrees of freedom
# (590 observations deleted due to missingness)
# AIC: 829
# 
# Number of Fisher Scoring iterations: 4

1 - pchisq(805, 664)
# [1] 0.0001347837

caret::confusionMatrix(table(predict(recurSE, type = "response") >= 0.5,
                             recurSE$model$status == "recurrence"),
                       positive = "TRUE", mode = "everything")
# FALSE TRUE
# FALSE   393  187
# TRUE     53   43
# 
# Accuracy : 0.645           
# 95% CI : (0.6076, 0.6811)
# No Information Rate : 0.6598          
# P-Value [Acc > NIR] : 0.8033          
# 
# Kappa : 0.0793          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.18696         
#             Specificity : 0.88117         
#          Pos Pred Value : 0.44792         
#          Neg Pred Value : 0.67759         
#               Precision : 0.44792         
#                  Recall : 0.18696         
#                      F1 : 0.26380         
#              Prevalence : 0.34024         
#          Detection Rate : 0.06361         
#    Detection Prevalence : 0.14201         
#       Balanced Accuracy : 0.53406         
#                                           
#        'Positive' Class : TRUE            

tidy(recurSE)
#    term          estimate std.error statistic  p.value
#  1 (Intercept)    -0.638    0.886     -0.720  0.472   
#  2 studyNCI       -0.169    0.916     -0.185  0.853   
#  3 studyNKI        0.0589   0.626      0.0942 0.925   
#  4 studySTNO2     -0.814    0.833     -0.977  0.328   
#  5 studyTRANSBIG   0.455    0.632      0.719  0.472   
#  6 studyUCSF      -0.946    0.823     -1.15   0.250   
#  7 studyUNT        0.556    0.649      0.856  0.392   
#  8 studyUPP       -0.455    0.669     -0.681  0.496   
#  9 age_dx         -0.0235   0.00948   -2.48   0.0131  
# 10 grade           0.442    0.128      3.46   0.000544
# 11 N               0.776    0.371      2.09   0.0365  
# 12 erpositive     -0.0425   0.203     -0.209  0.834   

recurSE.a <- augment(recurSE, recurSE$model)

# try with batch instead of study
recurB <- glm(status ~ batch + age_dx + grade + N,
              family = binomial(link = logit), data = control)

summary(recurB)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.5960  -0.9333  -0.6066   1.1321   2.3128  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)   
# (Intercept)  -0.430191   0.877914  -0.490  0.62412   
# batchKIU      0.357434   0.688485   0.519  0.60365   
# batchNCI     -0.035787   0.922568  -0.039  0.96906   
# batchNKI      0.510860   0.660442   0.774  0.43922   
# batchNKI2    -0.351744   0.655161  -0.537  0.59135   
# batchOXFU     0.922922   0.695695   1.327  0.18463   
# batchSTNO2   -0.699792   0.815566  -0.858  0.39087   
# batchUCSF    -1.035498   0.832153  -1.244  0.21337   
# batchUPPU    -0.341547   0.680341  -0.502  0.61565   
# batchVDXGUYU  1.096663   0.711348   1.542  0.12315   
# batchVDXIGRU  0.599343   0.692722   0.865  0.38693   
# batchVDXKIU  -0.070544   0.703688  -0.100  0.92015   
# batchVDXOXFU  1.027782   0.752084   1.367  0.17176   
# batchVDXRHU   0.433628   0.711762   0.609  0.54237   
# age_dx       -0.027609   0.009564  -2.887  0.00389 **
# grade         0.377431   0.122416   3.083  0.00205 **
# N             1.140394   0.394989   2.887  0.00389 **
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 874.03  on 682  degrees of freedom
# Residual deviance: 794.96  on 666  degrees of freedom
# (583 observations deleted due to missingness)
# AIC: 828.96
# 
# Number of Fisher Scoring iterations: 4

1 - pchisq(794.96, 682)
# [1] 0.001738064

caret::confusionMatrix(table(predict(recurB, type = "response") >= 0.5,
                             recurB$model$status == "recurrence"),
                       positive = "TRUE", mode = "everything")
# FALSE TRUE
# FALSE   399  148
# TRUE     53   83
# 
# Accuracy : 0.7057        
# 95% CI : (0.67, 0.7397)
# No Information Rate : 0.6618        
# P-Value [Acc > NIR] : 0.008009      
# 
# Kappa : 0.2691        
# 
# Mcnemar's Test P-Value : 3.351e-11     
#                                         
#             Sensitivity : 0.3593        
#             Specificity : 0.8827        
#          Pos Pred Value : 0.6103        
#          Neg Pred Value : 0.7294        
#               Precision : 0.6103        
#                  Recall : 0.3593        
#                      F1 : 0.4523        
#              Prevalence : 0.3382        
#          Detection Rate : 0.1215        
#    Detection Prevalence : 0.1991        
#       Balanced Accuracy : 0.6210        
#                                         
#        'Positive' Class : TRUE          
                                        
tidy(recurB)
#    term         estimate std.error statistic p.value
#  1 (Intercept)   -0.430    0.878     -0.490  0.624  
#  2 batchKIU       0.357    0.688      0.519  0.604  
#  3 batchNCI      -0.0358   0.923     -0.0388 0.969  
#  4 batchNKI       0.511    0.660      0.774  0.439  
#  5 batchNKI2     -0.352    0.655     -0.537  0.591  
#  6 batchOXFU      0.923    0.696      1.33   0.185  
#  7 batchSTNO2    -0.700    0.816     -0.858  0.391  
#  8 batchUCSF     -1.04     0.832     -1.24   0.213  
#  9 batchUPPU     -0.342    0.680     -0.502  0.616  
# 10 batchVDXGUYU   1.10     0.711      1.54   0.123  
# 11 batchVDXIGRU   0.599    0.693      0.865  0.387  
# 12 batchVDXKIU   -0.0705   0.704     -0.100  0.920  
# 13 batchVDXOXFU   1.03     0.752      1.37   0.172  
# 14 batchVDXRHU    0.434    0.712      0.609  0.542  
# 15 age_dx        -0.0276   0.00956   -2.89   0.00389
# 16 grade          0.377    0.122      3.08   0.00205
# 17 N              1.14     0.395      2.89   0.00389

recurB.a <- augment(recurB, recurB$model)

recurBE <- glm(status ~ batch + age_dx + grade + N + er,
               family = binomial(link = logit), data = control)
summary(recurBE)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.5745  -0.9294  -0.6068   1.1298   2.3026  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)   
# (Intercept)  -0.406557   0.901767  -0.451  0.65210   
# batchKIU      0.351171   0.687606   0.511  0.60955   
# batchNCI     -0.054157   0.921539  -0.059  0.95314   
# batchNKI      0.497109   0.660278   0.753  0.45152   
# batchNKI2    -0.356017   0.653479  -0.545  0.58589   
# batchOXFU     1.034975   0.699963   1.479  0.13924   
# batchSTNO2   -0.899965   0.844542  -1.066  0.28659   
# batchUCSF    -1.001651   0.832305  -1.203  0.22880   
# batchUPPU    -0.359851   0.679169  -0.530  0.59622   
# batchVDXGUYU  1.077683   0.709940   1.518  0.12902   
# batchVDXIGRU  0.582160   0.691823   0.841  0.40008   
# batchVDXKIU  -0.087752   0.702627  -0.125  0.90061   
# batchVDXOXFU  1.006564   0.751489   1.339  0.18043   
# batchVDXRHU   0.415955   0.710427   0.586  0.55821   
# age_dx       -0.027091   0.009655  -2.806  0.00502 **
#   grade         0.368164   0.132383   2.781  0.00542 **
#   N             1.096522   0.398897   2.749  0.00598 **
#   erpositive   -0.012231   0.210939  -0.058  0.95376   
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 866.89  on 675  degrees of freedom
# Residual deviance: 788.18  on 658  degrees of freedom
# (590 observations deleted due to missingness)
# AIC: 824.18
# 
# Number of Fisher Scoring iterations: 4

1 - pchisq(788.18, 658)
# [1] 0.0003443863

caret::confusionMatrix(table(predict(recurBE, type = "response") >= 0.5,
                             recurBE$model$status == "recurrence"),
                       positive = "TRUE", mode = "everything")

#       FALSE TRUE
# FALSE   390  148
# TRUE     56   82
# 
# Accuracy : 0.6982          
# 95% CI : (0.6621, 0.7326)
# No Information Rate : 0.6598          
# P-Value [Acc > NIR] : 0.01851         
# 
# Kappa : 0.2557          
# 
# Mcnemar's Test P-Value : 1.875e-10       
#                                           
#             Sensitivity : 0.3565          
#             Specificity : 0.8744          
#          Pos Pred Value : 0.5942          
#          Neg Pred Value : 0.7249          
#               Precision : 0.5942          
#                  Recall : 0.3565          
#                      F1 : 0.4457          
#              Prevalence : 0.3402          
#          Detection Rate : 0.1213          
#    Detection Prevalence : 0.2041          
#       Balanced Accuracy : 0.6155          
#                                           
#        'Positive' Class : TRUE            
                                          
tidy(recurBE)
#    term         estimate std.error statistic p.value
#  1 (Intercept)   -0.407    0.902     -0.451  0.652  
#  2 batchKIU       0.351    0.688      0.511  0.610  
#  3 batchNCI      -0.0542   0.922     -0.0588 0.953  
#  4 batchNKI       0.497    0.660      0.753  0.452  
#  5 batchNKI2     -0.356    0.653     -0.545  0.586  
#  6 batchOXFU      1.03     0.700      1.48   0.139  
#  7 batchSTNO2    -0.900    0.845     -1.07   0.287  
#  8 batchUCSF     -1.00     0.832     -1.20   0.229  
#  9 batchUPPU     -0.360    0.679     -0.530  0.596  
# 10 batchVDXGUYU   1.08     0.710      1.52   0.129  
# 11 batchVDXIGRU   0.582    0.692      0.841  0.400  
# 12 batchVDXKIU   -0.0878   0.703     -0.125  0.901  
# 13 batchVDXOXFU   1.01     0.751      1.34   0.180  
# 14 batchVDXRHU    0.416    0.710      0.586  0.558  
# 15 age_dx        -0.0271   0.00965   -2.81   0.00502
# 16 grade          0.368    0.132      2.78   0.00542
# 17 N              1.10     0.399      2.75   0.00598
# 18 erpositive    -0.0122   0.211     -0.0580 0.954  

recurBE.a <- augment(recurBE, recurBE$model)

# compare with glance
library(rlang)

models <- c("recur", "recurE", "recurS", "recurSE", "recurB", "recurBE")

# recur = age_dx + grade + N, E = er, S = study, B = batch
bind_cols(model = models, map(syms(models), function(x) glance(eval(x))) %>% bind_rows())
#   model   null.deviance df.null logLik   AIC   BIC deviance df.residual  nobs
# 1 recur            874.     682  -415.  837.  856.     829.         679   683
# 2 recurE           867.     675  -412.  834.  857.     824.         671   676
# 3 recurS           874.     682  -406.  833.  883.     811.         672   683
# 4 recurSE          867.     675  -403.  829.  883.     805.         664   676
# 5 recurB           874.     682  -397.  829.  906.     795.         666   683 *
# 6 recurBE          867.     675  -394.  824.  905.     788.         658   676

# TODO:  outlier analysis

# get microarray data for control group
controlSet <- readRDS("data/metaGxBreast/mgxSet.rds")[unique(control$study)]

# not possible because of platform merge in original data?
# # fix platform variable in control
# stkPlatform <- read_delim("data/metaGxBreast/stkPlatform.tsv", delim = " 	") %>%
#   separate(2, into = c("patient", "platform"), sep = "-") %>%
#   arrange(patient) %>%
#   mutate(patient = paste0("STK_", str_remove(substr(patient, 3, 5), "^00|^0")))
# 
# left_join(select(pheno, study, sample_name) %>% filter(study == "STK"),
#   stkPlatform, by = c("sample_name" = "patient")) %>%
#   summarize(missing = is.na(gsm)) %>%
#   janitor::tabyl(missing)
# # missing   n   percent
# #   FALSE 132 0.5866667
# #    TRUE  93 0.4133333
# 
# annSets <- MetaGxBreast::loadBreastEsets(unique(control$study))
# # snapshotDate(): 2020-04-27
# # Ids with missing data: CAL, NCI, NKI, UCSF, METABRIC
# skt <- pData(annSets$esets$STK)
