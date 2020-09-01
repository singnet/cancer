## logistic regression models
library(tidyverse)
library(glmulti)
pheno <- read_csv("data/metaGxBreast/metaGXcovarTable.csv.xz", guess_max = 4000)
dim(pheno)
# [1] 9717   33

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

# fix gpl
pheno$gpl[pheno$study == "UNC4"] <- "unc4"
pheno$gpl[pheno$study == "METABRIC"] <- "metabric"
pheno$gpl[pheno$study == "TCGA"] <- "tcga"
pheno$gpl[pheno$study == "VDX"] <- "GPL96"

pbasic <- filter(pheno, replicate != TRUE) %>%
  select(study, batch, sample_name, age = age_at_initial_pathologic_diagnosis, grade, N, er, pgr, her2, Platform, gpl, vital_status, recurrence_status, treatment, pam50) %>%
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
table(pbasic$her2)
# negative positive 
# 2897      803 
table(pbasic$er)
# negative positive 
# 2367     5479

table(pbasic$gpl, exclude = NULL)
#        GPL10379         GPL1352        GPL14374          GPL180         GPL3883         GPL4819          GPL570 
#             623             154             143             118             105             150            1119 
# GPL6106,GPL6986         GPL6244         GPL6486         GPL8300           GPL96     GPL96,GPL97        metabric 
#              75             318             152             169            1755             410            2114 
#            tcga            unc4            <NA> 
#            1073             305             925 


table(pbasic$pam50, exclude = NULL)
# Basal   Her2   LumA   LumB Normal   <NA> 
#  2007   1342   2752   2986    586     35 

# death model
deathPheno <- glmulti::glmulti(vital_status ~ study + batch + gpl + age + grade + N + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 150 models:
# Best model: vital_status~1+study+batch+gpl+age+grade+N:grade
# Best model: vital_status~1+batch+gpl+age+grade+N:grade
# Crit= 3105.16769785665
# Mean crit= 3515.74239333462
# Completed.
print(deathPheno)
# ...
# From 100 models:
# Best IC: 3105.16769785665
# Best model:
# [1] "vital_status ~ 1 + study + batch + gpl + age + grade + N:grade"
# [1] "vital_status ~ 1 + batch + gpl + age + grade + N:grade"
# Evidence weight: 0.436240761718023
# Worst IC: 3996.23943029271
# 2 models within 2 IC units.
# 5 models to reach 95% of evidence weight.

deathPhenoR <- glmulti::glmulti(vital_status ~ study + batch + gpl + age + er + pgr + her2 + grade + N + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 1050 models:
# Best model: vital_status~1+study+batch+gpl+er+pgr+her2+age+N:grade
# Best model: vital_status~1+gpl+er+pgr+her2+age+N:grade
# Best model: vital_status~1+study+gpl+er+pgr+her2+age+N:grade
# Best model: vital_status~1+batch+gpl+er+pgr+her2+age+N:grade
# Crit= 271.172488312395
# Mean crit= 279.046836525694

print(deathPhenoR)Level: 2 / Marginality: FALSE
# From 100 models:
# Best IC: 271.172488312395
# Best model:
# [1] "vital_status ~ 1 + study + batch + gpl + er + pgr + her2 + age + "
# [2] "    N:grade"                                                      
# [1] "vital_status ~ 1 + gpl + er + pgr + her2 + age + N:grade"
# [1] "vital_status ~ 1 + study + gpl + er + pgr + her2 + age + N:grade"
# [1] "vital_status ~ 1 + batch + gpl + er + pgr + her2 + age + N:grade"
# Evidence weight: 0.100295641924876
# Worst IC: 282.979409554131
# 8 models within 2 IC units.
# 44 models to reach 95% of evidence weight.

deathPhenoTx <- glmulti::glmulti(vital_status ~ study + batch + gpl + age + chemo + hormone + grade + N + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 500 models:
# Best model: vital_status~1+batch+gpl+chemo+age+grade+N:grade
# Best model: vital_status~1+study+batch+gpl+chemo+age+grade+N:grade
# Crit= 2780.34613511761
# Mean crit= 2858.387448018

print(deathPhenoTx)
# From 100 models:
# Best IC: 2780.34613511761
# Best model:
# [1] "vital_status ~ 1 + batch + gpl + chemo + age + grade + N:grade"
# [1] "vital_status ~ 1 + study + batch + gpl + chemo + age + grade + N:grade"           
# Evidence weight: 0.203998001137023
# Worst IC: 2888.76169609257
# 4 models within 2 IC units.
# 12 models to reach 95% of evidence weight.

deathPhenoTxR <- glmulti::glmulti(vital_status ~ study + batch + gpl + age + er + chemo + hormone + grade + N + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 1050 models:
# Best model: vital_status~1+batch+gpl+er+chemo+age+N:grade
# Best model: vital_status~1+study+batch+gpl+er+chemo+age+N:grade
# Crit= 2749.4736915327
# Mean crit= 2792.71886772045

print(deathPhenoTxR)
# From 100 models:
# Best IC: 2749.4736915327
# Best model:
# [1] "vital_status ~ 1 + batch + gpl + er + chemo + age + N:grade"
# [1] "vital_status ~ 1 + study + batch + gpl + er + chemo + age + N:grade"
# Evidence weight: 0.276062758977648
# Worst IC: 2833.82511735616
# 4 models within 2 IC units.
# 11 models to reach 95% of evidence weight.

recurPheno <- glmulti::glmulti(recurrence_status ~ study + batch + gpl + age + grade + N + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 150 models:
# Best model: recurrence_status~1+gpl+age+grade+N:grade
# Crit= 1169.71557142853
# Mean crit= 1550.52588211423

print(recurPheno)
# ...
# From 100 models:
# Best IC: 1169.71557142853
# Best model:
# [1] "recurrence_status ~ 1 + gpl + age + grade + N:grade"
# Evidence weight: 0.450835168441691
# Worst IC: 2070.43891244668
# 2 models within 2 IC units.
# 3 models to reach 95% of evidence weight.

recurPhenoR <- glmulti::glmulti(recurrence_status ~ study + batch + gpl + age + er + pgr + her2 + grade + N + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 1050 models:
# Best model: recurrence_status~1+gpl+er+pgr+her2+N:grade
# Best model: recurrence_status~1+batch+gpl+er+pgr+her2+N:grade
# Best model: recurrence_status~1+study+batch+gpl+er+pgr+her2+N:grade
# Best model: recurrence_status~1+study+gpl+er+pgr+her2+N:grade
# Crit= 307.192004755517
# Mean crit= 315.651761000601

print(recurPhenoR)
# ...
# From 100 models:
# Best IC: 307.192004755517
# Best model:
# [1] "recurrence_status ~ 1 + gpl + er + pgr + her2 + N:grade"
# [1] "recurrence_status ~ 1 + batch + gpl + er + pgr + her2 + N:grade"
# [1] "recurrence_status ~ 1 + study + batch + gpl + er + pgr + her2 + N:grade"
# [1] "recurrence_status ~ 1 + study + gpl + er + pgr + her2 + N:grade"
# Evidence weight: 0.089067168648353
# Worst IC: 323.45508141455
# 8 models within 2 IC units.
# 47 models to reach 95% of evidence weight.

recurPhenoTx <- glmulti::glmulti(recurrence_status ~ study + batch + gpl + age + grade + N + chemo + hormone + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 500 models:
# Best model: recurrence_status~1+gpl+chemo+grade+N
# Crit= 785.000211995179
# Mean crit= 807.896582782132

print(recurPhenoTx)
# From 100 models:
# Best IC: 785.000211995179
# Best model:
# [1] "recurrence_status ~ 1 + gpl + chemo + grade + N"
# Evidence weight: 0.286685313091568
# Worst IC: 828.556121155249
# 3 models within 2 IC units.
# 15 models to reach 95% of evidence weight.

recurPhenoTxR <- glmulti::glmulti(recurrence_status ~ study + batch + gpl + age + er + grade + N + chemo + hormone + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 1050 models:
# Best model: recurrence_status~1+gpl+er+chemo+N:grade
# Crit= 781.679845694414
# Mean crit= 792.462545229538

print(recurPhenoTxR)
# From 100 models:
# Best IC: 781.679845694414
# Best model:
# [1] "recurrence_status ~ 1 + gpl + er + chemo + N:grade"
# Evidence weight: 0.184428436287965
# Worst IC: 800.585478792299
# 5 models within 2 IC units.
# 28 models to reach 95% of evidence weight.

recurPhenoTxR2 <- glmulti::glmulti(recurrence_status ~ age + er + grade + N + chemo + hormone + grade:N, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 150 models:
# Best model: recurrence_status~1+er+chemo+age+grade+N:grade
# Crit= 1622.95267387007
# Mean crit= 1717.53015601479

print(recurPhenoTxR2)
# From 100 models:
# Best IC: 1622.95267387007
# Best model:
# [1] "recurrence_status ~ 1 + er + chemo + age + grade + N:grade"
# Evidence weight: 0.310282873486889
# Worst IC: 2015.95011354683
# 2 models within 2 IC units.
# 8 models to reach 95% of evidence weight.

# turn glmulti object into tibble and add prediction
# function to pull summary info for all formulas from glmulti object
glmAll <- function(obj) tibble(formula = obj@formulas, ic = obj@crits, K = obj@K) %>%
  mutate(formula_string = map(formula, ~ str_c(str_trim(deparse(.)), collapse = " "))) %>%
  unnest(formula_string)

recurPhenoSum <- glmAll(recurPheno) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = pbasic))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$recurrence_status == "recurrence")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

recurPhenoRSum <- glmAll(recurPhenoR) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = pbasic))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$recurrence_status == "recurrence")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

recurPhenoTxSum <- glmAll(recurPhenoTx) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = pbasic))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$recurrence_status == "recurrence")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

recurPhenoTxRSum <- glmAll(recurPhenoTxR) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = pbasic))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$recurrence_status == "recurrence")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

recurPhenoTxRSum2 <- glmAll(recurPhenoTxR2) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = pbasic))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$recurrence_status == "recurrence")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

deathPhenoSum <- glmAll(deathPheno) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = pbasic))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$vital_status == "deceased")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

deathPhenoRSum <- glmAll(deathPhenoR) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = pbasic))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$vital_status == "deceased")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

deathPhenoTxSum <- glmAll(deathPhenoTx) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = pbasic))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$vital_status == "deceased")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

deathPhenoTxRSum <- glmAll(deathPhenoTxR) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = pbasic))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$vital_status == "deceased")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

bestMods <- map(ls(pattern = "Sum$"), ~ eval(sym(.))) %>%
  map(~ filter(., AccuracyPValue < 0.05)) %>%
  map(~ select(., formula_string, samples = ref, BIC = ic, Accuracy, null = AccuracyNull)) %>%
  map(~ mutate(., improvement = Accuracy / null)) %>%
  # map(~ mutate(., samples = length(map(samples, unlist)))) %>%
  bind_rows(.id = "search") %>%
  arrange(desc(improvement), BIC)

# why doesn't this work in pipe?
bestMods$samples <- unlist(map(bestMods$samples, length))

bestMods[c(51, 31, 34, 17, 10, 8, 1),]
# search formula_string                                                           samples   BIC Accuracy  null  improvement
# 6 recurrence_status ~ 1 + batch + grade + N                                        1617 2067.    0.678 0.653        1.039
# 7 recurrence_status ~ 1 + gpl + chemo + hormone + age + grade + N                   616  794.    0.664 0.630        1.054
# 8 recurrence_status ~ 1 + gpl + chemo + hormone + grade + N + N:grade               616  795.    0.664 0.630        1.054
# 7 recurrence_status ~ 1 + gpl + er + chemo + hormone + N:grade                      607  783.    0.662 0.626        1.058
# 8 recurrence_status ~ 1 + batch + gpl + chemo + hormone + grade + N:grade           616  823.    0.672 0.630        1.067
# 8 recurrence_status ~ 1 + batch + gpl + chemo + hormone + age + grade + N:grade     616  829.    0.674 0.630        1.070
# 8 recurrence_status ~ 1 + batch + gpl + chemo + hormone + age + N:grade             616  827.    0.677 0.630        1.075
