## logistic regression models
library(tidyverse)
pheno <- read_csv("data/metaGxBreast/metaGXcovarTable.csv.xz", guess_max = 4000)
dim(pheno)
# [1] 9717   33

# generate models on metagx samples
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


# death model
deathPheno <- glmulti::glmulti(vital_status ~ study + batch + gpl + age + grade +  N + grade:N + pam50, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 250 models:
# Best model: vital_status~1+study+batch+gpl+age+grade+N:grade
# Best model: vital_status~1+batch+gpl+age+grade+N:grade
# Crit= 3105.16769785665
# Mean crit= 3235.21362600431
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

deathPhenoR <- glmulti::glmulti(vital_status ~ study + batch + gpl + age + er + pgr + her2 + grade +   N + grade:N + pam50, family = binomial(link = logit), data = pbasic, crit = "bic")
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

deathPhenoTx <- glmulti::glmulti(vital_status ~ study + batch + gpl + age + chemo + hormone + grade +  N + grade:N + pam50, family = binomial(link = logit), data = pbasic, crit = "bic")
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

deathPhenoTxR <- glmulti::glmulti(vital_status ~ study + batch + gpl + age + er + chemo + hormone + grade +  N + grade:N + pam50, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 2100 models:
# Best model: vital_status~1+batch+gpl+er+chemo+age+N:grade
# Best model: vital_status~1+study+batch+gpl+er+chemo+age+N:grade
# Crit= 2749.4736915327
# Mean crit= 2771.74926606851


print(deathPhenoTxR)
# From 100 models:
# Best IC: 2749.4736915327
# Best model:
# [1] "vital_status ~ 1 + batch + gpl + er + chemo + age + N:grade"
# [1] "vital_status ~ 1 + study + batch + gpl + er + chemo + age + N:grade"
# Evidence weight: 0.252940187037279
# Worst IC: 2789.12559947283
# 4 models within 2 IC units.
# 14 models to reach 95% of evidence weight.

deathPhenoTxR2 <- glmulti::glmulti(vital_status ~ age + er + grade + N + chemo + hormone + grade:N + pam50, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 250 models:
# Best model: vital_status~1+er+hormone+pam50+age+N:grade
# Crit= 3461.87836047293
# Mean crit= 3540.56097399871

print(deathPhenoTxR2)
# From 100 models:
# Best IC: 3461.87836047293
# Best model:
# [1] "vital_status ~ 1 + er + hormone + pam50 + age + N:grade"
# Evidence weight: 0.745488759828933
# Worst IC: 3607.95446563751
# 1 models within 2 IC units.
# 3 models to reach 95% of evidence weight.

# repeat with recurrence_status as outcome
recurPheno <- glmulti::glmulti(recurrence_status ~ study + batch + gpl + age + grade +  N + grade:N + pam50, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 250 models:
# Best model: recurrence_status~1+gpl+age+grade+N:grade
# Crit= 1169.71557142853
# Mean crit= 1242.2765814601

print(recurPheno)
# ...
# From 100 models:
# Best IC: 1169.71557142853
# Best model:
# [1] "recurrence_status ~ 1 + gpl + age + grade + N:grade"
# Evidence weight: 0.450283595517677
# Worst IC: 1392.52130702981
# 2 models within 2 IC units.
# 3 models to reach 95% of evidence weight.

recurPhenoR <- glmulti::glmulti(recurrence_status ~ study + batch + gpl + age + er + pgr + her2 + grade +  N + grade:N + pam50, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 2100 models:
# Best model: recurrence_status~1+gpl+er+pgr+her2+N:grade
# Best model: recurrence_status~1+batch+gpl+er+pgr+her2+N:grade
# Best model: recurrence_status~1+study+batch+gpl+er+pgr+her2+N:grade
# Best model: recurrence_status~1+study+gpl+er+pgr+her2+N:grade
# Crit= 307.192004755517
# Mean crit= 314.85431835728

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

recurPhenoTx <- glmulti::glmulti(recurrence_status ~ study + batch + gpl + age + grade + N + chemo + hormone + grade:N + pam50, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 1050 models:
# Best model: recurrence_status~1+gpl+chemo+grade+N
# Crit= 785.000211995179
# Mean crit= 801.33567831792
# Completed.

print(recurPhenoTx)
# From 100 models:
# Best IC: 785.000211995179
# Best model:
# [1] "recurrence_status ~ 1 + gpl + chemo + grade + N"
# Evidence weight: 0.286554178684986
# Worst IC: 813.603430463537
# 3 models within 2 IC units.
# 15 models to reach 95% of evidence weight.

recurPhenoTxR <- glmulti::glmulti(recurrence_status ~ study + batch + gpl + age + er + grade + N + chemo + hormone + grade:N + pam50, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 2100 models:
# Best model: recurrence_status~1+gpl+er+chemo+N:grade
# Crit= 781.679845694414
# Mean crit= 792.20287204432

print(recurPhenoTxR)
# From 100 models:
# Best IC: 781.679845694414
# Best model:
# [1] "recurrence_status ~ 1 + gpl + er + chemo + N:grade"
# Evidence weight: 0.184236728509429
# Worst IC: 798.51753804812
# 5 models within 2 IC units.
# 28 models to reach 95% of evidence weight.

recurPhenoTxR2 <- glmulti::glmulti(recurrence_status ~ age + er + grade + N + chemo + hormone + grade:N + pam50, family = binomial(link = logit), data = pbasic, crit = "bic")
# After 250 models:
# Best model: recurrence_status~1+er+chemo+pam50+age+grade+N:grade
# Crit= 1618.25355307555
# Mean crit= 1635.32578205236

print(recurPhenoTxR2)
# From 100 models:
# Best IC: 1618.25355307555
# Best model:
# [1] "recurrence_status ~ 1 + er + chemo + pam50 + age + grade + N:grade"
# Evidence weight: 0.204290456675771
# Worst IC: 1646.42499062867
# 4 models within 2 IC units.
# 19 models to reach 95% of evidence weight.

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

deathPhenoTxR2Sum <- glmAll(deathPhenoTxR2) %>%
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

bestMods[c(1:5),]
# search formula_string                                                          samples   BIC Accuracy  null improvement
# 5      recurrence_status ~ 1 + batch + gpl + pam50 + age + N + N:grade             933 1228.    0.699 0.644        1.085
# 5      recurrence_status ~ 1 + study + batch + gpl + pam50 + age + N + N:grade     933 1228.    0.699 0.644        1.085
# 5      recurrence_status ~ 1 + study + batch + gpl + pam50 + age + N:grade         933 1221.    0.696 0.644        1.080
# 5      recurrence_status ~ 1 + batch + gpl + pam50 + age + N:grade                 933 1221.    0.696 0.644        1.080
# 7      recurrence_status ~ 1 + gpl + chemo + hormone + pam50 + age + N:grade       616  809.    0.679 0.630        1.077
