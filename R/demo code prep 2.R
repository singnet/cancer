## demo code prep 2
library(tidyverse)

load("data/curatedBreastData/mlEsets.rdata")

# get infogan embeddings
files <- list.files("ml/InfoGANs_latent_codes/curatedBreastData/")
cig <-  map(files, ~ read_csv(paste0("ml/InfoGANs_latent_codes/curatedBreastData/", .))) %>%
  set_names(str_remove(files, ".csv"))
for(n in names(cig)) {
  names(cig[[n]])[-1] <- paste0("d", names(cig[[n]][-1]))
}

noNorm15es <- noNorm15es[, cig$codes_48$patient_ID]

# TODO: compute pam50 $ other subtypes
pdata15 <- pData(noNorm15es) %>%
  rownames_to_column("patient_ID") %>%
  filter(patient_ID %in% cig$codes_48$patient_ID)

# cluster embedding vectors
# library(cValid)
# intern <- clValid(cig$codes_48, nClust = 2:24, 
#                   clMethods = c("hierarchical","kmeans","pam"), validation = "internal")
# summary(intern) %>% kable() %>% kable_styling()

library(factoextra)
# best clusters 5 or 7
km_wss <- fviz_nbclust(cig$codes_48[, -1], kmeans, method = "wss", k.max = 24) +
  theme_minimal() + ggtitle("infoGAN 48 dimensional embedding vector\nwithin kmeans cluster sums of squares")

km_sil <- fviz_nbclust(cig$codes_48[, -1], kmeans, method = "silhouette", k.max = 24) +
  theme_minimal() + ggtitle("infoGAN 48 dimensional embedding vector\naverage silhouette kmeans clusters")

pam_wss <- fviz_nbclust(cig$codes_48[, -1], cluster::pam, method = "wss", k.max = 24) +
  theme_minimal() + ggtitle("infoGAN 48 dimensional embedding vector\nwithin PAM cluster sums of squares")

pam_sil <- fviz_nbclust(cig$codes_48[, -1], cluster::pam, method = "silhouette", k.max = 24) +
  theme_minimal() + ggtitle("infoGAN 48 dimensional embedding vector\naverage silhouette PAM clusters")

gap48 <- cluster::clusGap(cig$codes_48[, -1], FUN = kmeans, nstart = 25, K.max = 15, B = 100)
fviz_gap_stat(gap48) + theme_minimal() + ggtitle("infoGAN 48 dimensional embedding vector\ngap statistic kmeans clusters")

gap48_pam <- cluster::clusGap(cig$codes_48[, -1], FUN = cluster::pam, K.max = 15, B = 100)
fviz_gap_stat(gap48) + theme_minimal() + ggtitle("infoGAN 48 dimensional embedding vector\ngap statistic PAM clusters")

km48_3 <- kmeans(cig$codes_48[, -1], centers = 3, nstart = 100)
table(km48_3$cluster)
#   1    2    3 
# 593 1039  593 

km48_4 <- kmeans(cig$codes_48[, -1], centers = 4, nstart = 100)
table(km48_4$cluster)
#   1   2   3   4 
# 341 786 708 390 

km48_5 <- kmeans(cig$codes_48[, -1], centers = 5, nstart = 100)
table(km48_5$cluster)
#   1   2   3   4   5 
# 508 614 408 232 463 

km48_7 <- kmeans(cig$codes_48[, -1], centers = 7, nstart = 100)
table(km48_7$cluster)
#   1   2   3   4   5   6   7 
# 196 396 430 277 235 407 284 

km48_9 <- kmeans(cig$codes_48[, -1], centers = 9, nstart = 100)
table(km48_9$cluster)
#   1   2   3   4   5   6   7   8   9 
# 196  25 241 454 200 250 404 191 264 

pam48_5 <- cluster::pam(cig$codes_48[, -1], k = 5, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)
pam48_7 <- cluster::pam(cig$codes_48[, -1], k = 7, diss = FALSE, metric = "euclidean", pamonce = 5, trace.lev = 5)

comClust <- data.frame(patient_ID = as.character(cig$codes_48$patient_ID), k5 = paste0("k", km48_5$cluster), k7 = paste0("k", km48_7$cluster), k9 = paste0("k", km48_9$cluster), p5 = paste0("k", pam48_5$clustering), p7 = paste0("k", pam48_7$clustering))

pdata15 <- left_join(pdata15, comClust)

# save pam data for notebook
save(pam_wss, pam_sil, gap48_pam, file = "reports/set15clust.rdata")

# pc plot of embeddings
library(ggfortify)
for(n in names(cig)) {
  print(autoplot(prcomp(dplyr::select(cig[[n]], -1)), data = pdata15, colour = "gpl", title = n,
                 loadings = TRUE, loadings.colour = 'blue',
                 loadings.label = TRUE, loadings.label.size = 3))
}

# survival curves
library(survival)
pdata15$months2 <- coalesce(pdata15$RFS_months_or_MIN_months_of_RFS, pdata15$DFS_months_or_MIN_months_of_DFS)

pam50_fit <- survfit(Surv(months2, posOutcome2 == 0) ~ pam_coincide, data = pdata15)
autoplot(pam50_fit, surv.connect = FALSE)

p5_fit <- survfit(Surv(months2, posOutcome2 == 0) ~ p5, data = pdata15)
autoplot(p5_fit, surv.connect = FALSE)

p7_fit <- survfit(Surv(months2, posOutcome2) ~ p7, data = pdata15)
autoplot(p7_fit)

k5_fit <- survfit(Surv(months2, posOutcome2) ~ k5, data = pdata15)
autoplot(k5_fit)

k7_fit <- survfit(Surv(months2, posOutcome2) ~ k7, data = pdata15)
autoplot(k7_fit)

# save for notebook
write_csv(pdata15, "reports/coincideClinDat.csv")

# try survminer to get only one set of curves
library(survminer)
ggsurvplot(pam50_fit, data = pdata15, risk.table = TRUE, pval = TRUE, conf.int = TRUE)
ggsurvplot(p5_fit, data = pdata15, risk.table = TRUE, pval = TRUE, conf.int = TRUE)

# biplot alternate embeddings
pdata9893 <- rownames_to_column(pdata9893, "patient_ID") %>%
  mutate(patient_ID = as.character(patient_ID)) %>% 
  left_join(comClust) %>%
  column_to_rownames("patient_ID")
pData(gse9893es) <- pdata9893

s2_9893 <- esetVis::esetSpectralMap(gse9893es,
                                  title = "GSE9893 microarray data set\n BMC normalization",
                                  colorVar = "k5",  color = ggplotColours(5),
                                  shapeVar = "RFS", shape = c(1, 4),
                                  topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                  topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                  topSamplesJust = c(1, 0), topSamplesCex = 3,
                                  returnAnalysis = TRUE)
print(s2_9893$plot)

s2_9893_34 <- esetVis::esetSpectralMap(gse9893es,
                                     title = "GSE9893 microarray data set\n BMC normalization",
                                     colorVar = "k5", # color = colorPalette,
                                     shapeVar = "RFS", shape = c(1, 4),
                                     topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                     topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                     topSamplesJust = c(1, 0), topSamplesCex = 3,
                                     dim = 3:4, returnAnalysis = TRUE)
print(s2_9893_34$plot)

s3_9893 <- esetVis::esetSpectralMap(gse9893es,
                                    # title = "GSE9893 microarray data set\n BMC normalization",
                                    colorVar = "p5", # color = colorPalette,
                                    shapeVar = "RFS", shape = c(1, 4),
                                    topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                    topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                    topSamplesJust = c(1, 0), topSamplesCex = 3,
                                    returnAnalysis = TRUE)
print(s3_9893$plot)

s3_9893_34 <- esetVis::esetSpectralMap(gse9893es,
                                       # title = "GSE9893 microarray data set\n BMC normalization",
                                       colorVar = "p5", # color = colorPalette,
                                       shapeVar = "RFS", shape = c(1, 4),
                                       topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                       topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                       topSamplesJust = c(1, 0), topSamplesCex = 3,
                                       dim = 3:4, returnAnalysis = TRUE)
print(s3_9893_34$plot)

s4_9893 <- esetVis::esetSpectralMap(gse9893es,
                                    title = "GSE9893 microarray data set\n BMC normalization",
                                    colorVar = "k7",  color = ggplotColours(5),
                                    shapeVar = "RFS", shape = c(1, 4),
                                    topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                    topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                    topSamplesJust = c(1, 0), topSamplesCex = 3,
                                    returnAnalysis = TRUE)
print(s4_9893$plot)

s4_9893_34 <- esetVis::esetSpectralMap(gse9893es,
                                       title = "GSE9893 microarray data set\n BMC normalization",
                                       colorVar = "k7", # color = colorPalette,
                                       shapeVar = "RFS", shape = c(1, 4),
                                       topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                       topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                       topSamplesJust = c(1, 0), topSamplesCex = 3,
                                       dim = 3:4, returnAnalysis = TRUE)
print(s4_9893_34$plot)

s5_9893 <- esetVis::esetSpectralMap(gse9893es,
                                    title = "GSE9893 microarray data set\n BMC normalization",
                                    colorVar = "p7", # color = colorPalette,
                                    shapeVar = "RFS", shape = c(1, 4),
                                    topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                    topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                    topSamplesJust = c(1, 0), topSamplesCex = 3,
                                    returnAnalysis = TRUE)
print(s5_9893$plot)

s5_9893_34 <- esetVis::esetSpectralMap(gse9893es,
                                       title = "GSE9893 microarray data set\n BMC normalization",
                                       colorVar = "p7", # color = colorPalette,
                                       shapeVar = "RFS", shape = c(1, 4),
                                       topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                       topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                       topSamplesJust = c(1, 0), topSamplesCex = 3,
                                       dim = 3:4, returnAnalysis = TRUE)
print(s5_9893_34$plot)

# save for notebook
save(s3_9893, s3_9893_34, file = "reports/gse9893_p5.rdata")

# plot comparing pam50 & infogan k = 5
library(ggpubr)
ggarrange(s9893$plot, s3_9893$plot) %>%
  annotate_figure(top = text_grob("GSE9893 - PAM50 subtypes vs infoGAN PAM, k = 5", size = 15))

# redo gse9893 & gse20194 using combat normed expression values
combat15es2 <- combat15es
pData(combat15es2) <- bind_cols(pData(combat15es2)[comClust$patient_ID,], comClust)

gse20194cbt <- combat15es[, rownames(pData(gse20194es))]
s20194cbt <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194cbt, list(treatment_protocol_number = c(1, 5))),
                                      title = "GSE20194 breast cancer microarray data set\n COMBAT normalization",
                                      colorVar = "pam_coincide", # color = colorPalette,
                                      shapeVar = "pCR", shape = c(1, 4),
                                      # shapeVar = "channel_count", shape = 15:16,
                                      sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                      # symmetryAxes = "separate",
                                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                      topSamplesJust = c(1, 0), topSamplesCex = 3,
                                      returnAnalysis = TRUE)

print(s20194cbt$plot)

s20194cbt_34 <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194cbt, list(treatment_protocol_number = c(1, 5))),
                                         title = "GSE20194 breast cancer microarray data set\n COMBAT normalization",
                                         colorVar = "pam_coincide", # color = colorPalette,
                                         shapeVar = "pCR", shape = c(1, 4),
                                         # shapeVar = "channel_count", shape = 15:16,
                                         sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                         # symmetryAxes = "separate",
                                         topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                         topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                         topSamplesJust = c(1, 0), topSamplesCex = 3,
                                         dim = 3:4, returnAnalysis = TRUE)

print(s20194cbt_34$plot)

save(s20194cbt, s20194cbt_34, file = "reports/gse20194cbt_pam50.rdata")

gse20194cbti <- combat15es2[, rownames(pData(gse20194es))]
s20194cbti <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194cbti, list(treatment_protocol_number = c(1, 5))),
                                       title = "GSE20194 breast cancer microarray data set\n COMBAT normalization",
                                       colorVar = "p5", # color = colorPalette,
                                       shapeVar = "pCR", shape = c(1, 4),
                                       # shapeVar = "channel_count", shape = 15:16,
                                       sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                       # symmetryAxes = "separate",
                                       topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                       topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                       topSamplesJust = c(1, 0), topSamplesCex = 3,
                                       returnAnalysis = TRUE)

print(s20194cbti$plot)

s20194cbti_34 <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194cbti, list(treatment_protocol_number = c(1, 5))),
                                          title = "GSE20194 breast cancer microarray data set\n COMBAT normalization",
                                          colorVar = "p5", # color = colorPalette,
                                          shapeVar = "pCR", shape = c(1, 4),
                                          # shapeVar = "channel_count", shape = 15:16,
                                          sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                          # symmetryAxes = "separate",
                                          topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                          topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                          topSamplesJust = c(1, 0), topSamplesCex = 3,
                                          dim = 3:4, returnAnalysis = TRUE)

print(s20194cbti_34$plot)

save(s20194cbti, s20194cbti_34, file = "reports/gse20194cbt_infogan.rdata")

gse9893cbt <- combat15es[, rownames(pData(gse9893es))]
s9893cbt <- esetVis::esetSpectralMap(gse9893cbt,
                                     # title = "GSE9893 microarray data set\n COMBAT normalization",
                                     colorVar = "pam_coincide", # color = ggplotColours(5),
                                     shapeVar = "RFS", shape = c(1, 4),
                                     # shapeVar = "channel_count", shape = 15:16,
                                     # sizeVar = "age", sizeRange = c(2, 6),
                                     # symmetryAxes = "separate",
                                     topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                     topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                     topSamplesJust = c(1, 0), topSamplesCex = 3,
                                     returnAnalysis = TRUE)
print(s9893cbt$plot)

s9893cbt_34 <- esetVis::esetSpectralMap(gse9893cbt,
                                        # title = "GSE9893 microarray data set\n COMBAT normalization",
                                        colorVar = "pam_coincide", # color = colorPalette,
                                        shapeVar = "RFS", shape = c(1, 4),
                                        # shapeVar = "channel_count", shape = 15:16,
                                        # sizeVar = "age", sizeRange = c(2, 6),
                                        # symmetryAxes = "separate",
                                        topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                        topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                        topSamplesJust = c(1, 0), topSamplesCex = 3,
                                        dim = 3:4, returnAnalysis = TRUE)
print(s9893cbt_34$plot)

save(s9893cbt, s9893cbt_34, file = "reports/gse9893cbt_pam50.rdata")

gse9893cbti <- combat15es2[, rownames(pData(gse9893es))]

s9893cbti <- esetVis::esetSpectralMap(gse9893cbti,
                                      # title = "GSE9893 microarray data set\n COMBAT normalization",
                                      colorVar = "p5", # color = ggplotColours(5),
                                      shapeVar = "RFS", shape = c(1, 4),
                                      # shapeVar = "channel_count", shape = 15:16,
                                      # sizeVar = "age", sizeRange = c(2, 6),
                                      # symmetryAxes = "separate",
                                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                      topSamplesJust = c(1, 0), topSamplesCex = 3,
                                      returnAnalysis = TRUE)
print(s9893cbti$plot)

s9893cbti_34 <- esetVis::esetSpectralMap(gse9893cbti,
                                         # title = "GSE9893 microarray data set\n COMBAT normalization",
                                         colorVar = "p5", # color = colorPalette,
                                         shapeVar = "RFS", shape = c(1, 4),
                                         # shapeVar = "channel_count", shape = 15:16,
                                         # sizeVar = "age", sizeRange = c(2, 6),
                                         # symmetryAxes = "separate",
                                         topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                         topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                         topSamplesJust = c(1, 0), topSamplesCex = 3,
                                         dim = 3:4, returnAnalysis = TRUE)
print(s9893cbti_34$plot)

save(s9893cbti, s9893cbti_34, file = "reports/gse9893cbt_infogan.rdata")

save(combat15es2, gse20194cbt, gse20194cbti, gse9893cbt, gse9893cbti, file = "data/curatedBreastData/mlEsets2.rdata")

xtabs(~ pam_coincide + RFS, data = pData(gse9893es))
#                RFS
# pam_coincide  0  1
#       Basal   0  1
#       Her2   24 63
#       LumA   18  7
#       LumB   10 31
#       Normal  0  1

xtabs(~ p5 + RFS, data = pData(gse9893es))
#        RFS
# p5    0  1
#   k1  7  2
#   k2 11  5
#   k3 24 65
#   k4  7 10
#   k5  3 21
# 
xtabs(~ pam_coincide + RFS + ER_preTrt, data = pData(gse9893es))
# , , ER_preTrt = 0
# 
#                RFS
# pam_coincide  0  1
#       Basal   0  0
#       Her2    1  4
#       LumA    0  0
#       LumB    2  1
#       Normal  0  0
# 
# , , ER_preTrt = 1
# 
#                RFS
# pam_coincide  0  1
#       Basal   0  1
#       Her2   23 59
#       LumA   18  7
#       LumB    8 30
#       Normal  0  1

xtabs(~ pam_coincide + p5, data = pData(gse9893es))
#                          p5
# pam_coincide k1 k2 k3 k4 k5
#       Basal   0  0  0  1  0
#       Her2    0  0 62 13 12
#       LumA    9 13  3  0  0
#       LumB    0  3 24  3 11
#       Normal  0  0  0  0  1

# logistic regression models
gse9893p <- Biobase::pData(s3_9893$analysis$esetUsed) %>%
  as_tibble() %>%
  select(size = tumor_size_cm_preTrt_preSurgery, node = preTrt_lymph_node_status, tumor = tumor_stage_preTrt, grade = hist_grade, radio, age, er = ER_preTrt, pr = PR_preTrt, pam = pam_coincide, p5, RFS) %>%
  mutate(node = as.numeric(str_remove(node, "N")), tumor = as.numeric(str_remove(tumor, "T")),
         radio = as.numeric(radio)) %>%
  mutate(across(where(is.character), as.factor))

logreg <- glmulti::glmulti(RFS ~ age + node + radio + tumor + grade + size + er + pr + pam + p5, family = binomial(link = logit), data = gse9893p, crit = "bic", level = 1)

print(logreg)
# glmulti.analysis
# Method: h / Fitting: glm / IC used: bic
# Level: 1 / Marginality: FALSE
# From 100 models:
#   Best IC: 153.271306699477
# Best model:
#   [1] "RFS ~ 1 + node + grade + pr"
# Evidence weight: 0.165416382911834
# Worst IC: 165.739485846973
# 4 models within 2 IC units.
# 43 models to reach 95% of evidence weight.

glmAll <- function(obj) tibble(formula = obj@formulas, ic = obj@crits, K = obj@K) %>%
  mutate(formula_string = map(formula, ~ str_c(str_trim(deparse(.)), collapse = " "))) %>%
  unnest(formula_string)

logregSum <- glmAll(logreg) %>%
  mutate(model = map(formula, ~ glm(., family = binomial(link = logit), data = gse9893p))) %>%
  mutate(pred = map(model, ~ predict(., type = "response") >= 0.5)) %>%
  mutate(ref = map(model, ~ .$model$RFS == "1")) %>%
  mutate(cm = map2(pred, ref, ~ caret::confusionMatrix(data = as.factor(.x), reference = as.factor(.y)))) %>%
  mutate(overall = map(cm, ~ pluck(., "overall"))) %>%
  unnest_wider(overall)

select(logregSum, -4:-8) %>%
  filter(Accuracy == max(Accuracy))
#   formula      ic     K Accuracy Kappa AccuracyLower AccuracyUpper AccuracyNull AccuracyPValue McnemarPValue
# 1 <formula>  163.    11    0.835 0.603         0.760         0.893        0.677      0.0000302         0.136
filter(logregSum, Accuracy == max(Accuracy)) %>%
  pull(formula)
# [[1]]
# RFS ~ 1 + p5 + age + node + radio + tumor + grade + pr
# <environment: 0x55aef5c4a840>

# compare subtypes in best model
pam50 <- glm(RFS ~ 1 + pam + age + node + radio + tumor + grade + pr, family = binomial(link = logit), data = gse9893p)
summary(pam50)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.3550  -0.4391   0.4228   0.6634   1.9606  
# 
# Coefficients:
#               Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   16.41637 2399.54596   0.007 0.994541    
# pamHer2      -14.41458 2399.54481  -0.006 0.995207    
# pamLumA      -16.28990 2399.54486  -0.007 0.994583    
# pamLumB      -13.89100 2399.54488  -0.006 0.995381    
# pamNormal     -0.51578 3393.46875   0.000 0.999879    
# age            0.02456    0.03022   0.813 0.416346    
# node          -0.77732    0.52771  -1.473 0.140746    
# radio          0.47983    0.63315   0.758 0.448540    
# tumor         -0.62085    0.43883  -1.415 0.157127    
# grade         -1.61714    0.42872  -3.772 0.000162 ***
# pr             1.73606    0.67433   2.575 0.010038 *  
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 167.40  on 132  degrees of freedom
# Residual deviance: 113.04  on 122  degrees of freedom
# (22 observations deleted due to missingness)
# AIC: 135.04
# 
# Number of Fisher Scoring iterations: 15

pam50pred <- predict(pam50, type = "response") >= 0.5
pam50cm <- caret::confusionMatrix(data = as.factor(pam50pred),
                                  reference = as.factor(gse9893p$RFS[as.numeric(names(pam50pred))] == "1"))
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

autoplot(pam50)

igan <- glm(RFS ~ 1 + p5 + age + node + radio + tumor + grade + pr, family = binomial(link = logit), data = gse9893p)
summary(igan)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.9562  -0.4814   0.3823   0.6324   1.8595  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)  1.69659    2.96225   0.573   0.5668    
# p5k2        -1.12475    1.40751  -0.799   0.4242    
# p5k3         1.18284    1.28569   0.920   0.3576    
# p5k4         0.81701    1.38886   0.588   0.5564    
# p5k5         3.01974    1.58092   1.910   0.0561 .  
# age          0.03230    0.02927   1.103   0.2698    
# node        -0.74720    0.55012  -1.358   0.1744    
# radio        0.29553    0.65463   0.451   0.6517    
# tumor       -0.78755    0.45379  -1.735   0.0827 .  
# grade       -1.94127    0.48565  -3.997 6.41e-05 ***
# pr           1.61842    0.68650   2.357   0.0184 *  
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 167.40  on 132  degrees of freedom
# Residual deviance: 109.01  on 122  degrees of freedom
# (22 observations deleted due to missingness)
# AIC: 131.01
# 
# Number of Fisher Scoring iterations: 5

iganpred <- predict(igan, type = "response") >= 0.5
igancm <- caret::confusionMatrix(data = as.factor(iganpred),
                                  reference = as.factor(gse9893p$RFS[as.numeric(names(iganpred))] == "1"))
igancm
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

xtabs(~ treatment_protocol_number + pCR, data = pData(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5)))))
#                           pCR
# treatment_protocol_number   0   1
#                         1 113  43
#                         5  60   4

xtabs(~ pam_coincide + pCR, data = pData(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5)))))
#             pCR
# pam_coincide  0  1
#       Basal  21 21
#       Her2    6 10
#       LumA   76  3
#       LumB   57 11
#       Normal 13  2

xtabs(~ p5 + pCR, data = pData(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5)))))
#     pCR
# p5    0  1
#   k1 45 18
#   k2 32 18
#   k3 38  3
#   k4 46  3
#   k5 12  5

xtabs(~ p7 + pCR, data = pData(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5)))))
#     pCR
# p7    0  1
#   k1 41 17
#   k2 10  1
#   k3 33 20
#   k4 27  3
#   k5 39  1
#   k6 22  3
#   k7  1  2

xtabs(~ p5 + pCR + treatment_protocol_number, data = pData(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5)))))
# , , treatment_protocol_number = 1
# 
#     pCR
# p5    0  1
#   k1 31 17
#   k2 18 15
#   k3 25  3
#   k4 29  3
#   k5 10  5
# 
# , , treatment_protocol_number = 5
# 
# pCR
# p5    0  1
#   k1 14  1
#   k2 14  3
#   k3 13  0
#   k4 17  0
#   k5  2  0

# embedding models vs pam50 & coincide
pamClusters <- readRDS("data/curatedBreastData/embeding_vector_result_8genes/EmbeddingClusters.rds")
gse9893as <- rownames_to_column(Biobase::pData(gse9893es), "patient_id") %>%
  left_join(select(pamClusters$pam2178, 1:5), by = c("patient_id" = "patient_id"))
gse9893as <- select(gse9893as, size = tumor_size_cm_preTrt_preSurgery, node = preTrt_lymph_node_status, tumor = tumor_stage_preTrt, grade = hist_grade, radio, age, er = ER_preTrt, pr = PR_preTrt, pam = pam_coincide, p5, p7, c5, c7, c9, c20, RFS) %>%
  mutate(node = as.numeric(str_remove(node, "N")), tumor = as.numeric(str_remove(tumor, "T")),
         radio = as.numeric(radio)) %>%
  mutate(across(where(is.character), as.factor))
gene8 <- glm(RFS ~ 1 + c5 + age + node + radio + tumor + grade + pr, family = binomial(link = logit), data = gse9893as)
summary(gene8)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -2.3923  -0.6572   0.4447   0.7210   1.9986  
# 
# Coefficients:
#             Estimate Std. Error z value Pr(>|z|)    
# (Intercept)  0.59233    2.48105   0.239 0.811305    
# c5c3         1.04284    0.94161   1.108 0.268075    
# c5c5         1.51530    0.93244   1.625 0.104142    
# age          0.04007    0.02810   1.426 0.153892    
# node        -1.36067    0.75595  -1.800 0.071871 .  
# radio        0.79257    0.59645   1.329 0.183909    
# tumor       -1.37724    0.68019  -2.025 0.042889 *  
# grade       -1.40399    0.41781  -3.360 0.000779 ***
# pr           1.75550    0.63457   2.766 0.005668 ** 
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 167.4  on 132  degrees of freedom
# Residual deviance: 120.1  on 124  degrees of freedom
# (22 observations deleted due to missingness)
# AIC: 138.1
# 
# Number of Fisher Scoring iterations: 5

gene8pred <- predict(gene8, type = "response") >= 0.5
gene8cm <- caret::confusionMatrix(data = as.factor(gene8pred),
                                 reference = as.factor(gse9893as$RFS[as.numeric(names(gene8pred))] == "1"))
gene8cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction FALSE TRUE
#      FALSE    25    7
#      TRUE     18   83
# 
# Accuracy : 0.812           
# 95% CI : (0.7352, 0.8745)
# No Information Rate : 0.6767          
# P-Value [Acc > NIR] : 0.0003557       
# 
# Kappa : 0.5397          
# 
# Mcnemar's Test P-Value : 0.0455003       
#                                           
#             Sensitivity : 0.5814          
#             Specificity : 0.9222          
#          Pos Pred Value : 0.7813          
#          Neg Pred Value : 0.8218          
#              Prevalence : 0.3233          
#          Detection Rate : 0.1880          
#    Detection Prevalence : 0.2406          
#       Balanced Accuracy : 0.7518          
#                                           
#        'Positive' Class : FALSE           

autoplot(gene8)

gse9893as50 <- rownames_to_column(Biobase::pData(gse9893es), "patient_id") %>%
  left_join(select(pamClusters$pam2233, 1:5), by = c("patient_id" = "patient_id"))
gse9893as50 <- select(gse9893as50, size = tumor_size_cm_preTrt_preSurgery, node = preTrt_lymph_node_status, tumor = tumor_stage_preTrt, grade = hist_grade, radio, age, er = ER_preTrt, pr = PR_preTrt, pam = pam_coincide, p5, p7, c5, c7, c9, c20, RFS) %>%
  mutate(node = as.numeric(str_remove(node, "N")), tumor = as.numeric(str_remove(tumor, "T")),
         radio = as.numeric(radio)) %>%
  mutate(across(where(is.character), as.factor))

gene50 <- glm(RFS ~ 1 + c5 + age + node + radio + tumor + grade + pr, family = binomial(link = logit), data = gse9893as50)
summary(gene50)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -2.1612  -0.6316   0.4580   0.7494   2.0274  
# 
# Coefficients:
#               Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   16.25616 1625.93470   0.010 0.992023    
# c5c4         -15.37697 1625.93262  -0.009 0.992454    
# c5c5         -15.85664 1625.93276  -0.010 0.992219    
# age            0.04041    0.02781   1.453 0.146225    
# node          -1.43836    0.49546  -2.903 0.003695 ** 
# radio          0.57333    0.57339   1.000 0.317358    
# tumor         -0.75652    0.52070  -1.453 0.146250    
# grade         -1.40505    0.41805  -3.361 0.000777 ***
# pr             1.58931    0.62247   2.553 0.010673 *  
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 167.40  on 132  degrees of freedom
# Residual deviance: 121.39  on 124  degrees of freedom
# (22 observations deleted due to missingness)
# AIC: 139.39
# 
# Number of Fisher Scoring iterations: 15

gene50pred <- predict(gene50, type = "response") >= 0.5
gene50cm <- caret::confusionMatrix(data = as.factor(gene50pred),
                                  reference = as.factor(gse9893as$RFS[as.numeric(names(gene50pred))] == "1"))
gene50cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction FALSE TRUE
#      FALSE    22    6
#      TRUE     21   84
# 
# Accuracy : 0.797           
# 95% CI : (0.7186, 0.8617)
# No Information Rate : 0.6767          
# P-Value [Acc > NIR] : 0.001456        
# 
# Kappa : 0.4896          
# 
# Mcnemar's Test P-Value : 0.007054        
#                                           
#             Sensitivity : 0.5116          
#             Specificity : 0.9333          
#          Pos Pred Value : 0.7857          
#          Neg Pred Value : 0.8000          
#              Prevalence : 0.3233          
#          Detection Rate : 0.1654          
#    Detection Prevalence : 0.2105          
#       Balanced Accuracy : 0.7225          
#                                           
#        'Positive' Class : FALSE           

autoplot(gene50)

gene50_p7 <- glm(RFS ~ 1 + p7 + age + node + radio + tumor + grade + pr, family = binomial(link = logit), data = gse9893as50)
summary(gene50_p7)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -2.4717  -0.3948   0.2823   0.5489   1.6804  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)    1.44519    3.00259   0.481 0.630293    
# p7k2           2.06977    1.42118   1.456 0.145291    
# p7k3          -1.09549    1.44349  -0.759 0.447901    
# p7k4           1.62962    1.35317   1.204 0.228474    
# p7k5          -0.12300    1.38191  -0.089 0.929078    
# p7k6          17.62054 1934.36321   0.009 0.992732    
# p7k7          19.59270 3956.18058   0.005 0.996049    
# age            0.02801    0.03064   0.914 0.360662    
# node          -0.61507    0.57616  -1.068 0.285735    
# radio          0.58224    0.67044   0.868 0.385153    
# tumor         -0.86161    0.47376  -1.819 0.068961 .  
# grade         -1.97904    0.52751  -3.752 0.000176 ***
# pr             1.65045    0.77325   2.134 0.032809 *  
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 167.4  on 132  degrees of freedom
# Residual deviance: 101.0  on 120  degrees of freedom
# (22 observations deleted due to missingness)
# AIC: 127
# 
# Number of Fisher Scoring iterations: 16

gene50_p7pred <- predict(gene50_p7, type = "response") >= 0.5
gene50_p7cm <- caret::confusionMatrix(data = as.factor(gene50_p7pred),
                                   reference = as.factor(gse9893as$RFS[as.numeric(names(gene50_p7pred))] == "1"))
gene50_p7cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction FALSE TRUE
#      FALSE    25    8
#      TRUE     18   82
# 
# Accuracy : 0.8045          
# 95% CI : (0.7268, 0.8681)
# No Information Rate : 0.6767          
# P-Value [Acc > NIR] : 0.0007359       
# 
# Kappa : 0.5243          
# 
# Mcnemar's Test P-Value : 0.0775562       
#                                           
#             Sensitivity : 0.5814          
#             Specificity : 0.9111          
#          Pos Pred Value : 0.7576          
#          Neg Pred Value : 0.8200          
#              Prevalence : 0.3233          
#          Detection Rate : 0.1880          
#    Detection Prevalence : 0.2481          
#       Balanced Accuracy : 0.7463          
#                                           
#        'Positive' Class : FALSE           

autoplot(gene50_p7)

gene50_c7 <- glm(RFS ~ 1 + c7 + age + node + radio + tumor + grade + pr, family = binomial(link = logit), data = gse9893as50)
summary(gene50_c7)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -2.3172  -0.6075   0.4394   0.6922   1.7877  
# 
# Coefficients:
#             Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -0.30602    2.51109  -0.122 0.903004    
# c7c7         0.90093    0.70610   1.276 0.201982    
# age          0.03966    0.02778   1.428 0.153322    
# node        -1.19566    0.48439  -2.468 0.013572 *  
# radio        0.54898    0.57741   0.951 0.341728    
# tumor       -0.11209    0.53112  -0.211 0.832859    
# grade       -1.54953    0.42816  -3.619 0.000296 ***
# pr           1.67708    0.62414   2.687 0.007209 ** 
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 167.40  on 132  degrees of freedom
# Residual deviance: 121.27  on 125  degrees of freedom
# (22 observations deleted due to missingness)
# AIC: 137.27
# 
# Number of Fisher Scoring iterations: 5

gene50_c7pred <- predict(gene50_c7, type = "response") >= 0.5
gene50_c7cm <- caret::confusionMatrix(data = as.factor(gene50_c7pred),
                                      reference = as.factor(gse9893as$RFS[as.numeric(names(gene50_c7pred))] == "1"))
gene50_c7cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction FALSE TRUE
# FALSE    23    6
# TRUE     20   84
# 
# Accuracy : 0.8045          
# 95% CI : (0.7268, 0.8681)
# No Information Rate : 0.6767          
# P-Value [Acc > NIR] : 0.0007359       
# 
# Kappa : 0.5117          
# 
# Mcnemar's Test P-Value : 0.0107874       
#                                           
#             Sensitivity : 0.5349          
#             Specificity : 0.9333          
#          Pos Pred Value : 0.7931          
#          Neg Pred Value : 0.8077          
#              Prevalence : 0.3233          
#          Detection Rate : 0.1729          
#    Detection Prevalence : 0.2180          
#       Balanced Accuracy : 0.7341          
#                                           
#        'Positive' Class : FALSE           

autoplot(gene50_c7)

gene50_c9 <- glm(RFS ~ 1 + c9 + age + node + radio + tumor + grade + pr, family = binomial(link = logit), data = gse9893as50)
summary(gene50_c9)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -2.2023  -0.5621   0.4270   0.7197   1.9371  
# 
# Coefficients:
#               Estimate Std. Error z value Pr(>|z|)   
# (Intercept)   18.33029 1380.63710   0.013  0.98941   
# c9c6         -17.27696 1380.63451  -0.013  0.99002   
# c9c7         -16.98252 1380.63459  -0.012  0.99019   
# c9c8         -17.31431 1380.63458  -0.013  0.98999   
# age            0.03586    0.02775   1.292  0.19626   
# node          -1.58903    0.50895  -3.122  0.00180 **
# radio          0.59985    0.58838   1.019  0.30797   
# tumor         -0.70677    0.53156  -1.330  0.18365   
# grade         -1.41458    0.43108  -3.281  0.00103 **
# pr             1.41732    0.63971   2.216  0.02672 * 
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 167.4  on 132  degrees of freedom
# Residual deviance: 116.2  on 123  degrees of freedom
# (22 observations deleted due to missingness)
# AIC: 136.2
# 
# Number of Fisher Scoring iterations: 16

gene50_c9pred <- predict(gene50_c9, type = "response") >= 0.5
gene50_c9cm <- caret::confusionMatrix(data = as.factor(gene50_c9pred),
                                      reference = as.factor(gse9893as$RFS[as.numeric(names(gene50_c9pred))] == "1"))
gene50_c9cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction FALSE TRUE
#      FALSE    24    6
#      TRUE     19   84
# 
# Accuracy : 0.812           
# 95% CI : (0.7352, 0.8745)
# No Information Rate : 0.6767          
# P-Value [Acc > NIR] : 0.0003557       
# 
# Kappa : 0.5336          
# 
# Mcnemar's Test P-Value : 0.0163951       
#                                           
#             Sensitivity : 0.5581          
#             Specificity : 0.9333          
#          Pos Pred Value : 0.8000          
#          Neg Pred Value : 0.8155          
#              Prevalence : 0.3233          
#          Detection Rate : 0.1805          
#    Detection Prevalence : 0.2256          
#       Balanced Accuracy : 0.7457          
#                                           
#        'Positive' Class : FALSE           

autoplot(gene50_c9)

gene50_c20 <- glm(RFS ~ 1 + c20 + age + node + radio + tumor + grade + pr, family = binomial(link = logit), data = gse9893as50)
summary(gene50_c20)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -2.2045  -0.6092   0.4304   0.6527   1.8980  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   -0.70518    2.80006  -0.252 0.801162    
# c20c12        14.29019 1455.39771   0.010 0.992166    
# c20c13         0.32108    1.01087   0.318 0.750765    
# c20c14         0.30590    0.96204   0.318 0.750505    
# c20c20        -2.30839    1.43531  -1.608 0.107772    
# c20c6         -1.03448    1.09031  -0.949 0.342725    
# c20c8          0.31328    1.03740   0.302 0.762666    
# age            0.05319    0.02904   1.832 0.066997 .  
# node          -0.88350    0.81433  -1.085 0.277949    
# radio          0.68185    0.60355   1.130 0.258585    
# tumor         -0.65696    0.47197  -1.392 0.163935    
# grade         -1.53873    0.42917  -3.585 0.000337 ***
# pr             1.81532    0.69272   2.621 0.008778 ** 
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 167.40  on 132  degrees of freedom
# Residual deviance: 117.75  on 120  degrees of freedom
# (22 observations deleted due to missingness)
# AIC: 143.75
# 
# Number of Fisher Scoring iterations: 14

gene50_c20pred <- predict(gene50_c20, type = "response") >= 0.5
gene50_c20cm <- caret::confusionMatrix(data = as.factor(gene50_c20pred),
                                      reference = as.factor(gse9893as$RFS[as.numeric(names(gene50_c20pred))] == "1"))
gene50_c20cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction FALSE TRUE
#      FALSE    26    9
#      TRUE     17   81
# 
# Accuracy : 0.8045          
# 95% CI : (0.7268, 0.8681)
# No Information Rate : 0.6767          
# P-Value [Acc > NIR] : 0.0007359       
# 
# Kappa : 0.5304          
# 
# Mcnemar's Test P-Value : 0.1698105       
#                                           
#             Sensitivity : 0.6047          
#             Specificity : 0.9000          
#          Pos Pred Value : 0.7429          
#          Neg Pred Value : 0.8265          
#              Prevalence : 0.3233          
#          Detection Rate : 0.1955          
#    Detection Prevalence : 0.2632          
#       Balanced Accuracy : 0.7523          
#                                           
#        'Positive' Class : FALSE           

autoplot(gene50_c20)

