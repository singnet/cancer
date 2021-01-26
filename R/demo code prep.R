## demo
# get bc data
library(tidyverse)
combat15 <- read_csv("data/curatedBreastData/microarray data/merged-combat15.csv.xz")
bmc15 <- read_csv("data/curatedBreastData/microarray data/ex15bmcMerged.csv.xz")
pam50coincide <- read_csv("data/curatedBreastData/pam50/coincideTypes.csv")
ct <- read_csv("data/curatedBreastData/bcClinicalTable.csv", guess_max = 5000) %>%
  select(where(~ sum(is.na(.)) != 2719))
pam50centroids <- read_csv("data/curatedBreastData/pam50/pam50centroids.csv")

# make unnormalized set for comparison
cbc <- readRDS("data/curatedBreastData/bcProcessedEsetList.rds")
cbcMat <- Coincide::exprSetListToMatrixList(cbc)
names(cbcMat) <- names(cbc)
sapply(cbcMat, dim)
noNorm15 <- Coincide::mergeDatasetList(
  cbcMat,
  minNumGenes = 10000,
  minNumPatients = 40,
  batchNormalize = "none",
  NA_genesRemove = FALSE,
  outputFile = "data/curatedBreastData/microarray data/noNorm/noNorm15"
)

# transpose to samples in rows & save
noNorm15 <- bind_cols(patient_ID = colnames(noNorm15$mergedExprMatrix), t(noNorm15$mergedExprMatrix))
identical(names(combat15), names(noNorm15))
# [1] TRUE
identical(names(bmc15), names(noNorm15))
# [1] TRUE

write_csv(noNorm15, "./data/curatedBreastData/microarray data/ex15noNormMerged.csv.xz")
summary(colMeans(noNorm15[, -1]))
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 2.345   3.669   5.217   5.360   6.660  12.717 

summary(colMeans(bmc15[, -1]))
#       Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -4.100e-16 -4.811e-17 -1.508e-18 -1.097e-18  4.640e-17  3.134e-16 

summary(colMeans(combat15[, -1]))
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 2.345   3.669   5.218   5.360   6.661  12.718 

# compare esetVis - ualizations
# add subtyping data
pdata <- filter(ct, patient_ID %in% noNorm15$patient_ID) %>%
  select(where(~ sum(is.na(.)) != 2237)) %>%
  select(-4, -5, -7:-10, -13, -14) %>%
  arrange(match(patient_ID, noNorm15$patient_ID)) %>%
  mutate(channel_count = as.factor(channel_count)) %>%
  left_join(select(pam50coincide, - pam_cat)) %>%
  rename(pam_coincide = pam_name)

# add combined outcome variable
bmc15mldata1 <- read_csv("data/curatedBreastData/bmc15mldata1.csv") 

# these samples (from GSE9893 and GSE19615) are missing from merged data set
# TODO: were these removed as duplicages by coincide merge function?
setdiff(pdata$patient_ID, bmc15mldata1$patient_ID)
#[1] 249619 249623 249659 249670 249682 249683 250009 491205 491228 491285 491185 491188
pdata <- left_join(pdata, bmc15mldata1) %>%
  mutate(posOutcome2 = coalesce(RFS, DFS)) %>%
  mutate(radio = as.factor(radio), surgery = as.factor(surgery),chemo = as.factor(chemo), hormone = as.factor(hormone), pCR = as.factor(pCR), RFS = as.factor(RFS), DFS, posOutcome = as.factor(posOutcome), posOutcome2 = as.factor(posOutcome2))

# make esets
pdata <- column_to_rownames(pdata, "patient_ID")

noNorm15es <- t(noNorm15[, -1])
colnames(noNorm15es) <- noNorm15$patient_ID
noNorm15es <- Biobase::ExpressionSet(noNorm15es, Biobase::AnnotatedDataFrame(pdata))
bmc15es <- t(bmc15[, -1])
colnames(bmc15es) <- bmc15$patient_ID
bmc15es <- Biobase::ExpressionSet(bmc15es, Biobase::AnnotatedDataFrame(pdata))
combat15es <- t(combat15[, -1])
colnames(combat15es) <- combat15$patient_ID
combat15es <- Biobase::ExpressionSet(combat15es, Biobase::AnnotatedDataFrame(pdata))

## plot
# NOTE:  esetVis imports MASS which replaces dplyr "select"

# qc plots showing channel counts
non12 <- esetVis::esetSpectralMap(noNorm15es,
                      # title = "15 batch seperated breast cancer microarray data sets\n no normalization",
                      colorVar = "series_id", # color = colorPalette,
                      shapeVar = "channel_count", shape = c(1, 4),
                      # shapeVar = "channel_count", shape = 15:16,
                      # sizeVar = "age", sizeRange = c(2, 6),
                      # symmetryAxes = "separate",
                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                      topSamplesJust = c(1, 0), topSamplesCex = 3,
                      returnAnalysis = TRUE)


non34 <- esetVis::esetSpectralMap(noNorm15es,
                      # title = "15 batch seperated breast cancer microarray data sets\n no normalization",
                      colorVar = "series_id", # color = colorPalette,
                      shapeVar = "channel_count", shape = c(1, 4),
                      # sizeVar = "age", sizeRange = c(2, 6),
                      # symmetryAxes = "separate",
                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                      topSamplesJust = c(1, 0), topSamplesCex = 3,
                      dim = 3:4, returnAnalysis = TRUE)
      
bmc12 <- esetVis::esetSpectralMap(bmc15es,
                      # title = "15 batch seperated breast cancer microarray data sets\n BMC normalization",
                      colorVar = "series_id", # color = colorPalette,
                      shapeVar = "channel_count", shape = c(1, 4),
                      # sizeVar = "age", sizeRange = c(2, 6),
                      # symmetryAxes = "separate",
                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                      topSamplesJust = c(1, 0), topSamplesCex = 3,
                      returnAnalysis = TRUE)
      
bmc34 <- esetVis::esetSpectralMap(bmc15es,
                      # title = "15 batch seperated breast cancer microarray data sets\n BMC normalization",
                      colorVar = "series_id", # color = colorPalette,
                      shapeVar = "channel_count", shape = c(1, 4),
                      # sizeVar = "age", sizeRange = c(2, 6),
                      # symmetryAxes = "separate",
                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                      topSamplesJust = c(1, 0), topSamplesCex = 3,
                      dim = 3:4, returnAnalysis = TRUE)
      

com12 <- esetVis::esetSpectralMap(combat15es,
                      # title = "15 batch seperated breast cancer microarray data sets\n ComBat normalization",
                      colorVar = "series_id", # color = colorPalette,
                      shapeVar = "channel_count", shape = c(1, 4),
                      # sizeVar = "age", sizeRange = c(2, 6),
                      # symmetryAxes = "separate",
                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                      topSamplesJust = c(1, 0), topSamplesCex = 3,
                      returnAnalysis = TRUE)
      
com34 <- esetVis::esetSpectralMap(combat15es,
                      # title = "15 batch seperated breast cancer samples \n ComBat normalization",
                      colorVar = "series_id", # color = colorPalette,
                      shapeVar = "channel_count", shape = c(1, 4),
                      # sizeVar = "age", sizeRange = c(2, 6),
                      # symmetryAxes = "separate",
                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                      topSamplesJust = c(1, 0), topSamplesCex = 3,
                      dim = 3:4, returnAnalysis = TRUE)

esp <- c(non12, non34, bmc12, bmc34, com12, com34)
for(n in esp) print(n)

# save data for r notebook
save(non12, non34, bmc12, bmc34, com12, com34, file = "reports/set15specQC.rdata")


# plots showing outcome
non12_2 <- esetVis::esetSpectralMap(noNorm15es,
                                  title = "15 batch seperated breast cancer microarray data sets\n no normalization",
                                  colorVar = "series_id", # color = colorPalette,
                                  shapeVar = "posOutcome", shape = c(1, 4, 20),
                                  # shapeVar = "channel_count", shape = 15:16,
                                  # sizeVar = "age", sizeRange = c(2, 6),
                                  # symmetryAxes = "separate",
                                  topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                  topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                  topSamplesJust = c(1, 0), topSamplesCex = 3,
                                  returnAnalysis = TRUE)


non34_2 <- esetVis::esetSpectralMap(noNorm15es,
                                  title = "15 batch seperated breast cancer microarray data sets\n no normalization",
                                  colorVar = "series_id", # color = colorPalette,
                                  shapeVar = "posOutcome", shape = c(1, 4, 20),
                                  # sizeVar = "age", sizeRange = c(2, 6),
                                  # symmetryAxes = "separate",
                                  topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                  topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                  topSamplesJust = c(1, 0), topSamplesCex = 3,
                                  dim = 3:4, returnAnalysis = TRUE)

bmc12_2 <- esetVis::esetSpectralMap(bmc15es,
                                  title = "15 batch seperated breast cancer microarray data sets\n BMC normalization",
                                  colorVar = "series_id", # color = colorPalette,
                                  shapeVar = "posOutcome", shape = c(1, 4, 20),
                                  # sizeVar = "age", sizeRange = c(2, 6),
                                  # symmetryAxes = "separate",
                                  topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                  topSamplesJust = c(1, 0), topSamplesCex = 3,
                                  returnAnalysis = TRUE)

bmc34_2 <- esetVis::esetSpectralMap(bmc15es,
                                  title = "15 batch seperated breast cancer microarray data sets\n BMC normalization",
                                  colorVar = "series_id", # color = colorPalette,
                                  shapeVar = "posOutcome", shape = c(1, 4, 20),
                                  # sizeVar = "age", sizeRange = c(2, 6),
                                  # symmetryAxes = "separate",
                                  topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                  topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                  topSamplesJust = c(1, 0), topSamplesCex = 3,
                                  dim = 3:4, returnAnalysis = TRUE)


com12_2 <- esetVis::esetSpectralMap(combat15es,
                                  title = "15 batch seperated breast cancer microarray data sets\n ComBat normalization",
                                  colorVar = "series_id", # color = colorPalette,
                                  shapeVar = "posOutcome", shape = c(1, 4, 20),
                                  # sizeVar = "age", sizeRange = c(2, 6),
                                  # symmetryAxes = "separate",
                                  topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                  topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                  topSamplesJust = c(1, 0), topSamplesCex = 3,
                                  returnAnalysis = TRUE)

com34_2 <- esetVis::esetSpectralMap(combat15es,
                                  title = "15 batch seperated breast cancer samples \n ComBat normalization",
                                  colorVar = "series_id", # color = colorPalette,
                                  shapeVar = "posOutcome", shape = c(1, 4, 20),
                                  # sizeVar = "age", sizeRange = c(2, 6),
                                  # symmetryAxes = "separate",
                                  topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                  topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                  topSamplesJust = c(1, 0), topSamplesCex = 3,
                                  dim = 3:4, returnAnalysis = TRUE)

esp_2 <- c(non12_2, non34_2, bmc12_2, bmc34_2, com12_2, com34_2)
for(n in esp_2) print(n$plot)

# analyze individual studies gse9893 and gse20194 with bmc normalization
gse9893 <- read_csv("data/curatedBreastData/microarray data/example15bmc/study_9893_GPL5049_all-bmc15.csv.xz")
pdata9893 <- filter(pdata15, patient_ID %in% gse9893$patient_ID) %>%
  # mutate(RFS = as.logical(RFS)) %>%
  column_to_rownames("patient_ID")
gse9893es <- t(gse9893[, -1])
colnames(gse9893es) <- gse9893$patient_ID
gse9893es <- Biobase::ExpressionSet(gse9893es, Biobase::AnnotatedDataFrame(pdata9893))
s9893 <- esetVis::esetSpectralMap(gse9893es,
                               # title = "GSE9893 microarray data set\n BMC normalization",
                               colorVar = "pam_coincide", # color = ggplotColours(5),
                               shapeVar = "RFS", shape = c(1, 4),
                               # shapeVar = "channel_count", shape = 15:16,
                               # sizeVar = "age", sizeRange = c(2, 6),
                               # symmetryAxes = "separate",
                               topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                               topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                               topSamplesJust = c(1, 0), topSamplesCex = 3,
                               returnAnalysis = TRUE)
print(s9893$plot)

s9893_34 <- esetVis::esetSpectralMap(gse9893es,
                                  # title = "GSE9893 microarray data set\n BMC normalization",
                                  colorVar = "pam_coincide", # color = colorPalette,
                                  shapeVar = "RFS", shape = c(1, 4),
                                  # shapeVar = "channel_count", shape = 15:16,
                                  # sizeVar = "age", sizeRange = c(2, 6),
                                  # symmetryAxes = "separate",
                                  topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                  topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                  topSamplesJust = c(1, 0), topSamplesCex = 3,
                                  dim = 3:4, returnAnalysis = TRUE)
print(s9893_34$plot)

save(s9893, s9893_34, file = "reports/gse9893_pam50.rdata")

gse20194 <- read_csv("data/curatedBreastData/microarray data/example15bmc/study_20194_GPL96_all-bmc15.csv.xz")
pdata20194 <- filter(pdata15, patient_ID %in% gse20194$patient_ID) %>%
  # mutate(pCR = as.logical(pCR)) %>%
  column_to_rownames("patient_ID")
gse20194es <- t(gse20194[, -1])
colnames(gse20194es) <- gse20194$patient_ID
gse20194es <- Biobase::ExpressionSet(gse20194es, Biobase::AnnotatedDataFrame(pdata20194))
s20194 <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5))),
                               title = "GSE20194 breast cancer microarray data set\n BMC normalization",
                               colorVar = "pam_coincide", # color = colorPalette,
                               shapeVar = "pCR", shape = c(1, 4),
                               # shapeVar = "channel_count", shape = 15:16,
                               sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                               # symmetryAxes = "separate",
                               topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                               topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                               topSamplesJust = c(1, 0), topSamplesCex = 3,
                               returnAnalysis = TRUE)

print(s20194$plot)

s20194_34 <- esetVis::esetSpectralMap(GOexpress::subEset(gse20194es, list(treatment_protocol_number = c(1, 5))),
                                   title = "GSE20194 breast cancer microarray data set\n BMC normalization",
                                   colorVar = "pam_coincide", # color = colorPalette,
                                   shapeVar = "pCR", shape = c(1, 4),
                                   # shapeVar = "channel_count", shape = 15:16,
                                   sizeVar = "treatment_protocol_number", sizeRange = c(2, 4),
                                   # symmetryAxes = "separate",
                                   topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                                   topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                   topSamplesJust = c(1, 0), topSamplesCex = 3,
                                   dim = 3:4, returnAnalysis = TRUE)

print(s20194_34$plot)

save(bmc15es, combat15es, noNorm15es, gse9893es, gse20194es, file = "data/curatedBreastData/mlEsets.rdata")


broman::manyboxplot(Biobase::exprs(noNorm15es))
broman::manyboxplot(Biobase::exprs(bmc15es))
broman::manyboxplot(Biobase::exprs(combat15es))
broman::manyboxplot(Biobase::exprs(gse9893es))
broman::manyboxplot(Biobase::exprs(gse20194es))
broman::manyboxplot(Biobase::exprs(gse9893cbt))
broman::manyboxplot(Biobase::exprs(gse20194cbt))

