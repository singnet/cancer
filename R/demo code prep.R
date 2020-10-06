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
# make esets
pdata <- filter(ct, patient_ID %in% noNorm15$patient_ID) %>%
  select(where(~ sum(is.na(.)) != 2237)) %>%
  select(-4, -5, -7:-10, -13, -14) %>%
  arrange(match(patient_ID, noNorm15$patient_ID)) %>%
  mutate(channel_count = as.factor(channel_count)) %>%
  left_join(select(pam50coincide, - pam_cat)) %>%
  rename(pam_coincide = pam_name)

# add subtyping data
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
library(esetVis)

print(esetSpectralMap(noNorm15es,
                      title = "15 batch seperated breast cancer microarray data sets\n no normalization",
                      colorVar = "series_id", # color = colorPalette,
                      shapeVar = "channel_count", shape = 15:16,
                      # shapeVar = "channel_count", shape = 15:16,
                      # sizeVar = "age", sizeRange = c(2, 6),
                      # symmetryAxes = "separate",
                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                      topSamplesJust = c(1, 0), topSamplesCex = 3)
)

print(esetSpectralMap(noNorm15es,
                      title = "15 batch seperated breast cancer microarray data sets\n no normalization",
                      colorVar = "series_id", # color = colorPalette,
                      shapeVar = "channel_count", shape = 15:16,
                      # sizeVar = "age", sizeRange = c(2, 6),
                      # symmetryAxes = "separate",
                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                      topSamplesJust = c(1, 0), topSamplesCex = 3,
                      dim = 3:4)
      )

print(esetSpectralMap(bmc15es,
                      title = "15 batch seperated breast cancer microarray data sets\n BMC normalization",
                      colorVar = "series_id", # color = colorPalette,
                      shapeVar = "channel_count", shape = 15:16,
                      # sizeVar = "age", sizeRange = c(2, 6),
                      # symmetryAxes = "separate",
                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                      topSamplesJust = c(1, 0), topSamplesCex = 3)
)

print(esetSpectralMap(bmc15es,
                      title = "15 batch seperated breast cancer microarray data sets\n BMC normalization",
                      colorVar = "series_id", # color = colorPalette,
                      shapeVar = "channel_count", shape = 15:16,
                      # sizeVar = "age", sizeRange = c(2, 6),
                      # symmetryAxes = "separate",
                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                      topSamplesJust = c(1, 0), topSamplesCex = 3,
                      dim = 3:4)
)

print(esetSpectralMap(combat15es,
                      title = "15 batch seperated breast cancer microarray data sets\n ComBat normalization",
                      colorVar = "series_id", # color = colorPalette,
                      shapeVar = "channel_count", shape = 15:16,
                      # sizeVar = "age", sizeRange = c(2, 6),
                      # symmetryAxes = "separate",
                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                      topSamplesJust = c(1, 0), topSamplesCex = 3)
)

print(esetSpectralMap(combat15es,
                      title = "15 batch seperated breast cancer samples \n ComBat normalization",
                      colorVar = "series_id", # color = colorPalette,
                      shapeVar = "channel_count", shape = 15:16,
                      # sizeVar = "age", sizeRange = c(2, 6),
                      # symmetryAxes = "separate",
                      topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                      topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                      topSamplesJust = c(1, 0), topSamplesCex = 3,
                      dim = 3:4)
)

# TODO: compute pam50 $ other subtypes
# get infogan embeddings
files <- list.files("ml/InfoGANs_latent_codes/curatedBreastData/")
cig <-  map(files, ~ read_csv(paste0("ml/InfoGANs_latent_codes/curatedBreastData/", .))) %>%
  set_names(str_remove(files, ".csv"))

# make table of info on 15 studies
bcStudies <- names(cbc) %>%
  str_split("_", n = 4) %>%
  transpose() %>%
  map(unlist) %>%
  set_names(c("x", "gse", "gpl", "batch")) %>%
  as_tibble() %>%
  select(- x) %>%
  mutate(gse = paste0("GSE", gse), batch = na_if(batch, "all"))

# get sample & feature counts
bcCounts <- map(cbcMat, dim) %>%
  transpose() %>%
  map(unlist) %>%
  set_names(c("n genes", "n samples")) %>%
  as_tibble()
bcStudies <- bind_cols(bcStudies, bcCounts)

# use geometadb to find additional for geo studies
# GEOmetadb::getSQLiteFile("/mnt/biodata/GEO")
con <- DBI::dbConnect(RSQLite::SQLite(), "/mnt/biodata/GEO/GEOmetadb.sqlite")
gse <- tbl(con, "gse") %>%
  filter(gse %in% !!bcStudies$gse) %>%
  select(gse, title, pubmed_id, summary, web_link, overall_design, repeats, repeats_sample_list)
bcStudies <- left_join(bcStudies, gse, copy = TRUE)
DBI::dbDisconnect(con)

# add channel counts
bcChannelCounts <- read_csv("data/curatedBreastData/bcStudyChannelCount.csv") %>%
  select(-2) %>%
  transmute(gse = str_remove(study_ID, ",GSE25066|GSE16716,"), channel_count = channel_count) %>%
  distinct()
bcStudies <- left_join(bcStudies, bcChannelCounts)

# save complete table (repeats are all NA)
bcStudies <- select(bcStudies, - repeats, - repeats_sample_list)
write_csv(bcStudies, "data/curatedBreastData/bcStudiesInfo.csv")

# make markup table of 15 merged studies for README
bc15studies <- filter(bcStudies, gse %in% (pull(pdata, series_id) %>% str_remove(",GSE25066|GSE16716,"))) %>%
  filter(`n samples` >= 40) %>%
  mutate(pubmed_id = paste0("[", pubmed_id, "](https://pubmed.ncbi.nlm.nih.gov/", pubmed_id, "/)"))
bc15studies <- bc15studies[, c(1, 6:8, 10, 2:5, 11)]

# convert to markdown for pasting in to readme
#  merge batch lines (GSE17705 & GSE25065) by hand and add link to GSE20194 summary - http://edkb.fda.gov/MAQC/
knitr::kable(bc15studies, "pipe")
