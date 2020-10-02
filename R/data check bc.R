### data checkout
## breast cancer
library(curatedBreastData)

## characterize available data
library(readr)

# expand and save variable list
data("clinicalData")
cvd <- clinicalData$clinicalVarDef

# columns added by hand to facilitate subsetting
# varType:  study, patient, state, treatment
# time:  preTx, postTx
cvd <- read_csv("data/curatedBreastData/bcVarDesc.csv")

# save clinical data table
# note: table updated with channel_count labels in 'get geo channel counts.R'
ct <- clinicalData$clinicalTable
write_csv(ct, "data/curatedBreastData/bcClinicalTable.csv")

# TODO: check, this should be wrong
# how many patients have > 1 sample?
summary(table(ct$original_study_patient_ID) > 1)
#    Mode   FALSE    TRUE 
# logical    2613      40

# which studies have patients with > 1 sample?
table(ct$study_ID[table(ct$original_study_patient_ID) > 1])
# 2034  6577 32646 
#   22     7    18 

# how many patients died?
sum(ct$OS[!duplicated(ct$original_study_patient_ID)], na.rm = TRUE)
# [1] 276

# how many patients didn't receive treatment?
sum(ct$no_treatment[!duplicated(ct$original_study_patient_ID)] == "1", na.rm = TRUE)
# [1] 24

# treatment table
txTab <- ct[, c(3, 112:ncol(clinicalData$clinicalTable))]
txTab$patient_ID <- as.character(txTab$patient_ID)

# summary table formatted for clarity
txTsum <- as.data.frame.matrix(summary(txTab), "data/curatedBreastData/bcTreatmentTable.csv")
write.csv(txTsum, "data/curatedBreastData/bcTreatmentTable.csv", na = "", row.names = FALSE, quote = FALSE)

# outcome table
library(tidyverse)
outcomeVars <- select(cvd, -2) %>%
  filter(time == "postTx")
outTab <- ct %>%
  mutate_if(sapply(ct, is.character), as.factor) %>%
  select(pull(outcomeVars, variableName))
  
outTsum <- as.data.frame.matrix(summary(outTab))
write_csv(outTsum, "data/curatedBreastData/bcOutcomeTable.csv")

# check if units error described in curatedBreastData gitub has been corrected (days not months)
# looks like it was
summary(bmc15data[bmc15data$study == "study_16446_GPL570_all-bmc15", "DFS_months_or_MIN_months_of_DFS"])
# DFS_months_or_MIN_months_of_DFS
# Min.   : 2.25                  
# 1st Qu.:23.39                  
# Median :36.50                  
# Mean   :38.73                  
# 3rd Qu.:50.21                  
# Max.   :77.21                  
# NA's   :7                      
summary(bmc15data[bmc15data$study != "study_16446_GPL570_all-bmc15", "DFS_months_or_MIN_months_of_DFS"])
 # DFS_months_or_MIN_months_of_DFS
 # Min.   :  0.624                
 # 1st Qu.: 24.708                
 # Median : 45.186                
 # Mean   : 52.542                
 # 3rd Qu.: 69.840                
 # Max.   :192.600                
 # NA's   :1629                   
summary(bmc15data[bmc15data$study == "study_16446_GPL570_all-bmc15", "DFS_months_or_MIN_months_of_DFS"] / 28)
# DFS_months_or_MIN_months_of_DFS
# Min.   :0.08036                
# 1st Qu.:0.83546                
# Median :1.30357                
# Mean   :1.38337                
# 3rd Qu.:1.79337                
# Max.   :2.75765                
# NA's   :7

# covariate table
coVars <- select(cvd, -2) %>%
  filter(time != "postTx", varType != "treatment")
coTab <- ct %>%
  mutate_if(sapply(ct, is.character), as.factor) %>%
  select(pull(coVars, variableName))

coTsum <- as.data.frame.matrix(summary(coTab))
write_csv(coTsum, "data/curatedBreastData/bcCovariateTable.csv")

# determine subsets for training supervised models of outcome based on treatment and patient strata
data(curatedBreastDataExprSetList)
write.csv(data(package = "curatedBreastData")$results[, 3:4], file = "data/curatedBreastData/bcEsetList.csv", row.names = FALSE)

# TODO: impute missing values
cbc <- filterAndImputeSamples(curatedBreastDataExprSetList, outputFile = "data/curatedBreastData/cbcImputeOutput.txt", classIndex = "phenoData")
# TODO: characterize processing of esets
# are gene symbols aligned?  can they be updated to current?
cbc <- processExpressionSetList(exprSetList=curatedBreastDataExprSetList, outputFileDirectory = "data/curatedBreastData/")
saveRDS(cbc, "data/curatedBreastData/bcProcessedEsetList.rds")

library(Coincide)
cbcMat <- exprSetListToMatrixList(cbc)
names(cbcMat) <- names(cbc)
sapply(cbcMat, dim)
#      study_1379_GPL1223_all study_2034_GPL96_all study_4913_GPL3558_all study_6577_GPL3883_all
# [1,]                  14801                12722                   7470                   4242
# [2,]                     60                  286                     50                     88
#      study_9893_GPL5049_all study_12071_GPL5186_all study_12093_GPL96_all study_16391_GPL570_all
# [1,]                  13154                    3298                 12722                  18260
# [2,]                    155                      46                   136                     48
#      study_16446_GPL570_all study_17705_GPL96_JBI_Tissue_BC_Tamoxifen
# [1,]                  18260                                     12722
# [2,]                    114                                       103
#      study_17705_GPL96_MDACC_Tissue_BC_Tamoxifen study_18728_GPL570_all study_19615_GPL570_all
# [1,]                                       12722                  18260                  18260
# [2,]                                         195                     21                    115
#      study_19697_GPL570_all study_20181_GPL96_all study_20194_GPL96_all study_21974_GPL6480_all
# [1,]                  18260                 12722                 12722                   17499
# [2,]                     24                    53                   261                      32
#      study_21997_GPL1390_all study_21997_GPL5325_all study_21997_GPL7504_all study_22226_GPL1708_all
# [1,]                   14318                   17200                   17904                   18507
# [2,]                      35                      28                      31                     128
#      study_22226_GPL4133_all study_22358_GPL5325_all study_23428_GPL5325_all study_25055_GPL96_MDACC_M
# [1,]                   19648                   17220                   16823                     12722
# [2,]                      20                     122                      16                       221
#      study_25055_GPL96_MDACC_PERU study_25065_GPL96_LBJ study_25065_GPL96_MDACC study_25065_GPL96_MDACC_MDA
# [1,]                        12722                 12722                   12722                       12722
# [2,]                            6                    17                      71                          15
#      study_25065_GPL96_PERU study_25065_GPL96_Spain study_25065_GPL96_USO study_32646_GPL570_all
# [1,]                  12722                   12722                 12722                  18260
# [2,]                     25                      16                    54                    115
#      study_33658_GPL570_all
# [1,]                  18260
# [2,]                     11


# TODO: interpolate missing values to increase gene list
# dump alligned expression files
dir.create("data/curatedBreastData/microarray data/")
# dir.create("./data/curatedBreastData/bcDump/maxSet")
# for(n in 1:length(cbcMat)) {
#   write_csv(tibble::as_tibble(t(cbcMat[[n]]), rownames("patientID")),
#             paste0("./data/curatedBreastData/bcDump/maxSet/", names(cbcMat)[n], ".csv.xz"))
# }
# 
# # try for maximum sample merge
# cbcMerged <- mergeDatasetList(cbcMat,
#                               minNumGenes = 0,
#                               minNumPatients = 0,
#                               batchNormalize = "BMC",
#                               NA_genesRemove = FALSE,
#                               outputFile = "data/curatedBreastData/merged_datasets/bmc34")
# dim(cbcMerged$mergedExprMatrix)
# # [1]  278 2718
# summary(colMeans(cbcMerged$mergedExprMatrix))
# #     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# # -1.55629 -0.11062  0.01853  0.00000  0.12486  1.31687 
# 
# summary(rowMeans(cbcMerged$mergedExprMatrix))
# #       Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# # -3.116e-16 -4.946e-17 -5.747e-18 -1.564e-18  4.017e-17  2.453e-16 
# 
# length(intersect(maxGenes, pam50Short))
# # [1] 2

# checkout curatedBreastData_dataMatrixList_proc_minVar001_min10kGenes_min40Samples
bcd <- readRDS("~/R/CoINcIDE/breast_analysis/curatedBreastData_dataMatrixList_proc_minVar001_min10kGenes_min40Samples.rds")
names(bcd) <- paste0(names(bcd), "-bmc15")
sapply(bcd, dim)
#      study_1379_GPL1223_all-bmc15 study_2034_GPL96_all-bmc15 study_9893_GPL5049_all-bmc15
# [1,]                        14801                      12722                        13154
# [2,]                           60                        286                          155
#      study_12093_GPL96_all-bmc15 study_16391_GPL570_all-bmc15 study_16446_GPL570_all-bmc15
# [1,]                       12722                        18260                        18260
# [2,]                         136                           48                          114
#      study_17705_GPL96_JBI_Tissue_BC_Tamoxifen-bmc15 study_17705_GPL96_MDACC_Tissue_BC_Tamoxifen-bmc15
# [1,]                                           12722                                             12722
# [2,]                                             103                                               195
#      study_19615_GPL570_all-bmc15 study_20181_GPL96_all-bmc15 study_20194_GPL96_all-bmc15
# [1,]                        18260                       12722                       12722
# [2,]                          115                          53                         261
#      study_22226_GPL1708_all-bmc15 study_22358_GPL5325_all-bmc15 study_25055_GPL96_MDACC_M-bmc15
# [1,]                         18507                         17220                           12722
# [2,]                           128                           122                             221
#      study_25065_GPL96_MDACC-bmc15 study_25065_GPL96_USO-bmc15 study_32646_GPL570_all-bmc15
# [1,]                         12722                       12722                        18260
# [2,]                            71                          54                          115

# save aligned expression sets
dir.create("./data/curatedBreastData/microarray data/example15bmc")
for(n in 1:length(bcd)) {
  write_csv(data.frame(patient_ID = colnames(bcd[[n]]),
                       t(bcd[[n]])),
            paste0("./data/curatedBreastData/microarray data/example15bmc/", names(bcd)[n], ".csv.xz"))
}

# pick treatment variables for ml datasets
bmc15studies <- character(0)
for(n in seq_along(bcd)) bmc15studies <- c(bmc15studies, rep(names(bcd)[n], dim(bcd[[n]])[2]))
bmc15samples <- tibble(study = bmc15studies, patient_ID = unlist(lapply(bcd, colnames)))
dim(bmc15samples)
# [1] 2237    2
length(unique(bmc15samples$study))
# [1] 17

# all tx variables have 12 NAs, are they all the same samples?  yes!
bmc15txTab <- txTab[txTab$patient_ID %in% bmc15samples,]
sum(rowSums(is.na(bmc15txTab)) > 0)
# [1] 12
bmc15txTab <- na.omit(bmc15txTab)
dim(bmc15txTab)
# [1] 2225   29

sapply(bmc15txTab[, c(-1, -28, -29)], table)
# fulvestrant gefitinib removed for clarity, both are "0" for all samples
#   aromatase_inhibitor estrogen_receptor_blocker estrogen_receptor_blocker_and_stops_production
# 0                2135                      1511                                           2135
# 1                  90                       714                                             90
#   estrogen_receptor_blocker_and_eliminator anti_HER2 tamoxifen doxorubicin epirubicin docetaxel capecitabine
# 0                                     2225      2174      1511        1568       1841      2011         2012
# 1                                        0        51       714         657        384       214          213
#   fluorouracil paclitaxel cyclophosphamide anastrozole trastuzumab letrozole chemotherapy
# 0         1504       1562             1298        2224        2174      2136         2207
# 1          721        663              927           1          51        89           18
#   hormone_therapy no_treatment methotrexate cetuximab carboplatin other taxaneGeneral
# 0            2225         2217         2221      2225        2225  2109          2101
# 1               0            8            4         0           0   116           124

table(bmc15txTab$neoadjuvant_or_adjuvant)
# adj       mixed         neo unspecified 
# 958         132        1025         110 

table(bmc15txTab$study_specific_protocol_number)
#    1   10   11   12   13   14   15   16   17    2    3    4    5    6    7    8    9 
# 1714    2    9   20   24    3   22    3    5  274   33   20   66    4   21    2    3 

# make tx heatmap
bmc15txMat <- data.matrix(bmc15txTab[, c(-1, -28, -29)], rownames.force = TRUE) - 1
row.names(bmc15txMat) <- bmc15txTab$patient_ID
bmc15txMat <- bmc15txMat[, colSums(bmc15txMat) != 0]
batch <- as.numeric(as.factor(ct$study_ID[match(row.names(bmc15txMat), ct$patient_ID)]))
batchColor <- ggthemes::tableau_color_pal("Tableau 20", "regular")(15)
bmc15txHeatmap <- heatmap(bmc15txMat, Rowv = NA,
                          scale = "none",
                          margins = c(12,3),
                          labRow = rep("", 2225),
                          RowSideColors = batchColor[batch],
                          xlab = "treatment variable",
                          ylab = "study",
                          keep.dendro = TRUE)

# bmc15txHeatmap2 <- gplots::heatmap.2(bmc15txMat, Rowv = NA,
#                           scale = "none",
#                           margins = c(15,5),
#                           labRow = rep("", 2225),
#                           # RowSideColors = ,
#                           keep.dendro = TRUE)

# check which studies have > 1 batch
patient2batch <- unique(data.frame(color.order = batch, study = as.factor(ct$study_ID[match(row.names(bmc15txMat), ct$patient_ID)])))
# study_17705_GPL96_JBI_Tissue_BC_Tamoxifen-bmc15 study_17705_GPL96_MDACC_Tissue_BC_Tamoxifen-bmc15
#                                             103                                               195
# study_25065_GPL96_MDACC-bmc15                   study_25065_GPL96_USO-bmc15
#                           221                                            71 

# make combined treatment vector for initial ml study
chemoVars <- colnames(bmc15txMat)[c(6:12, 14, 16, 18:20)]
writeLines(chemoVars, "data/curatedBreastData/microarray data/example15bmc/bmc15chemoVars")
hormoneVars <- colnames(bmc15txMat)[c(1, 3:5, 13, 15)]
writeLines(hormoneVars, "data/curatedBreastData/microarray data/example15bmc/bmc15hormoneVars")

bmc15data1 <- left_join(bmc15samples, tibble(patient_ID = rownames(bmc15txMat), as_tibble(bmc15txMat))) %>%
  mutate(chemo = rowSums(.[chemoVars])) %>%
  mutate(hormone = rowSums(.[hormoneVars])) %>%
  mutate_at(c("chemo", "hormone"), ~if_else(. != 0, 1, 0)) %>%
  filter(!is.na(chemo))

# pick outcome variables for ml datasets
bmc15outTab <- ct[ct$patient_ID %in% bmc15samples$patient_ID, c("patient_ID", names(outTab))]
bmc15outMissing <- sapply(bmc15outTab, function(x) sum(is.na(x)))
bmc15outTab <- bmc15outTab[, bmc15outMissing != 2237]
bmc15outTab$patient_ID <- as.character(bmc15outTab$patient_ID)
sapply(bmc15outTab[, c(-1, -2, -grep("month|RCB|pCR_spectrum", names(bmc15outTab)))], table)
#   pCR near_pCR RFS DFS  OS OS_up_until_death metastasis relapseOneYearVsFivePlus dead 
# 0 874      340 294 144  43               101        484                       77  113
# 1 260       67 736 512 195                27         72                       51   42 
# died_from_cancer_if_dead  relapseOneYearVsThreePlus
# 0                    123                        149
# 1                     32                         51

pCRsamples <- bmc15outTab$patient_ID[!is.na(bmc15outTab$pCR)]

# does near_pCR overlap pCR?  near_pCR and pCR_spectrum expand on subsets of pCR
near_pCRsamples <- bmc15outTab$patient_ID[!is.na(bmc15outTab$near_pCR)]
setdiff(near_pCRsamples, pCRsamples)
# integer(0)

RFSsamples <- bmc15outTab$patient_ID[!is.na(bmc15outTab$RFS)]
DFSsamples <- bmc15outTab$patient_ID[!is.na(bmc15outTab$DFS)]
length(intersect(RFSsamples, DFSsamples))
# [1] 0
length(intersect(c(RFSsamples, DFSsamples), pCRsamples))
# [1] 583
length(intersect(DFSsamples, pCRsamples))
# [1] 460
length(intersect(RFSsamples, pCRsamples))
# [1] 123
length(union(pCRsamples, union(RFSsamples, DFSsamples)))
# [1] 2225

writeLines(as.character(pCRsamples), "./data/curatedBreastData/microarray data/example15bmc/pCRsamples")
writeLines(as.character(RFSsamples), "./data/curatedBreastData/microarray data/example15bmc/RFSsamples")
writeLines(as.character(DFSsamples), "./data/curatedBreastData/microarray data/example15bmc/DFSsamples")

# make combined output vector for initial ml experiment
bmc15data <- left_join(bmc15data1, bmc15outTab) %>%
  mutate(posOutcome = if_else(is.na(pCR), 0, as.double(pCR)) + if_else(is.na(RFS), 0, as.double(RFS)) + if_else(is.na(DFS), 0, as.double(DFS)))

summary(bmc15data$posOutcome)
#   0    1    2 
# 839 1276  110 

summary(bmc15data$pCR)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 0.0000  0.0000  0.0000  0.2293  0.0000  1.0000    1091 
summary(bmc15data$RFS)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 0.0000  0.0000  1.0000  0.7112  1.0000  1.0000    1207 
summary(bmc15data$DFS)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 0.0000  1.0000  1.0000  0.7805  1.0000  1.0000    1569 

# check against package summary variables
summary(ct$chemotherapyClass)
# 0    1 NA's 
# 1204 1495   20 
summary(ct$radiotherapyClass)
#    0    1 NA's 
# 2195  504   20 
summary(ct$hormone_therapyClass)
# 0    1 NA's 
# 1733  966   20 
length(intersect(ct$patient_ID[ct$chemotherapyClass == 1], bmc15data1$patient_ID[bmc15data1$chemo == 1]))
# [1] 1186
table(bmc15data1$chemo)
#    0    1 
# 1039 1186 
# > length(intersect(ct$patient_ID[ct$hormone_therapyClass == 1], bmc15data1$patient_ID[bmc15data1$hormone == 1]))
# [1] 855
table(bmc15data1$hormone)
#    0    1 
# 1370  855 
length(intersect(ct$patient_ID[!is.na(ct$radiotherapyClass)], bmc15data$patient_ID))
# [1] 2225
table(ct$surgery_type)
# breast preserving        mastectomy 
#               102                42 
length(intersect(ct$patient_ID[!is.na(ct$surgery_type)], bmc15data$patient_ID))
# [1] 48

radioSx <- ct[ct$patient_ID %in% bmc15data$patient_ID,
              c("patient_ID", "radiotherapyClass", "surgery_type")] %>%
  mutate(patient_ID = as.character(patient_ID))
bmc15data <- left_join(bmc15data, radioSx)
table(bmc15data$radiotherapyClass)
#    0    1 
# 1783  442 

select(bmc15data, study, patient_ID, radio = radiotherapyClass, surgery = surgery_type, chemo, hormone, pCR, RFS, DFS, posOutcome) %>%
  filter(!is.na(chemo)) %>%
write_csv("data/curatedBreastData/bmc15mldata1.csv")

# add hormone, chemo, sx and outcome variables to heatmap
mldat <- read_csv("data/curatedBreastData/bmc15mldata1.csv") %>%
  # select(patient_ID:hormone) %>%
  mutate(surgery = as.numeric(as.factor(surgery)))
bmc15txMat3 <- cbind(bmc15txMat, as.matrix(mldat[match(mldat$patient_ID, rownames(bmc15txMat)), c(3:6, 10)]))
bmc15txHeatmap3 <- heatmap(bmc15txMat3, Rowv = NA,
                          scale = "none",
                          margins = c(12,3),
                          labRow = rep("", 2225),
                          RowSideColors = batchColor[batch],
                          xlab = "treatment variable",
                          ylab = "patient",
                          keep.dendro = TRUE)


# TODO: which covariates should be predictable from gene expression data?
# make covariate table
bmc15coTab <- ct[ct$patient_ID %in% bmc15samples,
                 !(names(ct) %in% c(names(bmc15outTab)[-1], names(bmc15txTab)[-1]))]
bmc15coTabMissing <- sapply(bmc15coTab, function(x) sum(is.na(x)))
bmc15coTab <- bmc15coTab[, bmc15coTabMissing != 2225]

## make merged dataset
cbd <-readRDS("~/R/CoINcIDE/breast_analysis/curatedBreastData_dataMatrixList_proc_minVar001_min10kGenes_min40Samples.rds")

bcdMerged <- mergeDatasetList(cbd, batchNormalize = "BMC", outputFile = "data/curatedBreastData/microarray data/bmc15")
dim(bcdMerged$mergedExprMatrix)
# [1] 8832 2237

summary(colMeans(bcdMerged$mergedExprMatrix))
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# -0.631637 -0.044277  0.005615  0.000000  0.055772  0.620924 

summary(rowMeans(bcdMerged$mergedExprMatrix))
#       Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -4.100e-16 -4.811e-17 -1.508e-18 -1.097e-18  4.640e-17  3.134e-16 

library(sva)
bpparam <- MulticoreParam(8, progressbar = TRUE)
bcdCombat <- ComBat(bcdMerged$mergedExprMatrix, bmc15samples$study, prior.plots = TRUE, BPPARAM = bpparam)

summary(colMeans(bcdCombat))
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# -0.589734 -0.044929  0.001436  0.000000  0.050836  0.922588 

summary(rowMeans(bcdCombat))
#       Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -4.843e-16 -5.594e-17 -1.175e-18 -5.240e-19  5.324e-17  4.712e-16 

# try applying comBat directly on merged dataset with no normalization
bcdMerged2 <- mergeDatasetList(cbd, batchNormalize = "none", outputFile = "data/curatedBreastData/microarray data/none15")
dim(bcdMerged2$mergedExprMatrix)
# [1] 8832 2237

summary(colMeans(bcdMerged2$mergedExprMatrix))
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -0.4497  4.6548  4.9742  5.3596  5.7117 11.5605 

summary(rowMeans(bcdMerged2$mergedExprMatrix))
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 2.345   3.669   5.217   5.360   6.660  12.717 

bcdCombat2 <- ComBat(bcdMerged2$mergedExprMatrix, bmc15samples$study, prior.plots = TRUE, BPPARAM = bpparam)
summary(colMeans(bcdCombat2))
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 4.771   5.315   5.362   5.360   5.411   6.282 

summary(rowMeans(bcdCombat2))
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 2.345   3.669   5.218   5.360   6.661  12.718 
write_csv(as_tibble(t(bcdCombat2), rownames = "patient_ID"), "data/curatedBreastData/microarray data/merged-combat15.csv.xz")

# try applying comBat directly
bcdMerged3 <- mergeDatasetList(cbd, batchNormalize = "combat", outputFile = "data/curatedBreastData/microarray data/combat15")
# no batch effect detected. Before p-value was  0.1308666  
# Error in mergeDatasetList(cbd, batchNormalize = "combat", outputFile = "data/curatedBreastData/merged_datasets/combat15") : 
#   object 'sampleDataPostCombat' not found

#overlap with pam50
load("../CoINcIDE/Coincide_GenomeMedicine_dissertation_scripts/pam50Short_genes.RData")
load("../CoINcIDE/Coincide_GenomeMedicine_dissertation_scripts/pam50_centroids_updatedSymbols.RData")
writeLines(centroidMatrix$geneSymbol, "data/curatedBreastData/pam50symbols")
length(intersect(centroidMatrix$geneSymbol, row.names(bcdMerged$mergedExprMatrix)))
# [1] 35
length(intersect(pam50Short, row.names(bcdMerged$mergedExprMatrix)))
# [1] 35

write_csv(data.frame(patient_ID = colnames(bcdMerged$mergedExprMatrix),
                     t(bcdMerged$mergedExprMatrix)), "./data/curatedBreastData/microarray data/example15bmc/ex15bmcMerged.csv.xz")

# make dump of un-normalized study expression matrices
# simplify study names
names(cbcMat) <- paste0("GSE", str_remove(names(cbcMat), "study_") %>%
                          str_remove("_all") %>%
                          str_remove("_GPL[0-9]+") %>%
                          str_remove("_Tissue_BC_Tamoxifen"))

# turn into tibbles
cbcMat <- map(cbcMat, as_tibble, rownames = "hgnc")

# function to transpose the tibble
transpose_df <- function(df) {
  t_df <- data.table::transpose(df)
  colnames(t_df) <- rownames(df)
  rownames(t_df) <- colnames(df)
  t_df <- t_df %>%
    tibble::rownames_to_column(.data = .) %>%
    tibble::as_tibble(.)
  return(t_df)
}

# remap symbols with current hgnc data
# https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&col=gd_prev_sym&col=gd_aliases&col=gd_pub_eg_id&col=gd_pub_ensembl_id&status=Approved&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit
symbolMap <- read_tsv("data/curatedBreastData/hgncSymbolMap21aug20.tsv") %>%
  mutate(`Previous symbols` = str_split(`Previous symbols`, ", "),
         `Alias symbols` = str_split(`Alias symbols`, ", "))

# combine previous and alias list columns, unnest, and add current/alt symbol identity row
symbolMap <- mutate(symbolMap, alt = map2(`Previous symbols`, `Alias symbols`, c), .keep = "unused") %>%
  mutate(alt = map(alt, ~ if(identical(., c(NA, NA))) NA_character_ else na.omit(.))) %>%
  unnest(cols = alt)
symbolMap <- bind_rows(symbolMap, mutate(symbolMap, alt = `Approved symbol`) %>% distinct()) %>%
  arrange(`Approved symbol`)

bcSymbols <- map(cbcMat, ~ pull(., hgnc))

# % outdated symbols
bcOld <- map(bcSymbols, ~ setdiff(., symbolMap$`Approved symbol`))
map2_dbl(bcSymbols, bcOld, ~ round(length(.y) / length(.x), digits = 2))
# GSE1379             GSE2034             GSE4913             GSE6577             GSE9893 
#    0.17                0.17                0.16                0.16                0.16 
# GSE12071            GSE12093            GSE16391            GSE16446        GSE17705_JBI 
#     0.32                0.17                0.18                0.18                0.17 
# GSE17705_MDACC            GSE18728            GSE19615            GSE19697            GSE20181 
#           0.17                0.18                0.18                0.18                0.17 
# GSE20194            GSE21974            GSE21997            GSE21997            GSE21997 
#     0.17                0.18                0.17                0.18                0.18 
# GSE22226            GSE22226            GSE22358            GSE23428    GSE25055_MDACC_M 
#     0.37                0.31                0.18                0.17                0.17 
# GSE25055_MDACC_PERU        GSE25065_LBJ      GSE25065_MDACC  GSE25065_MDACC_MDA       GSE25065_PERU 
#                0.17                0.17                0.17                0.17                0.17 
# GSE25065_Spain        GSE25065_USO            GSE32646            GSE33658 
#           0.17                0.17                0.18                0.18 

# % missing symbols after replacement
bcCurrent <- map(bcSymbols, ~ symbolMap$`Approved symbol`[match(., symbolMap$alt)])
map_dbl(bcCurrent, ~ round(sum(is.na(.)) / length(.), digits = 2))
# GSE1379             GSE2034             GSE4913             GSE6577             GSE9893 
#    0.10                0.10                0.09                0.09                0.09 
# GSE12071            GSE12093            GSE16391            GSE16446        GSE17705_JBI 
#     0.29                0.10                0.10                0.10                0.10 
# GSE17705_MDACC            GSE18728            GSE19615            GSE19697            GSE20181 
#           0.10                0.10                0.10                0.10                0.10 
# GSE20194            GSE21974            GSE21997            GSE21997            GSE21997 
#     0.10                0.10                0.10                0.10                0.11 
# GSE22226            GSE22226            GSE22358            GSE23428    GSE25055_MDACC_M 
#     0.16                0.17                0.10                0.10                0.10 
# GSE25055_MDACC_PERU        GSE25065_LBJ      GSE25065_MDACC  GSE25065_MDACC_MDA       GSE25065_PERU 
#                0.10                0.10                0.10                0.10                0.10 
# GSE25065_Spain        GSE25065_USO            GSE32646            GSE33658 
#           0.10                0.10                0.10                0.10 

bcReplace <- map2(bcCurrent, bcSymbols, ~ if_else(is.na(.x), paste0(.y, "_obs"), .x))
cbcMat <- map2(cbcMat, bcReplace, ~ mutate(.x, hgnc = .y))

# 1% - 2% are duplicates, pick largest variance
map_dbl(bcReplace, ~ round(sum(duplicated(.)) / length(.), digits = 2))
# GSE1379             GSE2034             GSE4913             GSE6577             GSE9893 
#     204                 164                  69                  16                 179 
# GSE12071            GSE12093            GSE16391            GSE16446        GSE17705_JBI 
#       11                 164                 259                 259                 164 
# GSE17705_MDACC            GSE18728            GSE19615            GSE19697            GSE20181 
#            164                 259                 259                 259                 164 
# GSE20194            GSE21974            GSE21997            GSE21997            GSE21997 
#      164                 249                 194                 243                 272 
# GSE22226            GSE22226            GSE22358            GSE23428    GSE25055_MDACC_M 
#      314                 283                 241                 232                 164 
# GSE25055_MDACC_PERU        GSE25065_LBJ      GSE25065_MDACC  GSE25065_MDACC_MDA       GSE25065_PERU 
#                 164                 164                 164                 164                 164 
# GSE25065_Spain        GSE25065_USO            GSE32646            GSE33658 
#            164                 164                 259                 259 

# function to pick row with greatest variance
maxSdGroup <- function(df) {
  rowwise(df) %>%
    mutate(sd = sd(c_across(- hgnc))) %>%
    group_by(hgnc) %>%
    slice(which.max(sd)) %>%
    select(- sd)
}

cbcMat <- map(cbcMat, maxSdGroup)

saveRDS(map(cbcMat, ungroup), "data/curatedBreastData/cbcMatCurrent.rds")

dir.create("data/curatedBreastData/microarray data/noNorm", recursive = TRUE)
for(n in names(cbcMat)) {
  expMat <- transpose_df(column_to_rownames(cbcMat[[n]], "hgnc"))
  names(expMat)[1] <- "sample_name"
  mutate(expMat, across(where(is.numeric), formatC, digits = 6, format = "f")) %>%
    write_csv(paste0("data/curatedBreastData/microarray data/noNorm/", n, "_noNorm.csv.xz"))
}
