library(Biobase)
library(esetVis)
library(tidyverse)
library(tidyselect)

# get get clinical data for tamoxifen studies
cv <- read_csv("data/curatedBreastData/tamoxifenClinData.csv") %>%
  mutate(patient_ID = as.character(patient_ID), `> 5 yr survival` = as_factor(posOutcome))

# get esets (combat15 eset is contained in spectral plot output list)
load("~/R/cancer/reports/set15specQC.rdata")
cb15Eset <- com12$analysis$esetUsed
pData(cb15Eset) <- mutate(pData(cb15Eset), tamoxifen = as_factor(tamoxifen))

raw15list <- readRDS("data/curatedBreastData/bcProcessedEsetList.rds")

# plots showing outcome
cb15_12 <- esetSpectralMap(cb15Eset,
                           title = "15 batch seperated breast cancer microarray data sets\n empirical bayes norm",
                           colorVar = "tamoxifen", # color = colorPalette,
                           shapeVar = "posOutcome", shape = c(1, 4, 20),
                           # symmetryAxes = "separate",
                           topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                           topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                           topSamplesJust = c(1, 0), topSamplesCex = 3,
                           returnAnalysis = TRUE)

cb15_13 <- esetSpectralMap(cb15Eset,
                           title = "15 batch seperated breast cancer microarray data sets\n empirical bayes norm",
                           colorVar = "series_id", # color = colorPalette,
                           shapeVar = "posOutcome", shape = c(1, 4, 20),
                           # symmetryAxes = "separate",
                           topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                           topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                           topSamplesJust = c(1, 0), topSamplesCex = 3,
                           dim = c(1, 3),
                           returnAnalysis = TRUE)

# get SNR
library(SNAGEE)

# get ncbi ids
symMap <- read_csv("data/curatedBreastData/hgncSymbolMap21aug20long.csv")
sncb15 <- list(genes = symMap$`NCBI Gene ID`[match(rownames(exprs(cb15Eset)), symMap$alt)],
               data = exprs(cb15Eset))
sncb15Out <- qualStudy(sncb15)
# [1] 0.7113271

# differential expression
lmflscb15 <- limma::lmFit(cb15Eset, method = "ls")
hist(lmflscb15$Amean)
limma::plotSA(lmflscb15, main = "sigma (residual variance) vs mean log expression 15 studies")
cb15lm <- limma::lmFit(cb15Eset[, !is.na(pData(cb15Eset)$posOutcome)], cbind(2, as.numeric(na.omit(pData(cb15fEset)[, "posOutcome"]))) - 1)
cb15Eb <- limma::eBayes(cb15lm, trend = TRUE)
cb15Top <- limma::topTable(cb15Eb, 2, number = 120) %>%
  rownames_to_column("symbol") %>%
  as_tibble()
limma::volcanoplot(cb15Eb, coef = 2, highlight = 10, names = names(cb15fEb$Amean), main = "differential expression > 5 yr survival (15 studies not filtered)")

# filter minimally expressed genes
keep <- lmflscb15$Amean > 5
lmflscb15F <- limma::eBayes(lmflscb15[keep,], trend = TRUE)
limma::plotSA(lmflscb15F, main = "sigma vs mean log expression 15 studies, mean > 5", col = c("black", "coral"))
hist(lmflscb15F$Amean, main = "mean log expression 15 studies, filtered mean > 5")

cb15fEset <- cb15Eset[keep,!is.na(pData(cb15Eset)$posOutcome)]
cb15flm <- limma::lmFit(cb15fEset, cbind(2, as.numeric(pData(cb15fEset)[, "posOutcome"])) - 1)
cb15fEb <- limma::eBayes(cb15flm, trend = TRUE)
cb15fTop <- limma::topTable(cb15fEb, 2, number = 100) %>%
  rownames_to_column("symbol") %>%
  as_tibble()
limma::volcanoplot(cb15fEb, coef = 2, highlight = 10, names = names(cb15fEb$Amean), main = "differential expression > 5 yr survival (15 studies)")

setdiff(cb15Top$symbol, cb15fTop$symbol)
# [1] "ZNF623" "DARS2"  "HSF1" "LRP12" "RAD54B"  "GABRQ" "ZNF696"  "OPLAH" "C7orf10" "CCNE2"   "DHODH" 
# [12] "CXCL2"  "UTRN" "RAD9A" "SLC8A1"  "ENO3"  "MNAT1" "FOXD1"   "ZNF254"  "CHI3L2"  "MYH7B" "ADRA2C"
# [23] "PTK6" "TTLL4" "PLA2G5"  "RGS13" "PSTPIP2" "PIGZ" "CACNA1H" "BRCA1"   "TMEM121" "CGREF1"  "PLS1" 
# [34] "GP1BB"   "KIF23"   "DUSP26"

write_lines(cb15fTop$symbol, "data/curatedBreastData/diffExp/filteredGenes15studies.txt")
write_csv(cb15Top, "data/curatedBreastData/diffExp/top120genes15studies.csv")
write_csv(cb15fTop, "data/curatedBreastData/diffExp/top100genes15studiesFiltered.csv")


# estrogen blocker studies
tam5Eset <- readRDS("data/curatedBreastData/tamoxifenEset.rds")
pDtam5Eset <- left_join(rownames_to_column(pData(tam5Eset), "patient_ID"), cv) %>%
  column_to_rownames("patient_ID")
pData(tam5Eset) <- pDtam5Eset
table(pData(tam5Eset)$series_id)
# GSE12093  GSE1379 GSE16391 GSE17705  GSE9893 
#      136       60       48      298      155

tam5_12 <- esetSpectralMap(tam5Eset,
                           title = "5 study subset with aduvent tam5ifen treatment\n empirical bayes norm",
                           colorVar = "series_id", # color = colorPalette,
                           shapeVar = "> 5 yr survival", shape = c(1, 4, 20),
                           # symmetryAxes = "separate",
                           topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                           topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                           topSamplesJust = c(1, 0), topSamplesCex = 3,
                           returnAnalysis = TRUE)

tam5_13 <- esetSpectralMap(tam5Eset,
                           title = "5 study subset with aduvent tam5ifen treatment\n empirical bayes norm",
                           colorVar = "series_id", # color = colorPalette,
                           shapeVar = "> 5 yr survival", shape = c(1, 4, 20),
                           # symmetryAxes = "separate",
                           topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                           topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                           topSamplesJust = c(1, 0), topSamplesCex = 3,
                           dim = c(1, 3),
                           returnAnalysis = TRUE)

sntam5 <- list(genes = symMap$`NCBI Gene ID`[match(rownames(exprs(tam5Eset)), symMap$`Approved symbol`)],
               data = exprs(tam5Eset))
sntam5Out <- qualStudy(sntam5)
# [1] 0.5535425

# differential expression
lmflstam5 <- limma::lmFit(tam5Eset, method = "ls")
hist(lmflstam5$Amean, main = "mean log expression 5 studies")
limma::plotSA(lmflstam5, main = "sigma (residual variance) vs mean log expression 5 studies")
tam5lm <- limma::lmFit(tam5Eset[, !is.na(pData(tam5Eset)$posOutcome)], cbind(2, as.numeric(na.omit(pData(tam5Eset)[, "posOutcome"]))) - 1)
tam5Eb <- limma::eBayes(tam5lm, trend = TRUE)
tam5Top <- limma::topTable(tam5Eb, 2, number = 100) %>%
  rownames_to_column("symbol") %>%
  as_tibble()
limma::volcanoplot(tam5Eb, coef = 2, highlight = 10, names = names(tam5Eb$Amean), main = "differential expression > 5 yr survival (5 studies not filtered)")

# filter minimally expressed genes
keep <- lmflstam5$Amean > 5
lmflstam5F <- limma::eBayes(lmflstam5[keep,], trend = TRUE)
limma::plotSA(lmflstam5F, main = "sigma vs mean log expression 5 studies, mean > 5", col = c("black", "coral"))
hist(lmflstam5F$Amean, main = "mean log expression 5 studies, filtered mean > 5")

tam5fEset <- tam5Eset[keep,!is.na(pData(tam5Eset)$posOutcome)]
tam5flm <- limma::lmFit(tam5fEset, cbind(2, as.numeric(pData(tam5fEset)[, "posOutcome"])) - 1)
tam5fEb <- limma::eBayes(tam5flm, trend = TRUE)
tam5fTop <- limma::topTable(tam5fEb, 2, number = 100) %>%
  rownames_to_column("symbol") %>%
  as_tibble()
limma::volcanoplot(tam5fEb, coef = 2, highlight = 10, names = names(tam5fEb$Amean), main = "differential expression > 5 yr survival (5 studies)")

setdiff(tam5Top$symbol, tam5fTop$symbol)
#  [1] "MYH7B"    "REG3A"    "FOXL2"    "LRP12"    "IFNA8"    "F13B"     "HCRT"     "MYEF2"    "SLC2A2"  
# [10] "ARMC4"    "HRASLS"   "GUCA2B"   "CCNE2"    "OR12D3"   "CRX"      "ST18"     "LSAMP"    "MTTP"    
# [19] "OPN1SW"   "SLC17A3"  "OPLAH"    "ARL14"    "MCHR1"    "INSL4"    "SHOX"     "CXCL2"    "SLC25A21"
# [28] "ZNF696"   "KIAA0087" "MAN1C1"   "SLC38A4"  "PPFIA3"   "UTRN"     "SDPR"     "LAMC3"    "CARD14"  
# [37] "PARK2"   

write_lines(tam5fTop$symbol, "data/curatedBreastData/diffExp/filteredGenes5studies.txt")
write_csv(tam5Top, "data/curatedBreastData/diffExp/top100genes5studies.csv")
write_csv(tam5fTop, "data/curatedBreastData/diffExp/top100genes5studiesFiltered.csv")


# tamoxifen studies
tamoxEset <- tam5Eset[, pData(tam5Eset)$series_id != "GSE16391"]
pDtamoxEset <- left_join(rownames_to_column(pData(tamoxEset), "patient_ID"), cv) %>%
  column_to_rownames("patient_ID")
pData(tamoxEset) <- pDtamoxEset

tam4_12 <- esetSpectralMap(tamoxEset,
                           title = "4 study subset with aduvent tamoxifen treatment\n empirical bayes norm",
                           colorVar = "series_id", # color = colorPalette,
                           shapeVar = "> 5 yr survival", shape = c(1, 4, 20),
                           # symmetryAxes = "separate",
                           topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                           topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                           topSamplesJust = c(1, 0), topSamplesCex = 3,
                           returnAnalysis = TRUE)

tam4_13 <- esetSpectralMap(tamoxEset,
                           title = "4 study subset with aduvent tamoxifen treatment\n empirical bayes norm",
                           colorVar = "series_id", # color = colorPalette,
                           shapeVar = "> 5 yr survival", shape = c(1, 4, 20),
                           # symmetryAxes = "separate",
                           topGenes = 20, topGenesJust = c(1, 0), topGenesCex = 2, topGenesColor = "darkgrey",
                           topSamples = 15, topSamplesVar = "patient_ID", topSamplesColor = "black",
                           topSamplesJust = c(1, 0), topSamplesCex = 3,
                           dim = c(1, 3),
                           returnAnalysis = TRUE)

sntam4 <- list(genes = symMap$`NCBI Gene ID`[match(rownames(exprs(tamoxEset)), symMap$`Approved symbol`)],
               data = exprs(tamoxEset))
sntam4Out <- qualStudy(sntam4)
# [1] 0.5373691

# differential expression
lmflstamox <- limma::lmFit(tamoxEset, method = "ls")
hist(lmflstamox$Amean, main = "mean log expression 4 studies")
limma::plotSA(lmflstamox, main = "sigma (residual variance) vs mean log expression 4 studies")
tamoxlm <- limma::lmFit(tamoxEset[, !is.na(pData(tamoxEset)$posOutcome)], cbind(2, as.numeric(na.omit(pData(tamoxEset)[, "posOutcome"]))) - 1)
tamoxEb <- limma::eBayes(tamoxlm, trend = TRUE)
tamoxTop <- limma::topTable(tamoxEb, 2, number = 2000) %>%
  rownames_to_column("symbol") %>%
  as_tibble()
limma::volcanoplot(tamoxEb, coef = 2, highlight = 10, names = names(tamoxEb$Amean), main = "differential expression > 5 yr survival (4 studies not filtered)")

# filter minimally expressed genes
keep <- lmflstamox$Amean > 5
lmflstamoxF <- limma::eBayes(lmflstamox[keep,], trend = TRUE)
limma::plotSA(lmflstamoxF, main = "sigma vs mean log expression 4 studies, mean > 5", col = c("black", "coral"))
hist(lmflstamoxF$Amean, main = "mean log expression 4 studies, filtered mean > 5")

tamoxfEset <- tamoxEset[keep,!is.na(pData(tamoxEset)$posOutcome)]
tamoxflm <- limma::lmFit(tamoxfEset, cbind(2, as.numeric(pData(tamoxfEset)[, "posOutcome"])) - 1)
tamoxfEb <- limma::eBayes(tamoxflm, trend = TRUE)
tamoxfTop <- limma::topTable(tamoxfEb, 2, number = 1100) %>%
  rownames_to_column("symbol") %>%
  as_tibble()
limma::volcanoplot(tamoxfEb, coef = 2, highlight = 10, names = names(tamoxfEb$Amean), main = "differential expression > 5 yr survival (4 studies)")

setdiff(tamoxTop$symbol, tamoxfTop$symbol)

#  [1] "MYH7B"    "REG3A"    "FOXL2"    "LRP12"    "IFNA8"    "F13B"     "HCRT"     "MYEF2"    "SLC2A2"  
# [10] "ARMC4"    "HRASLS"   "GUCA2B"   "CCNE2"    "OR12D3"   "CRX"      "ST18"     "LSAMP"    "MTTP"    
# [19] "OPN1SW"   "SLC17A3"  "OPLAH"    "ARL14"    "MCHR1"    "INSL4"    "SHOX"     "CXCL2"    "SLC25A21"
# [28] "ZNF696"   "KIAA0087" "MAN1C1"   "SLC38A4"  "PPFIA3"   "UTRN"     "SDPR"     "LAMC3"    "CARD14"  
# [37] "PARK2"   

identical(tam5Top, tamoxTop)
# [1] TRUE

identical(tam5fTop, tamoxfTop)
# [1] TRUE

write_lines(rownames(exprs(tamoxfEset)), "data/curatedBreastData/diffExp/filteredGenes4studies.txt")
write_lines(rownames(exprs(cb15fEset)), "data/curatedBreastData/diffExp/filteredGenes15studies.txt")
write_lines(rownames(exprs(cb15Eset)), "data/curatedBreastData/diffExp/genes15studies.txt")
write_csv(tamoxTop, "data/curatedBreastData/diffExp/top2000genes4studies.csv")
write_csv(tamoxfTop, "data/curatedBreastData/diffExp/top1100genes4studiesFiltered.csv")

# save esets & limma ebayes results for broad gsea
save(cb15Eset, cb15fEset, tamoxEset, tamoxfEset, file = "data/curatedBreastData/diffExp/diffExpEsets.rdata")
save(cb15Eb, cb15fEb, tamoxEb, tamoxfEb, file = "data/curatedBreastData/diffExp/diffExpLimmaEbayes.rdata")

# try tidyTranscriptomics
library(tidybulk)
# library(SummarizedExperiment)


# get validation set from tamoxifen positive metagx samples (GSE58644)
# get expression matrix
gse58644em <- read_csv("data/curatedBreastData/gse58644tamoxifenCombat.csv.xz")

# get clincal data set
gse58644Pheno <- read_csv("data/curatedBreastData/gse58644tamoxifenClinData.csv")

# get imputed data
impMfi <- read_csv("ml/embedding vectors/imputation_results/MFI_imputed_tamox.csv") %>%
  mutate(across(c(patient_ID, gpl, pam_coincide, p5), as.character))
