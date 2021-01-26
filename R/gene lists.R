## make gene lists for for cbd15 embedding experiments
library(tidyverse)

# get name translation file
idmap <- read_tsv("data/curatedBreastData/hgncSymbolMap21aug20.tsv")

# get gene lists from esets
load("~/R/cancer/data/curatedBreastData/mlEsets2.rdata")

# gene lists from spectral bigraphs
load("reports/set15specQC.rdata")
comSpec12 <- com12$topElements$topGenes
comSpec34 <- com34$topElements$topGenes
intersect(comSpec12, comSpec34)
# [1] "SFRP1"

cs12eg <- filter(idmap, `Approved symbol` %in% comSpec12) %>%
  pull(`Ensembl gene ID`)
cs34eg <- filter(idmap, `Approved symbol` %in% comSpec34) %>%
  pull(`Ensembl gene ID`)

# get 50 gene list from first 2 components
comSpec100 <- esetVis::esetSpectralMap(com12$analysis$esetUsed,
                                  topGenes = 100,
                                  topSamples = 50, topSamplesVar = "patient_ID", topSamplesColor = "black",
                                  topSamplesJust = c(1, 0), topSamplesCex = 3,
                                  returnAnalysis = TRUE)
spec100 <- comSpec100$topElements$topGenes
intersect(comSpec34, spec100)
# [1] "PROM1"  "CXCL13" "CXCL9"  "UBD"    "ID4"    "SFRP1" 

spec100eg <- filter(idmap, `Approved symbol` %in% spec100) %>%
  pull(`Ensembl gene ID`)

# gene lists from other esetVis algos

  
# curated gene lists:  cancer related and stemmy cancer genes
pam50 <- read_lines("data/curatedBreastData/pam50/pam50symbols")
intersect(pam50, spec100)
# [1] "BIRC5"   "CCNB1"   "CDC20"   "ESR1"    "FOXA1"   "MAPT"    "RRM2"    "SFRP1"   "SLC39A6"

intersect(pam50, spec100[1:50])
# [1] "CDC20"   "ESR1"    "FOXA1"   "MAPT"    "RRM2"    "SFRP1"   "SLC39A6"

pam50eg <- filter(idmap, `Approved symbol` %in% pam50) %>%
  pull(`Ensembl gene ID`)

# abdu's 8
abdu8 <- c("ERBB2", "GATA3", "TP53", "MKI67", "PIK3C3", "FOXA1", "BRCA2", "BRCA1")

abdu8eg <- filter(idmap, `Approved symbol` %in% abdu8) %>%
  pull(`Ensembl gene ID`)

# xgboost top 50 features predicting posOuctocme from combat15 data set
xgb50 <- c('CDX4', 'GLRA1', 'OR12D3', 'DSCR4', 'HOXB8', 'C9', 'MTNR1B', 'MOS', 'HSD17B3', 'FGF20', 'KCNH4', 'ATP4B', 'CPB2', 'CRYBB1', 'ANGPTL3', 'MYH8', 'GYS2', 'SLC25A21', 'TAS2R7', 'F11', 'GABRA6', 'MYT1L', 'DEFB126', 'RPL18', 'GABRQ', 'ZFP37', 'PIP5K1B', 'MCM5', 'PRKAA1', 'WDR76', 'CHRM4', 'RPS6KC1', 'EIF1AY', 'WNT1', 'SCN3B', 'NLGN4Y', 'MAGEB1', 'NUDC', 'HIGD1A', 'OXCT2', 'GALR2', 'EEF1B2', 'RXRG', 'CALCA', 'TEX13A', 'CST3', 'IGFBP4', 'CRYGA', 'ESR1', 'ZNF750')

intersect(xgb50, spec100)
# [1] "ESR1"

xgb50eg <- filter(idmap, `Approved symbol` %in% xgb50) %>%
  pull(`Ensembl gene ID`)

# moses top 50 features predicting posOuctocme from combat15 data set
moses50 <- c("PRND", "FRS3", "FCN3", "DSCR4", "BRCA2", "CXCL6", "LMX1B", "DLX5", "OMP", "ADH6", "PGAP1", "ART3", "BCHE", "FGB", "IL1RAPL1", "FSTL4", "ASGR1", "ZNF135", "DLL3", "NPHS2", "ANGPT2", "GLP2R", "GRIA3", "HOXB8", "MSC", "PLA2R1", "CYP2F1", "TAS2R7", "NKX6.1", "WNT11", "CHST11", "CLCA4", "ENPEP", "PAH", "WFDC1", "CHGA", "SEZ6L", "UGT2A3", "PRDM16", "GALR2", "GUCA1A", "CASQ1", "NOS1AP", "CACNA2D3", "FHOD3", "SRGAP3", "TMOD2", "ATOH1", "SLC6A1", "HAS1")

intersect(moses50, spec100)
# character(0)

intersect(xgb50, moses50)
# [1] "DSCR4"  "HOXB8"  "TAS2R7" "GALR2" 

intersect(xgb50, pam50)
# [1] "ESR1"

intersect(pam50, moses50)
# character(0)

moses50eg <- filter(idmap, `Approved symbol` %in% moses50) %>%
  pull(`Ensembl gene ID`)

save(pam50, moses50, xgb50, spec100, comSpec12, comSpec34, file = "geneLists.rdata")

map(ls(pattern = "eg$"), ~write_lines(eval(as.name(.)), file = paste0("ml/embedding vectors/geneLists/", .)))
