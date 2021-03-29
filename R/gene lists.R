## make gene lists for for cbd15 embedding experiments
library(MASS)
library(tidyverse)

# get name translation file
idmap <- read_csv("data/curatedBreastData/hgncSymbolMap21aug20long.csv")

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

# load diffExp gene lists
cb15uni <- read_lines("data/curatedBreastData/diffExp/genes15studies.txt")
cb15uniEn <- idmap$`NCBI Gene ID`[match(cb15uni, idmap$`Approved symbol`)] %>%
  as.character() %>%
  na.omit()

cb15Funi <-read_lines("data/curatedBreastData/diffExp/filteredGenes15studies.txt")
cb15FuniEn <- idmap$`NCBI Gene ID`[match(cb15Funi, idmap$`Approved symbol`)] %>%
  as.character() %>%
  na.omit()

tamoxFuni <-read_lines("data/curatedBreastData/diffExp/filteredGenes4studies.txt")
tamoxFuniEn <- idmap$`NCBI Gene ID`[match(tamoxFuni, idmap$`Approved symbol`)] %>%
as.character() %>%
na.omit()

diffExp <- map(list(cb15 = "data/curatedBreastData/diffExp/top120genes15studies.csv",
                    cb15f = "data/curatedBreastData/diffExp/top100genes15studiesFiltered.csv",
                    tam4 = "data/curatedBreastData/diffExp/top2000genes4studies.csv",
                    tam4f = "data/curatedBreastData/diffExp/top1100genes4studiesFiltered.csv"), read_csv)

bestGeneSets <- map(diffExp, ~ pull(., symbol))
bestGeneSetsE <- map(bestGeneSets, ~ idmap$`NCBI Gene ID`[match(., idmap$`Approved symbol`)]) %>%
  map(as.character) %>%
  map(na.omit)

# run GOstats
library(Homo.sapiens)
library(GOstats)

# setup function, ont in ("BP", "MF", "CC")
makeHyperParam <- function(genes, universe, annot = "org.Hs.eg.db", ont, dir = "over") {
  new("GOHyperGParams",
      geneIds = genes,
      universeGeneIds = universe,
      annotation = annot,
      ontology = ont,
      pvalueCutoff = 1,
      conditional = TRUE,
      testDirection = dir)
}

enrichmentList <- function(genes, universe, annot = "org.Hs.eg.db", dir = "over") {
 list(BP = hyperGTest(makeHyperParam(genes, universe, annot, ont = "BP", dir)),
      MF = hyperGTest(makeHyperParam(genes, universe, annot, ont = "MF", dir)),
      CC = hyperGTest(makeHyperParam(genes, universe, annot, ont = "CC", dir))
      ) %>%
  map(summary)
}

bestGeneSetsGO <- map2(bestGeneSetsE, list(cb15uniEn, cb15FuniEn, cb15uniEn, tamoxFuniEn),
                       ~ enrichmentList(.x, .y))

bestGeneSetsGOsum <- map(bestGeneSetsGO, ~ map(., as_tibble))
for(i in 1:3) {
  for(j in 1:3) {
    names(bestGeneSetsGOsum[[i]][[j]]) <- c("GOID", "Pvalue","OddsRatio", "ExpCount", "Count", "Size", "Term")
    }
}

bestGeneSetsGOsum <- map(bestGeneSetsGOsum, ~ bind_rows(., .id = "name space"))

# TODO: add qvalues try on each namespace, p value density of combined tests is far from uniform
# goq <- qvalue::qvalue(GOEtable$Pvalue)
# goq$pi0
# # [1] 1
# GOEtable$qvalue <- goq$qvalues
# GOEtable$lfdr <- goq$lfdr

bestGenesBP01 <- map(bestGeneSetsGO, pluck, 1) %>%
  map(filter, Pvalue < 0.01) %>%
  map(pull, GOBPID)
map2(bestGenesBP01, names(bestGenesBP01), ~ write_lines(.x, paste0("data/curatedBreastData/diffExp/", .y, "BP01.txt")))

bestGenesMF01 <- map(bestGeneSetsGO, pluck, 2) %>%
  map(filter, Pvalue < 0.01) %>%
  map(pull, GOMFID)
map2(bestGenesMF01, names(bestGenesMF01), ~ write_lines(.x, paste0("data/curatedBreastData/diffExp/", .y, "MF01.txt")))

bestGenesCC01 <- map(bestGeneSetsGO, pluck, 3) %>%
  map(filter, Pvalue < 0.01) %>%
  map(pull, GOCCID)
map2(bestGenesCC01, names(bestGenesCC01), ~ write_lines(.x, paste0("data/curatedBreastData/diffExp/", .y, "CC01.txt")))

bestGenesBP05 <- map(bestGeneSetsGO, pluck, 1) %>%
  map(filter, Pvalue < 0.05) %>%
  map(pull, GOBPID)
map2(bestGenesBP05, names(bestGenesBP05), ~ write_lines(.x, paste0("data/curatedBreastData/diffExp/", .y, "BP05.txt")))

bestGenesMF05 <- map(bestGeneSetsGO, pluck, 2) %>%
  map(filter, Pvalue < 0.05) %>%
  map(pull, GOMFID)
map2(bestGenesMF05, names(bestGenesMF05), ~ write_lines(.x, paste0("data/curatedBreastData/diffExp/", .y, "MF05.txt")))

bestGenesCC05 <- map(bestGeneSetsGO, pluck, 3) %>%
  map(filter, Pvalue < 0.05) %>%
  map(pull, GOCCID)
map2(bestGenesCC05, names(bestGenesCC05), ~ write_lines(.x, paste0("data/curatedBreastData/diffExp/", .y, "CC05.txt")))
