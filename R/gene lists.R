## make gene lists for for cbd15 embedding experiments
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

# ml gene lists
tam4pv100 <- read_lines("https://mozi.ai/datasets/embedding-vectors/tamoxifen_group/100_pln_gen_go_pathway.txt")

# try pathwayPCA
library(pathwayPCA)

# expression data
cb15Eset <- combat15es2[, !is.na(pData(combat15es2)$posOutcome)]

dim(exprs(cb15Eset))
# [1] 8832 2225

dim(pData(cb15Eset))
# [1] 2213  115

# pheno table is missing samples!
cb15Eset <- cb15Eset[, rownames(pData(cb15Eset))]

dim(exprs(cb15Eset))
# [1] 8832 2213

dim(pData(cb15Eset))
# [1] 2213  115

# get tamoxifen subset
tam5Eset <- cb15Eset[, pData(cb15Eset)$series_id %in% c("GSE12093", "GSE1379", "GSE16391", "GSE17705", "GSE9893")]
tam4Eset <- tam5Eset[, pData(tam5Eset)$series_id != "GSE16391"]

# gene sets
msdbCBgenes <- read_gmt("/mnt/biodata/msigdb/msigdb.v7.4.CBsymbol.gmt")
names(msdbCBgenes)
# [1] "pathways" "TERMS"   

# pick out atomspace sets
gobpCBgenes <- transpose(msdbCBgenes) %>%
  keep(~ pluck(., 2) %>% str_detect("^GOBP_")) %>%
  transpose()
gobpCBgenes <- CreatePathwayCollection(gobpCBgenes$pathways, unlist(gobpCBgenes$TERMS))

t4gobp_omics <- CreateOmics(
  
  # protein expression data
  assayData_df = tibble(sample = rownames(pData(tam4Eset)), as_tibble(t(exprs(tam4Eset)))),
  
  # pathway collection
  pathwayCollection_ls = gobpCBgenes,
  
  # category phenotype
  response = tibble(sample = rownames(pData(tam4Eset)), no_relapse = as.factor(pData(tam4Eset)$posOutcome)),
  respType = "categorical",
  
  # retain pathways with > n proteins
  minPathSize = 1
)
# ======  Creating object of class OmicsCateg  =======
#   The input pathway database included 15440 unique features.
# The input assay dataset included 8832 features.
# Only pathways with at least 1 or more features included in the assay dataset are
# tested (specified by minPathSize parameter). There are 7440 pathways which meet
# this criterion.
# Because pathwayPCA is a self-contained test (PMID: 17303618), only features in
# both assay data and pathway database are considered for analysis. There are 6878
# such features shared by the input assay and pathway database.

t4gobp_aes1pc <- AESPCA_pVals(
  
  # The Omics data container
  object = t4gobp_omics,
  
  # One principal component per pathway
  numPCs = 1,
  
  # Use parallel computing with 2 cores
  parallel = TRUE, numCores = 8,
  
  # Estimate the p-values parametrically
  numReps = 0,
  
  # Control FDR via Benjamini-Hochberg
  adjustment = "BY"
)

goccCBgenes <- transpose(msdbCBgenes) %>%
  keep(~ pluck(., 2) %>% str_detect("^GOCC_")) %>%
  transpose()
goccCBgenes <- CreatePathwayCollection(goccCBgenes$pathways, unlist(goccCBgenes$TERMS))

t4gocc_omics <- CreateOmics(
  
  # protein expression data
  assayData_df = tibble(sample = rownames(pData(tam4Eset)), as_tibble(t(exprs(tam4Eset)))),
  
  # pathway collection
  pathwayCollection_ls = goccCBgenes,
  
  # category phenotype
  response = tibble(sample = rownames(pData(tam4Eset)), no_relapse = as.factor(pData(tam4Eset)$posOutcome)),
  respType = "categorical",
  
  # retain pathways with > n proteins
  minPathSize = 1
)

t4gocc_aes1pc <- AESPCA_pVals(
  
  # The Omics data container
  object = t4gocc_omics,
  
  # One principal component per pathway
  numPCs = 1,
  
  # Use parallel computing with 2 cores
  parallel = TRUE, numCores = 8,
  
  # Estimate the p-values parametrically
  numReps = 0,
  
  # Control FDR via Benjamini-Hochberg
  adjustment = "BY"
)

gomfCBgenes <- transpose(msdbCBgenes) %>%
  keep(~ pluck(., 2) %>% str_detect("^GOMF_")) %>%
  transpose()
gomfCBgenes <- CreatePathwayCollection(gomfCBgenes$pathways, unlist(gomfCBgenes$TERMS))

t4gomf_omics <- CreateOmics(
  
  # protein expression data
  assayData_df = tibble(sample = rownames(pData(tam4Eset)), as_tibble(t(exprs(tam4Eset)))),
  
  # pathway collection
  pathwayCollection_ls = gomfCBgenes,
  
  # category phenotype
  response = tibble(sample = rownames(pData(tam4Eset)), no_relapse = as.factor(pData(tam4Eset)$posOutcome)),
  respType = "categorical",
  
  # retain pathways with > n proteins
  minPathSize = 1
)

t4gomf_aes1pc <- AESPCA_pVals(
  
  # The Omics data container
  object = t4gomf_omics,
  
  # One principal component per pathway
  numPCs = 1,
  
  # Use parallel computing with 2 cores
  parallel = TRUE, numCores = 8,
  
  # Estimate the p-values parametrically
  numReps = 0,
  
  # Control FDR via Benjamini-Hochberg
  adjustment = "BY"
)

reactomeCBgenes <- transpose(msdbCBgenes) %>%
  keep(~ pluck(., 2) %>% str_detect("^REACTOME_")) %>%
  transpose()
reactomeCBgenes <- CreatePathwayCollection(reactomeCBgenes$pathways, unlist(reactomeCBgenes$TERMS))

t4rct_omics <- CreateOmics(
  
  # protein expression data
  assayData_df = tibble(sample = rownames(pData(tam4Eset)), as_tibble(t(exprs(tam4Eset)))),
  
  # pathway collection
  pathwayCollection_ls = reactomeCBgenes,
  
  # category phenotype
  response = tibble(sample = rownames(pData(tam4Eset)), no_relapse = as.factor(pData(tam4Eset)$posOutcome)),
  respType = "categorical",
  
  # retain pathways with > n proteins
  minPathSize = 1
)

t4rct_aes1pc <- AESPCA_pVals(
  
  # The Omics data container
  object = t4rct_omics,
  
  # One principal component per pathway
  numPCs = 1,
  
  # Use parallel computing with 2 cores
  parallel = TRUE, numCores = 8,
  
  # Estimate the p-values parametrically
  numReps = 0,
  
  # Control FDR via Benjamini-Hochberg
  adjustment = "BY"
)

# annotate $ save
save(t4gobp_aes1pc, t4gobp_omics, t4gocc_aes1pc, t4gocc_omics, t4gomf_aes1pc, t4gomf_omics, t4rct_aes1pc, t4rct_omics, file = "data/curatedBreastData/diffExp/pathwayPCAtam4.rdata")

tam4gobp <- mutate(t4gobp_aes1pc$pVals_df, STANDARD_NAME = str_replace(terms, "^GOBP", "GO")) %>%
  left_join(msigdb) %>%
  select(-HISTORICAL_NAMES:-GENESET_LISTING_URL, -DESCRIPTION_FULL, -TAGS, -FOUNDER_NAMES:-BUILD_DATE, -MEMBERS:-MEMBERS_MAPPING) %>%
  mutate(GOID = str_remove(EXTERNAL_DETAILS_URL, "http://amigo.geneontology.org/amigo/term/"))
write_tsv(tam4gobp, "data/curatedBreastData/diffExp/tam4_pathwayPCA_gobp_msigdb74.tsv")

tam4gocc <- mutate(t4gocc_aes1pc$pVals_df, STANDARD_NAME = str_replace(terms, "^GOCC", "GO")) %>%
  left_join(msigdb) %>%
  select(-HISTORICAL_NAMES:-GENESET_LISTING_URL, -DESCRIPTION_FULL, -TAGS, -FOUNDER_NAMES:-BUILD_DATE, -MEMBERS:-MEMBERS_MAPPING) %>%
  mutate(GOID = str_remove(EXTERNAL_DETAILS_URL, "http://amigo.geneontology.org/amigo/term/"))
write_tsv(tam4gocc, "data/curatedBreastData/diffExp/tam4_pathwayPCA_gocc_msigdb74.tsv")

tam4gomf <- mutate(t4gomf_aes1pc$pVals_df, STANDARD_NAME = str_replace(terms, "^GOMF", "GO")) %>%
  left_join(msigdb) %>%
  select(-HISTORICAL_NAMES:-GENESET_LISTING_URL, -DESCRIPTION_FULL, -TAGS, -FOUNDER_NAMES:-BUILD_DATE, -MEMBERS:-MEMBERS_MAPPING) %>%
  mutate(GOID = str_remove(EXTERNAL_DETAILS_URL, "http://amigo.geneontology.org/amigo/term/"))
write_tsv(tam4gomf, "data/curatedBreastData/diffExp/tam4_pathwayPCA_gomf_msigdb74.tsv")

tam4rct <- mutate(t4rct_aes1pc$pVals_df, STANDARD_NAME = terms) %>%
  left_join(msigdb) %>%
  select(-HISTORICAL_NAMES:-GENESET_LISTING_URL, -DESCRIPTION_FULL, -TAGS, -FOUNDER_NAMES:-BUILD_DATE, -MEMBERS:-MEMBERS_MAPPING) %>%
  mutate(ID = str_remove(EXTERNAL_DETAILS_URL, "http://www.reactome.org/cgi-bin/eventbrowser_st_id?ST_ID=REACT_"))
write_tsv(tam4rct, "data/curatedBreastData/diffExp/tam4_pathwayPCA_reactome_msigdb74.tsv")

# repeat for 15 study data set
cb15gobp_omics <- CreateOmics(
  
  # protein expression data
  assayData_df = tibble(sample = rownames(pData(cb15Eset)), as_tibble(t(exprs(cb15Eset)))),
  
   # pathway collection
  pathwayCollection_ls = gobpCBgenes,
  
  # category phenotype
  response = tibble(sample = rownames(pData(cb15Eset)), no_relapse = as.factor(pData(cb15Eset)$posOutcome)),
  respType = "categorical",
  
  # retain pathways with > n proteins
  minPathSize = 1
)

cb15gobp_aes1pc <- AESPCA_pVals(

    # The Omics data container
  object = cb15gobp_omics,
  
  # One principal component per pathway
  numPCs = 1,
  
  # Use parallel computing with 2 cores
  parallel = TRUE, numCores = 8,

  # Estimate the p-values parametrically
  numReps = 0,
  
  # Control FDR via Benjamini-Hochberg
  adjustment = "BY"
)

cb15gocc_omics <- CreateOmics(
  
  # protein expression data
  assayData_df = tibble(sample = rownames(pData(cb15Eset)), as_tibble(t(exprs(cb15Eset)))),
  
  # pathway collection
  pathwayCollection_ls = goccCBgenes,
  
  # category phenotype
  response = tibble(sample = rownames(pData(cb15Eset)), no_relapse = as.factor(pData(cb15Eset)$posOutcome)),
  respType = "categorical",
  
  # retain pathways with > n proteins
  minPathSize = 1
)

cb15gocc_aes1pc <- AESPCA_pVals(
  
  # The Omics data container
  object = cb15gocc_omics,
  
  # One principal component per pathway
  numPCs = 1,
  
  # Use parallel computing with 2 cores
  parallel = TRUE, numCores = 8,
  
  # Estimate the p-values parametrically
  numReps = 0,
  
  # Control FDR via Benjamini-Hochberg
  adjustment = "BY"
)

# repeat for 15 study data set
cb15gomf_omics <- CreateOmics(
  
  # protein expression data
  assayData_df = tibble(sample = rownames(pData(cb15Eset)), as_tibble(t(exprs(cb15Eset)))),
  
  # pathway collection
  pathwayCollection_ls = gomfCBgenes,
  
  # category phenotype
  response = tibble(sample = rownames(pData(cb15Eset)), no_relapse = as.factor(pData(cb15Eset)$posOutcome)),
  respType = "categorical",
  
  # retain pathways with > n proteins
  minPathSize = 1
)

cb15gomf_aes1pc <- AESPCA_pVals(
  
  # The Omics data container
  object = cb15gomf_omics,
  
  # One principal component per pathway
  numPCs = 1,
  
  # Use parallel computing with 2 cores
  parallel = TRUE, numCores = 8,
  
  # Estimate the p-values parametrically
  numReps = 0,
  
  # Control FDR via Benjamini-Hochberg
  adjustment = "BY"
)

# repeat for 15 study data set
cb15rct_omics <- CreateOmics(
  
  # protein expression data
  assayData_df = tibble(sample = rownames(pData(cb15Eset)), as_tibble(t(exprs(cb15Eset)))),
  
  # pathway collection
  pathwayCollection_ls = reactomeCBgenes,
  
  # category phenotype
  response = tibble(sample = rownames(pData(cb15Eset)), no_relapse = as.factor(pData(cb15Eset)$posOutcome)),
  respType = "categorical",
  
  # retain pathways with > n proteins
  minPathSize = 1
)

cb15rct_aes1pc <- AESPCA_pVals(
  
  # The Omics data container
  object = cb15rct_omics,
  
  # One principal component per pathway
  numPCs = 1,
  
  # Use parallel computing with 2 cores
  parallel = TRUE, numCores = 8,
  
  # Estimate the p-values parametrically
  numReps = 0,
  
  # Control FDR via Benjamini-Hochberg
  adjustment = "BY"
)

# annotate $ save
save(cb15gobp_aes1pc, cb15gobp_omics, cb15gocc_aes1pc, cb15gocc_omics, cb15gomf_aes1pc, cb15gomf_omics, cb15rct_aes1pc, cb15rct_omics, file = "data/curatedBreastData/diffExp/pathwayPCAcb15.rdata")

cb15gobp <- mutate(cb15gobp_aes1pc$pVals_df, STANDARD_NAME = str_replace(terms, "^GOBP", "GO")) %>%
  left_join(msigdb) %>%
  select(-HISTORICAL_NAMES:-GENESET_LISTING_URL, -DESCRIPTION_FULL, -TAGS, -FOUNDER_NAMES:-BUILD_DATE, -MEMBERS:-MEMBERS_MAPPING) %>%
  mutate(GOID = str_remove(EXTERNAL_DETAILS_URL, "http://amigo.geneontology.org/amigo/term/"))
write_tsv(cb15gobp, "data/curatedBreastData/diffExp/cb15_pathwayPCA_gobp_msigdb74.tsv")

cb15gocc <- mutate(cb15gocc_aes1pc$pVals_df, STANDARD_NAME = str_replace(terms, "^GOCC", "GO")) %>%
  left_join(msigdb) %>%
  select(-HISTORICAL_NAMES:-GENESET_LISTING_URL, -DESCRIPTION_FULL, -TAGS, -FOUNDER_NAMES:-BUILD_DATE, -MEMBERS:-MEMBERS_MAPPING) %>%
  mutate(GOID = str_remove(EXTERNAL_DETAILS_URL, "http://amigo.geneontology.org/amigo/term/"))
write_tsv(cb15gocc, "data/curatedBreastData/diffExp/cb15_pathwayPCA_gocc_msigdb74.tsv")

cb15gomf <- mutate(cb15gomf_aes1pc$pVals_df, STANDARD_NAME = str_replace(terms, "^GOMF", "GO")) %>%
  left_join(msigdb) %>%
  select(-HISTORICAL_NAMES:-GENESET_LISTING_URL, -DESCRIPTION_FULL, -TAGS, -FOUNDER_NAMES:-BUILD_DATE, -MEMBERS:-MEMBERS_MAPPING) %>%
  mutate(GOID = str_remove(EXTERNAL_DETAILS_URL, "http://amigo.geneontology.org/amigo/term/"))
write_tsv(cb15gomf, "data/curatedBreastData/diffExp/cb15_pathwayPCA_gomf_msigdb74.tsv")

cb15rct <- mutate(cb15rct_aes1pc$pVals_df, STANDARD_NAME = terms) %>%
  left_join(msigdb) %>%
  select(-HISTORICAL_NAMES:-GENESET_LISTING_URL, -DESCRIPTION_FULL, -TAGS, -FOUNDER_NAMES:-BUILD_DATE, -MEMBERS:-MEMBERS_MAPPING) %>%
  mutate(ID = str_remove(EXTERNAL_DETAILS_URL, "http://www.reactome.org/cgi-bin/eventbrowser_st_id?ST_ID=REACT_"))
write_tsv(cb15rct, "data/curatedBreastData/diffExp/cb15_pathwayPCA_reactome_msigdb74.tsv")

