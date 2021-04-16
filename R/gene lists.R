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

bestGeneSets <- map(diffExp, ~ pull(., logFC, symbol))

bestGeneSetsE <- map(bestGeneSets, ~ idmap$`NCBI Gene ID`[match(names(.), idmap$`Approved symbol`)]) %>%
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

# ml gene lists
tam4pv100 <- read_lines("https://mozi.ai/datasets/embedding-vectors/tamoxifen_group/100_pln_gen_go_pathway.txt")

# GSEA
library(EGSEA)

# make current genesets
msig74 <- XML::xmlParse("/mnt/biodata/msigdb/msigdb_v7.4.xml")
msig74 <- XML::xmlToList(msig74)
save(msig74,file = "/mnt/biodata/msigdb/msig74.rdata")

# make msigdb gmt file with matching symbol versions
symMap <- read_csv("data/curatedBreastData/hgncSymbolMap21aug20long.csv")
msigRaw <- read_lines("/mnt/biodata/msigdb/msigdb.v7.4.symbols.gmt") %>%
  str_split("\\t")

msgTib <- tibble(name = map(msigRaw, ~ `[`(., 1)), url = map(msigRaw, ~ `[`(., 2)), genes = map(msigRaw, ~ `[`(., -1:-2)))

# check symbol matches
msg74uni <- unique(unlist(msgTib$genes))
table(msg74uni %in% symMap$alt)
# FALSE  TRUE 
# 15827 23954

# make named mapping vector for "recode()"
msg2cb <- symMap$`Approved symbol`[match(msg74uni, symMap$alt)] %>%
  set_names(msg74uni)

# set up parallel package
library(furrr)
plan(multicore, workers = 12)
msgTib$CBgenes <- future_map(msgTib$genes, ~ recode(., !!!msg2cb, .default = NA_character_))

msgTib <- mutate(msgTib, name = map_chr(name, ~ .), url = map_chr(url, ~ .),
                 count = map_dbl(genes, length), missing = map_dbl(CBgenes, ~ sum(is.na(.))))

msgTib <- mutate(msgTib, CBgenes = map(CBgenes, na.omit))

# percent missing
mutate(msgTib, pct_missing = missing / count) %>%
  pull(pct_missing) %>%
  summary()
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.00000 0.06250 0.09798 0.11066 0.13725 1.00000 

filter(msgTib, count == missing) %>%
  select(name, count, missing)
#   name                                         count missing
# 1 GOBP_SOMATOSTATIN_RECEPTOR_SIGNALING_PATHWAY     5       5
# 2 GOBP_CGMP_CATABOLIC_PROCESS                      5       5
# 3 GOMF_SOMATOSTATIN_RECEPTOR_ACTIVITY              5       5
# 4 REACTOME_MUSCARINIC_ACETYLCHOLINE_RECEPTORS      5       5

saveRDS(msgTib, "/mnt/biodata/msigdb/msig74cb.rds")

# save CB ymbol gmt file
filter(msgTib, count != missing) %>%
  mutate(geneString = map_chr(CBgenes, ~paste(., collapse = "\t"))) %>%
  transmute(mgtLine = paste(name, url, geneString, sep = "\t")) %>%
  pull(mgtLine) %>%
  str_trim() %>%
  write_lines("/mnt/biodata/msigdb/msigdb.v7.4.CBsymbol.gmt")

# get msigdb annotations
msig74 <- XML::xmlParse("/mnt/biodata/msigdb/msigdb_v7.4.xml")
msig74 <- XML::xmlToList(msig74)
msigdb <- bind_rows(msig74)

# TODO: fix
cd15idx <- buildGMTIdx(bestGeneSets$cb15, "/mnt/biodata/msigdb/msigdb.v7.4.CBsymbol.gmt", label = "msigDB_7.4", name = "current msigDB gene sets")
# Error in `[<-.data.frame`(`*tmp*`, , "NumGenes", value = "/") : 
#   replacement has 1 row, data has 0

# make list of gmt files
gmts <- list(h = "h.all.v7.4.symbols.gmt", c1 = "c1.all.v7.4.symbols.gmt",
    c2 = list(reactome = "c2.cp.reactome.v7.4.symbols.gmt", wp = "c2.cp.wikipathways.v7.4.symbols.gmt"),
    c3 = list(mirdb = "c3.mir.mirdb.v7.4.symbols.gmt", gtrd = "c3.tft.gtrd.v7.4.symbols.gmt"),
    c4 = list(cgn ="c4.cgn.v7.4.symbols.gmt", cm = "c4.cm.v7.4.symbols.gmt"),
    c5 = list(bp = "c5.go.bp.v7.4.symbols.gmt", cc = "c5.go.cc.v7.4.symbols.gmt",
              mf = "c5.go.mf.v7.4.symbols.gmt", hpo = "c5.hpo.v7.4.symbols.gmt"),
    c6 = "c6.all.v7.4.symbols.gmt", c7 = "c7.all.v7.4.symbols.gmt", c8 = "c8.all.v7.4.symbols.gmt"
            ) %>%
 unlist()

# make list of subset names to filter msgTib
gmtsNames <- map(gmts, ~ read_lines(paste0("/mnt/biodata/msigdb/", .))) %>%
  map(~ str_split(., "\\t")) %>%
  map(~map_chr(., `[`, 1))

# TODEO: convert protein ids to gene symbols
# write_lines()

idx <- function(gl, gmt) map(gmt, ~ buildGMTIdx(gl, paste0("/mnt/biodata/msigdb/", .),
                                      label = paste(unlist(str_split(., "[.]"))[1:2], sep = "_")))

idxBest <- map(bestGeneSets, ~ idx(names(.), gmts))

gsaBest <- pmap(list(bestGeneSets, idxBest, list(cb15uni, cb15Funi, cb15uni, tamoxFuni), names(idxBest)),
                ~ egsea.ora(names(..1), universe = ..3, logFC = ..1,
                            title = ..4,  gs.annots = ..2,
                            symbolsMap = NULL, minSize = 2, display.top = 20, sort.by = "p.adj",
                            report.dir = "reports/egsea", kegg.dir = NULL, sum.plot.axis = "p.adj",
                            sum.plot.cutoff = NULL, num.threads = 10, report = TRUE,
                            interactive = FALSE, verbose = TRUE))

# check entrez matches
msigRawEn <- read_lines("/mnt/biodata/msigdb/msigdb.v7.4.entrez.gmt") %>%
  str_split("\\t")

msgTibEn <- tibble(name = map(msigRawEn, ~ `[`(., 1)), url = map(msigRawEn, ~ `[`(., 2)), genes = map(msigRawEn, ~ `[`(., -1:-2)))

msg74uniEn <- unique(unlist(msgTibEn$genes))
table(msg74uniEn %in% symMap$`NCBI Gene ID`)
# FALSE  TRUE 
# 16336 23445 

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

