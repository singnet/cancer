### GOstats for curated breast data set
library(tidyverse)

# get name translation file
idmap <- read_csv("data/curatedBreastData/hgncSymbolMap21aug20long.csv")

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
  map(na.omit) %>%
  map(as.character)

# run GOstats
options(connectionObserver = NULL)
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

# make feature by sample file (GCT) and pheno label files (CLS)
load("data/curatedBreastData/diffExp/diffExpEsets.rdata")
esets <- list(cb15Eset, cb15fEset, tamoxEset, tamoxfEset)
names(esets) <- c("cb15", "cb15f", "tamox", "tamoxf")

unlist(map(esets, dim))
# cb15.Features    cb15.Samples  cb15f.Features   cb15f.Samples  tamox.Features   tamox.Samples 
#          8832            2237            4665            2225            8832             649 
# tamoxf.Features  tamoxf.Samples 
#            4665             642 

rm(cb15Eset, cb15fEset, tamoxEset, tamoxfEset)

# remove samples with missing outcomes
for(n in 1:4) esets[[n]] <- esets[[n]][, !is.na(pData(esets[[n]])$posOutcome)]
unlist(map(esets, dim))
# cb15.Features    cb15.Samples  cb15f.Features   cb15f.Samples  tamox.Features   tamox.Samples 
#          8832            2225            4665            2225            8832             642 
# tamoxf.Features  tamoxf.Samples 
#            4665             642 

setwd("data/curatedBreastData/diffExp/gsea/")
for(n in 1:4) {
  ArrayTools::output.gct(esets[[n]], names(esets)[n])
  ArrayTools::output.cls(pData(esets[[n]]), "posOutcome", names(esets)[n])
}

rm(esets)
setwd("../../../..")

# get GSEA results from cli java binary
gseasets <- list("cb15", "cb15f", "tamox", "tamoxf", "cb15_as", "cb15f_as", "tamox_as", "tamoxf_as") %>%
  set_names()

d <- "data/curatedBreastData/diffExp/gsea/"

genesets <- map(gseasets, ~ read_tsv(paste(d, ., "gene_set_sizes.tsv", sep = "/"))) %>%
  map(~ select(., -X5, -TITLE)) %>%
  map(~ replace_na(., list(STATUS = "OK")))

genelists <- map(gseasets[1:4],
             ~ read_tsv(paste(d, ., str_subset(list.files(paste0(d, .)), "^ranked_g"), sep = "/"))) %>%
  map(~select(., - TITLE, - X4))

e1 <- map(gseasets,
          ~ read_tsv(paste(d, ., str_subset(list.files(paste0(d, .), "tsv$"), "r_1_"), sep = "/")),
          na = "---") %>%
  map(~select(., - 2, - 3, - X12))

e0 <- map(gseasets,
          ~ read_tsv(paste(d, ., str_subset(list.files(paste0(d, .), "tsv$"), "r_0_"), sep = "/")),
          na = "---") %>%
  map(~select(., - 2, - 3, - X12))

# get GO & reactome ids
msig74 <- XML::xmlParse("/mnt/biodata/msigdb/msigdb_v7.4.xml")
msig74 <- XML::xmlToList(msig74)

# save list parse for future use
save(msig74, "/mnt/biodata/msigdb/msig74.rdata")

msig74 <- set_names(msig74, NULL) %>%
  transpose() %>%
  map(unlist) %>%
  map( ~ na_if(., "")) %>%
  as_tibble()

# save info table
write_tsv(msig74, "/mnt/biodata/msigdb/msig74.tsv.xz")

msig74 <- select(msig74, STANDARD_NAME, EXACT_SOURCE, DESCRIPTION_BRIEF, PMID, GEOID)
write_csv(msig74, "data/curatedBreastData/diffExp/msig74info.csv.xz")

# add gene set info
names(msig74)[1] <- "NAME"

genesets <- map_at(genesets, 1:4, ~ left_join(., msig74))
genesets <- map_at(genesets, 5:8, ~ left_join(., select(msig74, - PMID, - GEOID)))

e0 <- map(e0, ~ left_join(., select(msig74, NAME, EXACT_SOURCE)))

e1 <- map(e1, ~ left_join(., select(msig74, NAME, EXACT_SOURCE)))

# make genesets column with portion of measured genes in gene sets
genesets <- map(genesets, ~ mutate(., coverage = `AFTER RESTRICTING TO DATASET` / `ORIGINAL SIZE`))
write_csv(genesets$tamoxf, "data/curatedBreastData/diffExp/msigdb_tamox4k.csv.xz")

e0 <- map2(e0, genesets, ~ left_join(.x, select(.y, NAME, coverage)))
write_csv(e0$tamoxf_as, "data/curatedBreastData/diffExp/gsea/gseaGOreactome_tamox4k.csv.xz")

e1 <- map2(e1, genesets, ~ left_join(.x, select(.y, NAME, coverage)))
write_csv(e1$tamoxf_as, "data/curatedBreastData/diffExp/gsea/gseaGOreactome_tamox4k_1.csv.xz")

# GSEA R package is not maintained, use cli bundle instead
# broad gsea

# library(GSEA)
#
# # parallel
# library(furrr)
# options(parallelly.fork.enable = TRUE)
# plan(multicore, workers = 4)
# 
# future_map(names(esets),
#      ~ GSEA(input.ds = paste0(., ".gct"),
#             input.cls = paste0(., ".cls"),
#             doc.string = paste0(., "_results_"),
#             gs.db = "/mnt/biodata/msigdb/msigdb.v7.4.CBsymbol-as.gmt",
#             gs.size.threshold.min = 1,
#             gs.size.threshold.max = 1000,
#             random.seed = 123456789,
#             save.intermediate.results = TRUE,
#             use.fast.enrichment.routine = TRUE,
#             gsea.type = "GSEA"))

# [1] "Computing random permutations' enrichment for gene set: 11691 REACTOME_SIGNALING_BY_THE_B_CELL_RECEPTOR_BCR"
# [1] "Computing random permutations' enrichment for gene set: 11692 REACTOME_ION_CHANNEL_TRANSPORT"
# [1] "Computing nominal p-values..."
# Error in if (phi[i, j] >= 0) { : missing value where TRUE/FALSE needed
#   In addition: Warning messages:
#     1: In readLines(file) : incomplete final line found on 'cb15.cls'
#   2: UNRELIABLE VALUE: Future (‘<none>’) unexpectedly generated random numbers without specifying argument 'seed'. There is a risk that those random numbers are not statistically sound and the overall results might be invalid. To fix this, specify 'seed=TRUE'. This ensures that proper, parallel-safe random numbers are produced via the L'Ecuyer-CMRG method. To disable this check, use 'seed=NULL', or set option 'future.rng.onMisuse' to "ignore". 

# make ranked gene list files
# TODO: these need to be complete lists not filtered for significance
# load("data/curatedBreastData/diffExp/diffExpLimmaEbayes.rdata")
# 
# bestRNK <- map(bestGeneSets, ~ tibble(CBsymbol = names(.), foldDiff = .))
# map2(bestRNK, names(bestRNK),
#      ~ write_tsv(.x, paste0("data/curatedBreastData/diffExp/gsea/gsea_", .y, ".rnk"), col_names = FALSE))
# 
# map2(list.files("data/curatedBreastData/diffExp/gsea/", "rnk$", full.names = TRUE), names(bestRNK),
#     ~ GSEA(.x,
#            output.directory = "data/curatedBreastData/diffExp/gsea/",
#            doc.string = paste0(.y, "_results_"),
#            gs.db = "/mnt/biodata/msigdb/msigdb.v7.4.CBsymbol-chromCoord.gmt",
#            gs.size.threshold.min = 1,
#            gs.size.threshold.max = 1000,
#            random.seed = 123456789,
#            gsea.type = "preranked"))

