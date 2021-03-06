# apply all classifier and prediction models from genefu to dataset
library(tidyverse)

# quantile normed datasets
repo <- "~/R/metaGx/data/mgxSet/"
files <- list.files(paste0(repo, "rlNorm/"))
rl <- map(paste0(repo, "rlNorm/", files), read_csv)
names(rl) <- str_extract(files, "[^_]*")

files <- list.files(paste0(repo, "rlNormNoBatch/"))
rlNoBatch <- map(paste0(repo, "rlNormNoBatch/", files), read_csv)
names(rlNoBatch) <- str_extract(files, "[^_]*")

files <- list.files(paste0(repo, "rlNormBatched/"))
rlBatched <- map(paste0(repo, "rlNormBatched/", files), read_csv)
names(rlBatched) <- c("DFHCC2_CISPLATIN", "DFHCC2_REFERENCE", str_extract(files[3:25], "[^_]*"))

# convert to matrices
rl <- map(rl, column_to_rownames, "sample_name") %>%
  map(as.matrix)

rlNoBatch <- map(rlNoBatch, column_to_rownames, "sample_name") %>%
  map(as.matrix)

rlBatched <- map(rlBatched, column_to_rownames, "sample_name") %>%
  map(as.matrix)

# apply molecular.subtyping x 9  to batched & nonbatched data sets
# make study _. list of cluster assignments function
clusterModels <- c("scmgene", "scmod1", "scmod2", "pam50", "ssp2006", "ssp2003", "intClust", "AIMS","claudinLow")

# get gene map
fmap <- read_tsv("data/metaGxBreast/hgncMap17jun20.tsv") %>%
  select(Gene.Symbol = `Approved symbol`, EntrezGene.ID = `NCBI Gene ID`)
fmap <- as.matrix(fmap)
rownames(fmap) <- fmap[, 1]

# construct function from expression matrix -> tibble of subtypes
library(genefu)

molSub <- function(x, y) partial(molecular.subtyping, sbt.model = x, data = y)
expMat2subtype <- function(mod, exp_m, annot_m, ...) enframe(molSub(mod, exp_m)(annot_m, ...)[[1]],
                                                      name = "sample", value = mod)
# apply pam50 clustering
pam50 <- vector("list", length = length(c(rl, rlNoBatch))) %>%
  set_names(names(c(rl, rlNoBatch)))
for(m in names(c(rl, rlNoBatch))) {
  pam50[[m]] <- try(expMat2subtype("pam50", c(rl, rlNoBatch)[[m]], fmap))
}
pam50$KOO
#... "no probe in common -> annot or mapping parameters are necessary for the mapping process!"
# pam50 <- discard

pam50 <- bind_rows(pam50[- 16])

pam50batch <- vector("list", length = length(c(rl, rlBatched))) %>%
  set_names(names(c(rl, rlBatched)))
for(m in names(c(rl, rlBatched))) {
  pam50batch[[m]] <- try(expMat2subtype("pam50", c(rl, rlBatched)[[m]], fmap))
}
pam50batch <- bind_rows(pam50batch[- 16])

pam50labels <- left_join(pam50, rename(pam50batch, pam50_batch = pam50)) %>%
  filter(!duplicated(sample)) %>%
  rename(sample_name = sample)

write_csv(pam50labels, "data/metaGxBreast/pam50labels.csv")

# update covariance table
pheno <- read_csv("data/metaGxBreast/metaGXcovarTable.csv.xz", guess_max = 10000)
pheno <- left_join(pheno, pam50labels)
write_csv(pheno, "data/metaGxBreast/metaGXcovarTable.csv.xz")

# version with class probabilities
molSub <- function(x, y) partial(molecular.subtyping, sbt.model = x, data = y)
expMat2subtype_2 <- function(mod, exp_m, annot_m, ...) molSub(mod, exp_m)(annot_m, ...)[[2]]
                                                           
# apply pam50 clustering
pam50_2 <- vector("list", length = length(c(rl, rlNoBatch))) %>%
  set_names(names(c(rl, rlNoBatch)))
for(m in names(c(rl, rlNoBatch))) {
  pam50_2[[m]] <- try(expMat2subtype_2("pam50", c(rl, rlNoBatch)[[m]], fmap))
}

pam50_2$KOO
#... "no probe in common -> annot or mapping parameters are necessary for the mapping process!"
# pam50_2_ <- discard

pam50_2 <- purrr::map(pam50_2[- 16], as_tibble, rownames = "sample_name") %>% 
  bind_rows(.id = "study") 

pam50_2_batch <- vector("list", length = length(c(rl, rlBatched))) %>%
  set_names(names(c(rl, rlBatched)))
for(m in names(c(rl, rlBatched))) {
  pam50_2_batch[[m]] <- try(expMat2subtype_2("pam50", c(rl, rlBatched)[[m]], fmap))
}
pam50_2_batch <- purrr::map(pam50_2_batch[- 16], as_tibble, rownames = "sample_name") %>% 
  bind_rows(.id = "batch") 

pam50_2_labels <- left_join(pam50_2,
                            rename_with(pam50_2_batch, ~ paste0(., "_batched"),
                                   all_of(c("Basal", "Her2", "LumA", "LumB", "Normal")))) %>%
  filter(!duplicated(sample_name))
write_csv(pam50_2_labels, "data/metaGxBreast/pam50labelProbabilities.csv.xz")

# apply scmgene model
scmgene <- vector("list", length = length(c(rl, rlNoBatch))) %>%
  set_names(names(c(rl, rlNoBatch)))
for(m in names(c(rl, rlNoBatch))) {
  scmgene[[m]] <- try(expMat2subtype("scmgene", c(rl, rlNoBatch)[[m]], fmap))
}
scmgene <- bind_rows(scmgene)

scmgenebatch <- vector("list", length = length(c(rl, rlBatched))) %>%
  set_names(names(c(rl, rlBatched)))
for(m in names(c(rl, rlBatched))) {
  scmgenebatch[[m]] <- try(expMat2subtype("scmgene", c(rl, rlBatched)[[m]], fmap))
}
scmgenebatch <- bind_rows(scmgenebatch[- 16])

scmgenelabels <- left_join(scmgene, rename(scmgenebatch, scmgene_batch = scmgene)) %>%
  filter(!duplicated(sample)) 

# remove translation from robust norm
# doesn't make a difference for pam50 or success of any other cluster models (all others still fail)
rl2 <- lapply(rl, `+`, 1)
rl2NoBatch <- lapply(rlNoBatch, `+`, 1)
rl2Batched <- lapply(rlBatched, `+`, 1)

# apply pam50 clustering without0 centering
# as expected this has no effect
pam50Pos <- vector("list", length = length(c(rl2, rl2NoBatch))) %>%
  set_names(names(c(rl2, rl2NoBatch)))
for(m in names(c(rl2, rl2NoBatch))) {
  pam50Pos[[m]] <- try(expMat2subtype("pam50", c(rl2, rl2NoBatch)[[m]], fmap))
}
pam50Pos$KOO
#... "no probe in common -> annot or mapping parameters are necessary for the mapping process!"
# pam50Pos <- discard

pam50Pos <- bind_rows(pam50Pos[- 16])

pam50Posbatch <- vector("list", length = length(c(rl2, rl2Batched))) %>%
  set_names(names(c(rl2, rl2Batched)))
for(m in names(c(rl2, rl2Batched))) {
  pam50Posbatch[[m]] <- try(expMat2subtype("pam50", c(rl2, rl2Batched)[[m]], fmap))
}
pam50Posbatch <- bind_rows(pam50Posbatch[- 16])

pam50Poslabels <- left_join(pam50Pos, rename(pam50Posbatch, pam50_batch = pam50)) %>%
  filter(!duplicated(sample)) 

pam50labels2 <- left_join(pam50labels, pam50Poslabels, by = "sample")
filter(pam50labels2, pam50.x:pam50_batch.x != pam50.y:pam50_batch.y)
# A tibble: 0 x 5
# … with 5 variables: sample <chr>, pam50.x <fct>, pam50_batch.x <fct>, pam50.y <fct>,
#   pam50_batch.y <fct>

# check out three gene cluster data
clusters <- read_csv("data/metaGxBreast/jnci-JNCI-11-0924-s02.csv", comment = "#")
#samplename: Unique identifier for tumor samples
#dataset: Acronym of the dataset
#series: Batch or subcohorts of tumor samples
#id: Identifier used in the original publication
#age: Age at diagnosis (years)
#er: Estrogen receptor status
#pgr: Progesterone receptor status
#her2: Human epidermal growth factor 2 status
#grade: Histological grade
#node: Nodal status
#size: Tumor size (cm)
#t.rfs: Time for relapse-free survival (days)
#e.rfs: Event for relapse-free survival
#t.dmfs: Time for distant metastasis-free survival (days)
#e.dmfs: Event for distant metastasis-free survival
#t.os: Time for overall survival (days)
#e.os: Event for overall survival
#treatment: Treatment (0=untreated; 1=treated)
#MAMMAPRINT: Risk classification (Low/HighRisk) computed using the published algorithm of the prognostic gene signature published by van't Veer et al. (13)
#ONCOTYPE: Risk classification (Low/Intermediate/HighRisk) computed using the published algorithm of the prognostic gene signature published by Paik et al. (15)
#GGI: Risk classification (Low/HighRisk) computed using the published algorithm of the prognostic gene signature published by Sotiriou et al. (16)
#SCMGENE: Subtype classification published in the present work.
#SCMOD2: Subtype classification published by Wirapati et al. (8)
#SCMOD1: Subtype classification published by Desmedt et al. (1)
#PAM50: Subtype classification published by Parker et al. (3)
#SSP2006: Subtype classification published by Hu et al. (2)
#SSP2003: Subtype classification published by Sorlie et al. (6)
dim(clusters)
# [1] 5715   27

table(clusters$dataset)

# note extra grade level in package data
table(clusters$grade)
#   1    2    3 
# 544 1440 1695 
table(pheno$grade)
#   1    2    3    4 
# 701 2299 2903   15 

covars <- left_join(pheno, clusters, by = c("sample_name" = "samplename"), suffix = c("_mgx", "_3g"))
setdiff(covars$study, covars$dataset)
# [1] "GSE25066" "GSE58644" "GSE32646" "GSE48091" "DUKE"     "EXPO"     "METABRIC" "TCGA"

# extra grade level is from new metagx data sets
table(covars$grade_mgx)
#   1    2    3 
# 379  940 1292 

# three gene data set removes some extra samples
covars <- filter(covars, sample_name %in% clusters$samplename)
dim(covars)
# [1] 4158  57

dim(clusters)[1] - dim(covars)[1]
# 1557

# three studies aren't included in metagx, the rest are from DUKE, EXPO, MAQC2, and STNO2
table(clusters$dataset[!(clusters$samplename %in% covars$sample_name)])
# DUKE      DUKE2 EORTC10994       EXPO        KOO      MAQC2       MDA4       MDA5        MSK 
#  171          6          2        353         53        186          4        298          1 
# PNC      STNO2        TAM       UCSF       VDX3 
#   1        103        242          1        136 

setdiff(clusters$dataset, pheno$study)
# [1] "TAM"  "MDA5" "VDX3"

setdiff(clusters$series, pheno$batch)
# [1] "OXFT"  "KIT"   "IGRT"  "AUST"  "VDX3"  "GUYT"  "GUYT2"

table(clusters$dataset[clusters$series %in% setdiff(clusters$series, pheno$batch)])
# MDA5  TAM VDX3 
# 298  242  136 

dim(clusters)[1] - 298 + 242 + 136
# [1] 5795

dim(clusters)[1] - (298 + 242 + 136)
# [1] 5039

dim(clusters)[1] - dim(covars)[1] - (298 + 242 + 136)
# [1] 881
171 + 353 + 186 + 103
# [1] 813

# intrinsic gene lists ssp2003, ssp2006, pam50 ("Basal", "Her2", "LumA", "LumB" or "Normal")

# 

# genious, ggi, 
dim(sig.ggi)
# [1] 128   9
ggi.current <- unique(
  hgncMap$`Approved symbol`[match(as.character(sig.ggi$EntrezGene.ID), hgncMap$`NCBI Gene ID`)]
  )
length(ggi.current)
# [1] 105
setdiff(sig.ggi$HUGO.gene.symbol, ggi.current)
# [1] "MARS"     "DDX39"    "CDC2"     "DLG7"     "ZWINT"    "KNTC2"    "TUBA6"    "H2AFZ"    "C20orf24"
# [10] "C16orf61" "MLF1IP"   "C12orf48" "FAM64A"   "C11orf60" "C11orf63"
write_lines(na.omit(ggi.current), "data/metaGxBreast/ggiGeneList.txt")

# prognistic score ihc4, gene70, gene76, endoPredict, oncotype dx, npi, pi3ca, tamr13
dim(sig.gene70)
# [1] 70  9
gene70.current <- unique(
  hgncMap$`Approved symbol`[match(as.character(sig.gene70$EntrezGene.ID), hgncMap$`NCBI Gene ID`)]
)
length(gene70.current)
# [1] 53
setdiff(sig.gene70$HUGO.gene.symbol, gene70.current)
# [1] NA         "WISP1"    "ZNF533"   "PECI"     "C20orf46" "HRASLS"   "C9orf30"  "ORC6L"    "GPR126"  
# [10] "AYTL2"    "KNTC2"    "C16orf61" "QSCN6L1" 
write_lines(na.omit(gene70.current), "data/metaGxBreast/gene70List.txt")


dim(sig.gene76)
# [1] 76 10

# gene modules: mod1, mod2 for scmod1, scmod2, scmgene
# also has mamaPrint and oncotype lists
attach("~/R/metaGx/metaGxData/data/breastSigs.RData")
sapply(breastSigs, dim)
#      ESR1 ERBB2 AURKA PLAU VEGF STAT1 CASP3 mammaPrint oncotype
# [1,]  469    28   229   68   14    95    10         62       16
# [2,]    3     3     3    3    3     3     3          3        3

# try all models
clusterLabels <- vector("list", length = length(clusterModels) %>%
                          set_names(clusterModels)
                        
                        for(n in names(clusterLabels)) {
                          clusterLabels[[n]] <- vector("list", length(rl2)) %>%
                            set_names(names(rl2))
                          for(m in names(clusterLabels[[n]])) {
                            clusterLabels[[m]] <- try(expMat2subtype(n, rl2[[m]], fmap))
                          }
                        }
                        
                        