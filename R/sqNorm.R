## make and check smooth quantile normalized files
library(tidyverse)
# get expression matrices with imputed missing values
# metaGX <- readxl::read_excel("data/metaGxBreast/MetaGxData.xlsx", sheet = 1, n_max = 39) %>%
#   select(2:7)
# mgxesets <-MetaGxBreast::loadBreastEsets(metaGX$Dataset, imputeMissing = TRUE)
# saveRDS(mgxesets, "data/metaGxBreast/mgxesets.rds")
mgxesets <- readRDS("data/metaGxBreast/mgxesets.rds")$esets

# update symbols
symbolMap <- read_tsv("data/curatedBreastData/hgncSymbolMap21aug20.tsv") %>%
  mutate(`Previous symbols` = str_split(`Previous symbols`, ", "),
         `Alias symbols` = str_split(`Alias symbols`, ", "),
         `NCBI Gene ID` = as.character(`NCBI Gene ID`))

# combine previous and alias list columns, and unnest
symbolMap <- mutate(symbolMap, alt = map2(`Previous symbols`, `Alias symbols`, c), .keep = "unused") %>%
  mutate(alt = map(alt, ~ if(identical(., c(NA, NA))) NA_character_ else na.omit(.))) %>%
  unnest(cols = alt)

# add current/alt symbol identity row for each gene
symbolMap <- bind_rows(symbolMap, mutate(symbolMap, alt = `Approved symbol`) %>% distinct()) %>%
  arrange(`Approved symbol`) 

# pull out feature data and add column of current symbols based on entrez id
fdata <- map(mgxesets, Biobase::fData) %>%
  map(~ mutate(., probeset = as.character(probeset), gene = as.character(gene),
               EntrezGene.ID = as.character(EntrezGene.ID))) %>%
  map( ~ mutate(., symbol = symbolMap$`Approved symbol`[match(EntrezGene.ID, symbolMap$`NCBI Gene ID`)]))

# updating symbols removes 12% - 32% of best_probes
map(fdata, ~ round(sum(is.na(filter(., best_probe) %>% pull(symbol))) / sum(pull(., best_probe)), digits = 3)) %>%
  unlist()
#   CAL      DFHCC     DFHCC2     DFHCC3       DUKE      DUKE2       EMC2 EORTC10994       EXPO     FNCLCC 
# 0.124      0.191      0.191      0.191      0.127      0.184      0.191      0.122      0.191      0.137 
#   HLP        IRB        KOO       LUND      LUND2      MAINZ      MAQC2       MCCC       MDA4        MSK 
# 0.159      0.191      0.094      0.136      0.109      0.122      0.122      0.171      0.124      0.122 
#   MUG       NCCS        NCI        NKI        PNC        STK      STNO2   TRANSBIG       UCSF       UNC4 
# 0.154      0.122      0.126      0.112      0.191      0.166      0.172      0.122      0.129      0.140 
#   UNT        UPP        VDX   METABRIC       TCGA   GSE25066   GSE32646   GSE58644   GSE48091 
# 0.153      0.166      0.124      0.317      0.141      0.122      0.191      0.146      0.111 

# including rejected probes with with same entrez as best probes doesn't seem to add any more...
map(fdata, ~ round(length(intersect(filter(., best_probe) %>% pull(EntrezGene.ID),
                   filter(., !is.na(symbol)) %>% pull(EntrezGene.ID))) / sum(pull(., best_probe)), digits = 3)) %>% unlist()
#   CAL      DFHCC     DFHCC2     DFHCC3       DUKE      DUKE2       EMC2 EORTC10994       EXPO     FNCLCC 
# 0.876      0.809      0.809      0.809      0.873      0.816      0.809      0.878      0.809      0.863 
#   HLP        IRB        KOO       LUND      LUND2      MAINZ      MAQC2       MCCC       MDA4        MSK 
# 0.841      0.809      0.906      0.864      0.891      0.878      0.878      0.829      0.876      0.878 
#   MUG       NCCS        NCI        NKI        PNC        STK      STNO2   TRANSBIG       UCSF       UNC4 
# 0.846      0.878      0.874      0.888      0.809      0.834      0.828      0.878      0.871      0.860 
#   UNT        UPP        VDX   METABRIC       TCGA   GSE25066   GSE32646   GSE58644   GSE48091 
# 0.847      0.834      0.876      0.683      0.859      0.878      0.809      0.854      0.889 

# how many missing current symbols are due to updated entrez ids? < 1%
missing <- map(fdata, ~ filter(., is.na(symbol)) %>% pull(gene))
map(missing, length) %>% unlist()
#  CAL      DFHCC     DFHCC2     DFHCC3       DUKE      DUKE2       EMC2 EORTC10994       EXPO     FNCLCC 
# 3211       7215       7215       7215       1783       8577       7215       2975       7215        839 
#  HLP        IRB        KOO       LUND      LUND2      MAINZ      MAQC2       MCCC       MDA4        MSK 
# 4140       7215         33       1510       5068       2975       2975       3190       3211       2975 
#  MUG       NCCS        NCI        NKI        PNC        STK      STNO2   TRANSBIG       UCSF       UNC4 
# 2101       2975        647       1628       7215       5681        637       2975       1050        744 
#  UNT        UPP        VDX   METABRIC       TCGA   GSE25066   GSE32646   GSE58644   GSE48091 
# 5704       5681       3211       9932       2748       2975       7212       3121       2414 

map2(missing, fdata, ~ round(length(intersect(.x, symbolMap$`Approved symbol`)) / sum(pull(.y, best_probe)), digits = 3)) %>% unlist
#   CAL      DFHCC     DFHCC2     DFHCC3       DUKE      DUKE2       EMC2 EORTC10994       EXPO     FNCLCC 
# 0.006      0.005      0.005      0.005      0.007      0.005      0.005      0.006      0.005      0.007 
#   HLP        IRB        KOO       LUND      LUND2      MAINZ      MAQC2       MCCC       MDA4        MSK 
# 0.005      0.005      0.000      0.007      0.059      0.006      0.006      0.006      0.006      0.006 
#   MUG       NCCS        NCI        NKI        PNC        STK      STNO2   TRANSBIG       UCSF       UNC4 
# 0.007      0.006      0.008      0.006      0.005      0.005      0.007      0.006      0.012      0.007 
#   UNT        UPP        VDX   METABRIC       TCGA   GSE25066   GSE32646   GSE58644   GSE48091 
# 0.005      0.005      0.006      0.005      0.005      0.006      0.005      0.000      0.005 

# add current symbols that are missing because entrez id has changed
fdata <- map(fdata, ~ mutate(., symbol = map2(gene, symbol, ~ if_else(.x %in% symbolMap$`Approved symbol`, .x, .y))))

map2(map(fdata, ~ filter(., is.na(symbol)) %>% pull(gene)), fdata,
     ~ round(length(intersect(.x, symbolMap$`Approved symbol`)) / sum(pull(.y, best_probe)), digits = 3)) %>%
  unlist
# CAL      DFHCC     DFHCC2     DFHCC3       DUKE      DUKE2       EMC2 EORTC10994       EXPO     FNCLCC 
#   0          0          0          0          0          0          0          0          0          0 
# HLP        IRB        KOO       LUND      LUND2      MAINZ      MAQC2       MCCC       MDA4        MSK 
#   0          0          0          0          0          0          0          0          0          0 
# MUG       NCCS        NCI        NKI        PNC        STK      STNO2   TRANSBIG       UCSF       UNC4 
#   0          0          0          0          0          0          0          0          0          0 
# UNT        UPP        VDX   METABRIC       TCGA   GSE25066   GSE32646   GSE58644   GSE48091 
#   0          0          0          0          0          0          0          0          0 

# add outdated symbols with "_obs" tag
fdata <- map(fdata, ~ mutate(., symbol = map2(symbol, gene, ~ if_else(is.na(.x), paste0(.y, "_obs"), .x))))
map(fdata, ~ sum(is.na(pull(., symbol)))) %>% unlist()
# CAL      DFHCC     DFHCC2     DFHCC3       DUKE      DUKE2       EMC2 EORTC10994       EXPO     FNCLCC 
#   0          0          0          0          0          0          0          0          0          0 
# HLP        IRB        KOO       LUND      LUND2      MAINZ      MAQC2       MCCC       MDA4        MSK 
#   0          0          0          0          0          0          0          0          0          0 
# MUG       NCCS        NCI        NKI        PNC        STK      STNO2   TRANSBIG       UCSF       UNC4 
#   0          0          0          0          0          0          0          0          0          0 
# UNT        UPP        VDX   METABRIC       TCGA   GSE25066   GSE32646   GSE58644   GSE48091 
#   0          0          0          0          0          0          0          0          0 

# why is the symbol column now a list?
fdata <- map(fdata, ~ mutate(., symbol = unlist(symbol)))

# add to expression sets
library(parallel)
mgxesetsCurrent <- map2(mgxesets, fdata, Biobase::`fData<-`)

# convert to tibbles, & join probe annotations

micro <- mclapply(mgxesetsCurrent, function(x) as_tibble(Biobase::exprs(x), rownames = "probeset"), mc.cores = 8, mc.preschedule = FALSE)
micro <- mcmapply(left_join, fdata, micro, mc.cores = 8, mc.preschedule = FALSE)

# use annotation filter, update symbols, select duplicate with greatest variation
mgxCompSet <- mclapply(micro, function(x) 
  filter(x, best_probe) %>%
    select(- probeset, - gene, - EntrezGene.ID, - best_probe) %>%
    rowwise() %>%
    mutate(sd = sd(c_across(where(is.numeric)))) %>%
    group_by(symbol) %>%
    filter(sd == max(sd) & !is.na(symbol)) %>%
    select(- sd),
  mc.cores = 10, mc.preschedule = FALSE)

# clean up names
names(mgxCompSet$TRANSBIG) <- str_remove(names(mgxCompSet$TRANSBIG), "TRANSBIG_")
# names(mgxSet$STNO2) <- str_remove(names(mgxSet$STNO2), "STNO2_")


# get batch info
pheno <- read_csv("data/metaGxBreast/metaGXcovarTable.csv.xz", guess_max = 4000)
batches <- group_by(pheno, study, batch) %>%
  summarise(batch_count = n())
batchSets <- unique(batches$study[duplicated(batches$study)])

# function to transpose tibbles
transpose_df <- function(df) {
  t_df <- data.table::transpose(df)
  colnames(t_df) <- rownames(df)
  rownames(t_df) <- colnames(df)
  t_df <- t_df %>%
    tibble::rownames_to_column(.data = .) %>%
    tibble::as_tibble(.)
  return(t_df)
}

# make smooth quantile normalized datasets using recurrence_status as group variable
library(qsmooth)
library(doParallel)
library(foreach)

registerDoParallel(cores = 10)

dir.create("data/metaGxBreast/sqNorm_recur")

recurSets <- filter(pheno, !is.na(recurrence_status)) %>%
  pull(study) %>%
  unique()

# non-batched
nb <- setdiff(recurSets, batchSets)
qsSet <- foreach(n = nb, .final = function(x) setNames(x, nb)) %dopar% {
  groupVar <- as.factor(filter(pheno, study == n) %>% pull(recurrence_status))
  expMat <- column_to_rownames(mgxCompSet[[n]], "symbol") %>%
    select(which(!is.na(groupVar)))
  qs <- qsmooth(expMat, na.omit(groupVar))
  expMat <- transpose_df(as_tibble(qs@qsmoothData, rownames = NA))
  names(expMat)[1] <- "sample_name"
  list(qsData = expMat, qsWeights = qs@qsmoothWeights)
}

qsSet <- transpose(qsSet)
qsWeights <- qsSet$qsWeights
qsSet <- qsSet$qsData

# batched sets are combat normalized first
b <- intersect(recurSets, batchSets)
qsSetB <- foreach(n = b, .errorhandling = "pass", .final = function(x) setNames(x, b)) %dopar% {
  groupVar <- as.factor(filter(pheno, study == n) %>% pull(recurrence_status))
  expMat <- column_to_rownames(mgxCompSet[[n]], "symbol") %>%
    select(which(!is.na(groupVar)))
  batch <- as.factor(filter(pheno, study == n) %>% pull(batch))
  qs <- qsmooth(expMat, na.omit(groupVar), batch[!is.na(groupVar)])
  expMat <- transpose_df(as_tibble(qs@qsmoothData, rownames = NA))
  names(expMat)[1] <- "sample_name"
  list(qsData = expMat, qsWeights = qs@qsmoothWeights)
}

qsSetB <- transpose(qsSetB)
qsWeights <- append(qsWeights, qsSetB$qsWeights)
qsWeights_recur <- qsWeights[order(names(qsWeights))]
qsSet <- append(qsSet, qsSetB$qsData)
qsSet_recur <- qsSet[order(names(qsSet))]


# save matrices with 6 digits to save space
for(n in names(qsSet_recur)) {
  mutate(qsSet_recur[[n]], across(where(is.numeric), formatC, digits = 6, format = "f")) %>%
    write_csv(paste0("data/metaGxBreast/sqNorm_recur/", n, "_recur_sqNorm.csv.xz"))
}

# scaled version
qsSet_recur_scaled <- map(qsSet_recur, ~ bind_cols(sample_name = pull(., 1), scale(select(., -1))))

dir.create("data/metaGxBreast/sqNorm_recur_scaled")

for(n in names(qsSet_recur_scaled)) {
  mutate(qsSet_recur[[n]], across(where(is.numeric), formatC, digits = 6, format = "f")) %>%
    write_csv(paste0("data/metaGxBreast/sqNorm_recur_scaled/", n, "_recur_sqNorm_scaled.csv.xz"))
}

save(qsSet_recur, qsSet_recur_scaled, qsWeights_recur, file = "data/metaGxBreast/mgxSQrecur.rdata")

# make smooth quantile normalized datasets using vital_status as group variable
dir.create("data/metaGxBreast/sqNorm_death")

deathSets <- filter(pheno, !is.na(vital_status)) %>%
  pull(study) %>%
  unique()

# non-batched
nb <- setdiff(deathSets, batchSets)
qsSet <- foreach(n = nb, .final = function(x) setNames(x, nb)) %dopar% {
  groupVar <- as.factor(filter(pheno, study == n) %>% pull(vital_status))
  expMat <- column_to_rownames(mgxCompSet[[n]], "symbol") %>%
    select(which(!is.na(groupVar)))
  qs <- qsmooth(expMat, na.omit(groupVar))
  expMat <- transpose_df(as_tibble(qs@qsmoothData, rownames = NA))
  names(expMat)[1] <- "sample_name"
  list(qsData = expMat, qsWeights = qs@qsmoothWeights)
}

qsSet <- transpose(qsSet)
qsWeights <- qsSet$qsWeights
qsSet <- qsSet$qsData

# batched sets are combat normalized first
b <- intersect(deathSets, batchSets)
qsSetB <- foreach(n = b, .errorhandling = "pass", .final = function(x) setNames(x, b)) %dopar% {
  groupVar <- as.factor(filter(pheno, study == n) %>% pull(vital_status))
  expMat <- column_to_rownames(mgxCompSet[[n]], "symbol") %>%
    select(which(!is.na(groupVar)))
  batch <- as.factor(filter(pheno, study == n) %>% pull(batch))
  qs <- qsmooth(expMat, na.omit(groupVar), batch[!is.na(groupVar)])
  expMat <- transpose_df(as_tibble(qs@qsmoothData, rownames = NA))
  names(expMat)[1] <- "sample_name"
  list(qsData = expMat, qsWeights = qs@qsmoothWeights)
}

qsSetB <- transpose(qsSetB)
qsWeights <- append(qsWeights, qsSetB$qsWeights)
qsWeights_death <- qsWeights[order(names(qsWeights))]
qsSet <- append(qsSet, qsSetB$qsData)
qsSet_death <- qsSet[order(names(qsSet))]

rm(qsSet, qsWeights)

# save matrices with 6 digits to save space
for(n in names(qsSet_death)) {
  mutate(qsSet_death[[n]], across(where(is.numeric), formatC, digits = 6, format = "f")) %>%
    write_csv(paste0("data/metaGxBreast/sqNorm_death/", n, "_death_sqNorm.csv.xz"))
}

# scaled version
qsSet_death_scaled <- map(qsSet_death, ~ bind_cols(sample_name = pull(., 1), scale(select(., -1))))
dir.create("data/metaGxBreast/sqNorm_death_scaled")

for(n in names(qsSet_death_scaled)) {
  mutate(qsSet_death[[n]], across(where(is.numeric), formatC, digits = 6, format = "f")) %>%
    write_csv(paste0("data/metaGxBreast/sqNorm_death_scaled/", n, "_death_sqNorm_scaled.csv.xz"))
}
save(qsSet_death, qsSet_death_scaled, qsWeights_death, file = "data/metaGxBreast/mgxSQdeath.rdata")

# TODO: make curatedBreastData sq versions
ct <- read_csv("data/curatedBreastData/bcClinicalTable.csv", guess_max = 10000)
bcFiles <- list.files("~/R/cancer.old/data/bcDump/example15bmc/", pattern = "^study", full.names = TRUE)
bc15 <- map(bcFiles, read_csv)
names(bc15) <- paste0("GSE",
                  str_remove(bcFiles, "^/home/mozi/R/cancer.old/data/bcDump/example15bmc//study_") %>%
  str_remove("_all-bmc15.csv.xz|-bmc15.csv.xz"))
