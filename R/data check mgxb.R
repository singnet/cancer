## check out bc data
# getspreadsheet info
library(tidyverse)
metaGX <- readxl::read_excel("data/metaGxBreast/MetaGxData.xlsx", sheet = 1, n_max = 39) %>%
  select(2:7)

library(MetaGxBreast)
esets <- loadBreastEsets(metaGX$Dataset)
# snapshotDate(): 2020-04-27
# Ids with missing data: CAL, EORTC10994, FNCLCC, LUND, LUND2, NCI, NKI, UCSF, METABRIC

duplicates <- esets$duplicates
esets <- esets$esets

# get covariate tables
fdata <- lapply(esets, fData)
for(n in names(fdata)) {
  fdata[[n]] <- tibble(probeset = as.character(fdata[[n]]$probeset),
                       gene = as.character(fdata[[n]]$gene),
                       entrez = as.character(fdata[[n]]$EntrezGene.ID),
                       best_probe = fdata[[n]]$best_probe)
}

sapply(fdata, function(x) table(x$best_probe))
#      CAL DFHCC DFHCC2 DFHCC3 DUKE DUKE2  EMC2 EORTC10994  EXPO FNCLCC   HLP   IRB KOO  LUND LUND2
# F   8481 22165  22165  22165 3249 25790 22165       8215 22165    957  7085 22165  26   766 14095
# T  12688 20282  20282  20282 8836 19700 20282      12752 20282   5107 19451 20282 254 10388  7913
#    MAINZ MAQC2  MCCC  MDA4   MSK   MUG  NCCS  NCI   NKI   PNC   STK STNO2 TRANSBIG UCSF UNC4   UNT
# F   8215  8215  4095  8481  8215  3573  8215 1042  1844 22165 17744   435     8215 1740  395 18075
# T  12752 12752 14953 12688 12752 10715 12752 4112 13116 20282 18434  3228    12752 6275 5025 18009
#         UPP   VDX METABRIC  TCGA GSE25066 GSE32646 GSE58644 GSE48091
# FALSE 17744  8481    11231    99     8215    22155     1260    10329
# TRUE  18434 12688    24924 19405    12752    20282    20202    12917

# convert to tibbles, & join probe annotations
micro <- mclapply(esets, function(x) as_tibble(exprs(x), rownames = "probeset"), mc.cores = 8, mc.preschedule = FALSE)
micro <- mcmapply(left_join, fdata, micro, mc.cores = 8, mc.preschedule = FALSE)

# remap symbols with current hgnc data
# https://www.genenames.org/cgi-bin/download/custom?col=gd_hgnc_id&col=gd_app_sym&col=gd_status&col=gd_pub_eg_id&col=gd_pub_ensembl_id&status=Approved&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit
hgncMap <- read_tsv("data/metaGxBreast/hgncMap17jun20.tsv", col_types = "ccccc") %>%
  select(2, 4, 5)

# are there symbols with no entrez? no
table(unlist(mclapply(micro, function(x) sum(is.na(x$entrez)), mc.cores = 8, mc.preschedule = FALSE)))
#  0 
# 39 

# any duplicate symbols that aren't NA? no
table(is.na(hgncMap[duplicated(hgncMap$`NCBI Gene ID`) | duplicated(hgncMap$`NCBI Gene ID`, fromLast = TRUE), 2]))
# TRUE 
# 2003 

microCurrentSymbols <- lapply(micro, function(x) hgncMap$`Approved symbol`[match(x$entrez, hgncMap$`NCBI Gene ID`)])
micro <- mcmapply(bind_cols, hgnc = microCurrentSymbols, micro)

# how many symbols need updating?
library(janitor)
print(bind_cols(bind_rows(lapply(micro,
                       function(x) tabyl(filter(x, best_probe) %>%
                                        transmute(diff = gene != hgnc), diff)), .id = "study") %>%
                                        select(1:3) %>%
                                        pivot_wider(names_from = 2, values_from = 3),
                bind_rows(lapply(micro,
                                 function(x) tabyl(transmute(x, diff = gene != hgnc), diff)), .id = "study") %>%
                  select(1:3) %>%
                  pivot_wider(names_from = 2, values_from = 3) %>%
                  select(2:4)),
      n = 40)
#              metaGX study filtered probes   all original probes
#    study        [same] [updated] [missing] [same] [updated] [missing]
#  1 CAL            11361     1115    212     18474     1599   1096
#  2 DFHCC          15871     2961   1450     34400     5240   2807
#  3 DFHCC2         15876     2956   1450     34400     5240   2807
#  4 DFHCC3         15874     2958   1450     34400     5240   2807
#  5 DUKE            8165      543    128     10945      668    472
#  6 DUKE2          14998     3375   1327     34715     6755   4020
#  7 EMC2           15861     2971   1450     34400     5240   2807
#  8 EORTC10994     11459     1077    216     18551     1521    895
#  9 EXPO           15877     2955   1450     34400     5240   2807
# 10 FNCLCC          4882      176     49      5819      187     58
# 11 HLP            17764     1135    552     24321     1466    749
# 12 IRB            15879     2953   1450     34400     5240   2807
# 13 KOO              238       16     NA       260       16      4
# 14 LUND            9923      383     82     10656      407     91
# 15 LUND2           5340     2464    109     13539     5542   2927
# 16 MAINZ          11458     1078    216     18551     1521    895
# 17 MAQC2          11457     1079    216     18551     1521    895
# 18 MCCC           13687      800    466     17536      952    560
# 19 MDA4           11361     1115    212     18474     1599   1096
# 20 MSK            11458     1078    216     18551     1521    895
# 21 MUG            10247      380     88     13623      540    125
# 22 NCCS           11458     1078    216     18551     1521    895
# 23 NCI             3607      461     44      4565      527     62
# 24 NKI            10762     2221    133     12357     2459    144
# 25 PNC            15876     2956   1450     34400     5240   2807
# 26 STK            15122     2324    988     30195     3900   2083
# 27 STNO2           3110       87     31      3529       99     35
# 28 TRANSBIG       11457     1079    216     18551     1521    895
# 29 UCSF            3275     2896    104      4274     3608    133
# 30 UNC4            4741      227     57      5123      234     63
# 31 UNT            14980     2305    724     30015     3950   2119
# 32 UPP            15118     2328    988     30195     3900   2083
# 33 VDX            11361     1115    212     18474     1599   1096
# 34 METABRIC       15850     4022   5052     24613     5819   5723
# 35 TCGA           18259      875    271     18345      884    275
# 36 GSE25066       11457     1079    216     18551     1521    895
# 37 GSE32646       15867     2965   1450     34390     5240   2807
# 38 GSE58644          NA    19656    546        NA    20864    598
# 39 GSE48091       11343     1451    123     20543     2503    200

# check for duplicate features
sapply(micro, function(x) sum(duplicated(x$hgnc)))
#      CAL      DFHCC     DFHCC2     DFHCC3       DUKE      DUKE2       EMC2 EORTC10994       EXPO 
#     8692      23614      23614      23614       3376      27116      23614       8430      23614 
#   FNCLCC        HLP        IRB        KOO       LUND      LUND2      MAINZ      MAQC2       MCCC 
#     1005       7636      23614         25        726      11250       8430       8430       4560 
#     MDA4        MSK        MUG       NCCS        NCI        NKI        PNC        STK      STNO2 
#     8692       8430       3660       8430       1085       1972      23614      18731        465 
# TRANSBIG       UCSF       UNC4        UNT        UPP        VDX   METABRIC       TCGA   GSE25066 
#     8430       1839        451      18798      18731       8692      16282        369       8430 
# GSE32646   GSE58644   GSE48091 
#    23604       1805      10451 

sapply(micro, function(x) sum(duplicated(x$hgnc[x$best_probe])))
#      CAL      DFHCC     DFHCC2     DFHCC3       DUKE      DUKE2       EMC2 EORTC10994       EXPO 
#      211       1449       1449       1449        127       1326       1449        215       1449 
#   FNCLCC        HLP        IRB        KOO       LUND      LUND2      MAINZ      MAQC2       MCCC 
#       48        551       1449          0         81        108        215        215        465 
#     MDA4        MSK        MUG       NCCS        NCI        NKI        PNC        STK      STNO2 
#      211        215         87        215         43        132       1449        987         30 
# TRANSBIG       UCSF       UNC4        UNT        UPP        VDX   METABRIC       TCGA   GSE25066 
#      215        103         56        723        987        211       5051        270        215 
# GSE32646   GSE58644   GSE48091 
#     1449        545        122 

rm(esets, fdata, hgncMap)

# make metaGX study comparable data set
# use annotation filter, update symbols, select duplicate with greatest variation
mgxSet <- mclapply(micro, function(x) 
  filter(x, best_probe) %>%
    select(- probeset, - gene, - entrez, - best_probe) %>%
    rowwise() %>%
    mutate(sd = sd(c_across(where(is.numeric)))) %>%
    group_by(hgnc) %>%
    filter(sd == max(sd) & !is.na(hgnc)) %>%
    select(- sd),
  mc.cores = 10, mc.preschedule = FALSE)

# clean up names
names(mgxSet$TRANSBIG) <- str_remove(names(mgxSet$TRANSBIG), "TRANSBIG_")
# names(mgxSet$STNO2) <- str_remove(names(mgxSet$STNO2), "STNO2_")

saveRDS(mgxSet, "data/metaGxBreast/mgxSet.rds")

# make aggregate pheno df
study <- names(esets)
pheno <- lapply(esets, pData)
pvars <- lapply(pheno, names)


# these differ from 26 vars in rest of data sets, put them at the beginning
sapply(pvars[36:39], length)
# GSE25066 GSE32646 GSE58644 GSE48091 
#       31       25       29       24 

pheno <- lapply(pheno, function(x) mutate_at(x, vars(one_of("alt_sample_name")), as.character) %>%
                mutate_at(vars(one_of("batch")), as.character) %>%
                mutate_at(vars(one_of("T")), as.character))
pheno = bind_rows(pheno[c(36, 38, 37, 39, 1:35)], .id = "study") %>%
  select(- alt_sample_name, - duplicates, - uncurated_author_metadata, - starts_with("percent_"), - tissue)

table(pheno$sample_type)
# healthy   tumor 
#    133    9584 

# how many batches
batches <- group_by(pheno, study, batch) %>%
  summarise(batch_count = n())
table(batches$study)
#    CAL      DFHCC     DFHCC2     DFHCC3       DUKE      DUKE2       EMC2 EORTC10994       EXPO 
#      1          1          2          1          1          1          1          1          1 
# FNCLCC   GSE25066   GSE32646   GSE48091   GSE58644        HLP        IRB        KOO       LUND 
#      1          1          1          1          1          1          1          1          4 
#  LUND2      MAINZ      MAQC2       MCCC       MDA4   METABRIC        MSK        MUG       NCCS 
#      1          1          1          1          1          6          1          1          1 
#    NCI        NKI        PNC        STK      STNO2       TCGA   TRANSBIG       UCSF       UNC4 
#      1          2          1          1          1          1          5          1          1 
#    UNT        UPP        VDX 
#      2          2          2 

# label missing batch labels in METABRIC as "6"
pheno <- replace_na(pheno, list(batch = "6"))

# re-lable first NKI batch from "NKI" to "NKI1"
pheno <- mutate(pheno, batch = str_replace(batch, "NKI$", "NKI1"))

# how many replicates are there?
table(group_by(pheno, study, unique_patient_ID) %>%
        summarize(uniqueIDcount = n()) %>% 
        filter(!is.na(unique_patient_ID)) %>% 
        pull(uniqueIDcount))
#    1    2 
# 1261    9

ungroup(pheno) %>%
  filter(duplicated(unique_patient_ID) & !is.na(unique_patient_ID)) %>% 
  select(study, sample_name, unique_patient_ID)
#    study      sample_name unique_patient_ID
# 1 DFHCC2   DFHCC2_REF1rep                T1
# 2 DFHCC2  DFHCC2_REF21rep               T21
# 3 DFHCC2  DFHCC2_REF25rep               T25
# 4 DFHCC2  DFHCC2_REF38rep               T38
# 5 DFHCC2  DFHCC2_REF55rep               T55
# 6 DFHCC2  DFHCC2_REF63rep               T63
# 7 DFHCC2  DFHCC2_REF80rep               T80
# 8 DFHCC2 DFHCC2_REF116rep              T116
# 9   MDA4    MDA4_M323_bis               323

# replicate flag to replace unique_patient_ID
pheno <- mutate(pheno, replicate = duplicated(unique_patient_ID) & !is.na(unique_patient_ID)) %>%
  select(-unique_patient_ID)

table(pheno$replicate)
# FALSE  TRUE 
#  9708     9 

# add study code, batch, platform
pheno <- left_join(pheno, select(metaGX, study = Dataset, Platform))

dim(pheno)
# [1] 9717   28

# what is overlap with curatedBreastData?
cbd <- read_csv("../cancer.old/data/bcClinicalTable.csv", guess_max = 5000) %>%
  select(GEO_platform_ID, study_ID, GEO_GSMID) %>%
  transmute(study_ID = paste0("GSE", as.character(study_ID)), GEO_platform_ID = GEO_platform_ID,
            sample_name = paste0("GSM", as.character(GEO_GSMID)))

# same studies?
intersect(cbd$study_ID, pheno$study)
# [1] "GSE32646"

# what about patients?
length(intersect(cbd$sample_name, pheno$sample_name))
# [1] 540

# GSE25066 is a superset of GSE25055 and GSE25065
unique(pheno$study[pheno$sample_name %in% cbd$sample_name])
# [1] "GSE25066" "GSE32646"

overlap <- unique(cbd$study_ID[cbd$sample_name %in% pheno$sample_name])
# [1] "GSE32646" "GSE25055" "GSE25065"

# which coincide data sets do these correspond to?
grep(paste0(str_remove(overlap, "GSE"), collapse = "|"), list.files("../cancer.old/data/bcDump/example15bmc/"), value = TRUE)
# [1] "study_25055_GPL96_MDACC_M-bmc15.csv.xz" "study_25065_GPL96_MDACC-bmc15.csv.xz"  
# [3] "study_25065_GPL96_USO-bmc15.csv.xz"     "study_32646_GPL570_all-bmc15.csv.xz"   

# coresponding to metaGx data sets:
metaGX$Dataset[metaGX$Dataset_accession %in% c("GSE32646", "GSE25055", "GSE25065")]

# note: table updated with pam50 labels in `label clusters.R`
# note: table updated with channel_count labels in 'get geo channel counts.R'
write_csv(pheno, "data/metaGXcovarTable.csv.xz")

# make metaGX comparable expression matrices without batch separation for ml analysis
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

dir.create("data/mgxSet/noNorm", recursive = TRUE)
for(n in names(mgxSet)) {
  expMat <- transpose_df(column_to_rownames(mgxSet[[n]], "hgnc"))
  names(expMat)[1] <- "sample_name"
  mutate(expMat, across(where(is.numeric), formatC, digits = 6, format = "f")) %>%
    write_csv(paste0("data/mgxSet/noNorm/", n, "_noNorm.csv.xz"))
}
rm(expMat)

# function to do robust linear normalization
# use genefu::rescale
rescale02 <- function(x) (genefu::rescale(x, q=0.05) - 0.5) * 2

dir.create("data/mgxSet/rlNorm", recursive = TRUE)
for(n in names(mgxSet)) {
  expMat <- as_tibble(map_if(mgxSet[[n]], is.numeric, rescale02))
  expMat <- transpose_df(column_to_rownames(expMat, "hgnc"))
 names(expMat)[1] <- "sample_name"
  mutate(expMat, across(where(is.numeric), formatC, digits = 6, format = "f")) %>%
    write_csv(paste0("data/mgxSet/rlNorm/", n, "_rlNorm.csv.xz"))
}
rm(expMat)

# eb normalization
# which data set as reference? tcga is largest single batch
# also has other associated omics data available
# https://portal.gdc.cancer.gov/projects/TCGA-BRCA
library(sva)
bpparam <- MulticoreParam(6)
batches <- group_by(pheno, study, batch) %>%
  summarise(batch_count = n())

# # fix batch names
# batches <- mutate(batches, batch = str_replace(batch, "(^[1-6])", "METABRIC\\1")) %>%
#   mutate(batch = str_replace(batch, "NKI$", "NKI1")) %>%
#   mutate(batch = str_replace(batch, "VDX$", "VDX1"))

# make eb normalized batches with tcga as reference batch 
# and 
mgxBset <- parallel::mcmapply(
  function(x, n)
    ComBat(dat = as.matrix(column_to_rownames(x, "hgnc")),
           batch = filter(pheno, study == n) %>% pull(batch),
           mod = model.matrix(as.factor(Platform), data = pheno),
           ref.batch = "TCGA",
           BPPAR = bpparam),
  mgxSet,
  names(mgxSet),
  mc.cores = 2, mc.preschedule = FALSE)

# function to compare before & after batch normalization
# summary difference between transcripts averaged over samples
geneDiff <- function(x, y) summary(rowSums(x) / dim(x)[2] - rowSums(y) / dim(y)[2])

mapply(geneDiff, lapply(mgxSet[c(3, 14, 34, 24, 28, 31, 32, 33)], function(x) as.matrix(column_to_rownames(x, "hgnc"))), mgxBset)
#                DFHCC2          LUND      METABRIC           NKI      TRANSBIG           UNT
# Min.    -0.1940524371 -4.897728e-02 -1.073853e-02 -1.396199e-02 -0.0405788473 -0.0118029302
# 1st Qu. -0.0047201991 -5.124977e-03 -6.371805e-04 -5.513199e-04 -0.0040964396 -0.0001180911
# Median  -0.0003108811  5.109938e-04 -7.200512e-05  2.044640e-05  0.0004875249  0.0000335710
# Mean    -0.0004580138  7.309895e-05 -1.604081e-04 -3.490541e-05  0.0007940716  0.0002501645
# 3rd Qu.  0.0050546602  5.964215e-03  2.995797e-04  5.975223e-04  0.0053734564  0.0003599726
# Max.     0.1187590015  4.226921e-02  8.071947e-03  1.096662e-02  0.0485097757  0.0308484156
#                   UPP           VDX
# Min.    -2.975160e-02 -0.1219866836
# 1st Qu. -2.167293e-03 -0.0079139182
# Median  -7.649264e-07  0.0011213660
# Mean    -2.377696e-04  0.0007174217
# 3rd Qu.  1.942649e-03  0.0092540656
# Max.     2.717275e-02  0.2256328476

sapply(mgxBset, dim)
#      DFHCC2 LUND METABRIC  NKI TRANSBIG   UNT   UPP   VDX
# [1,]  18832 3732    19866 8521    12536 17285 17446 12476
# [2,]     83  143     2114  337      198   133   251   344

# add subfolder with eb batched normalized replacements

# make coincide style merged data set
mgxSetMerge <- Coincide::mergeDatasetList(lapply(mgxSet, function(x) as.matrix(column_to_rownames(x, "hgnc"))), batchNormalize = "none")

mgxSetMergeCombat <- Coincide::mergeDatasetList(lapply(mgxSet, function(x) as.matrix(column_to_rownames(x, "hgnc"))), batchNormalize = "combat")

# this lost 1800 transcripts!
sapply(list(none = mgxSetMerge$mergedExprMatrix, combat = mgxSetMergeCombat$mergedExprMatrix), dim)
#      none combat
# [1,] 5808   3004
# [2,] 8095   8095

# TODO: other versions of merged set

# make metaGX comparable expression matrices for ml analysis separated by batch
# make function to split dfs with names
named_group_split <- function(.tbl, ...) {
  grouped <- group_by(.tbl, ...)
  names <- group_keys(grouped) %>% pull(1)
  grouped %>% 
    group_split() %>% 
    rlang::set_names(names)
}

# make list of batched data sets
batSam <- map(batchSets, function(x) filter(pheno, study == x) %>% select(sample_name, batch)) %>%
  map(function(x) named_group_split(x, batch, .keep = FALSE)) %>%
  map(function(x) map(x, function(y) pull(y, sample_name)))

# function to split df by column list and keep reference column
dfSplit <- function(df, batchList, ref) {
  df <- column_to_rownames(df, ref)
  map(batchList, function(x) select(df, x)) %>%
    map(rownames_to_column, ref)
}

mgxSetBatched <- mgxSet[batchSets]
mgxSetBatched <-   mapply(function(x, y) dfSplit(x, y, "hgnc"), mgxSetBatched[], batSam) %>%
  flatten() %>%
  append(mgxSet[!(names(mgxSet) %in% batchSets)])
saveRDS(mgxSetBatched, "data/metaGxBreast/mgxSetBatched.rds")

dir.create("data/metaGxBreast/noNormBatched", recursive = TRUE)
for(n in names(mgxSetBatched)) {
  expMat <- transpose_df(column_to_rownames(mgxSetBatched[[n]], "hgnc"))
  names(expMat)[1] <- "sample_name"
  mutate(expMat, across(where(is.numeric), formatC, digits = 6, format = "f")) %>%
    write_csv(paste0("data/metaGxBreast/noNormBatched/", n, "_noNorm.csv.xz"))
}
rm(expMat)

dir.create("data/metaGxBreast/rlNormBatched", recursive = TRUE)
for(n in names(mgxSetBatched[1:25])) {
  expMat <- as_tibble(map_if(mgxSetBatched[[n]], is.numeric, rescale02))
  expMat <- transpose_df(column_to_rownames(expMat, "hgnc"))
  names(expMat)[1] <- "sample_name"
  mutate(expMat, across(where(is.numeric), formatC, digits = 6, format = "f")) %>%
    write_csv(paste0("data/metaGxBreast/rlNormBatched/", n, "_rlNorm.csv.xz"))
}
rm(expMat)

# median normalization for moses

# combat normalization

## plots
# batch csv - tibble is saved as csv then changed for markdown rendering.
# some study gse & pmid links are pairs that must be hand edited after pasting into markdown doc
studies <- left_join(batches, metaGX[, -4:-5], by = c("study" = "Dataset"))
studies[33, 2] <- "6"
write_csv(studies, "data/metaGxStudies.csv") %>%
 mutate(batch = if_else(duplicated(study) | duplicated(study, fromLast = TRUE), batch, "")) %>%
  replace_na(list(Dataset_accession = "", Notes = "")) %>%
  mutate(study = if_else(duplicated(study), "", study),
         PMID = if_else(duplicated(PMID), "", PMID),
         Dataset_accession = if_else(duplicated(Dataset_accession), "", Dataset_accession),
         Notes = if_else(duplicated(Notes), "", Notes)) %>%
  mutate(Dataset_accession = if_else(grepl("^GSE", Dataset_accession),
                                     paste0("[", Dataset_accession,
                                           "](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=",
                                           Dataset_accession, ")"),
                                     Dataset_accession)) %>%
  mutate(PMID = if_else(grepl("^[0-9]", PMID),
                             paste0("[", PMID, "](https://www.ncbi.nlm.nih.gov/pubmed/", PMID, ")"),
                             PMID)) %>%
  knitr::kable("pipe")


# make sample count table from vignette code
numSamples <- sapply(esets, function(x) length(sampleNames(x)))
SampleNumberSummaryAll <- data.frame(NumberOfSamples = numSamples,
                                     row.names = names(esets))
total <- sum(SampleNumberSummaryAll[,"NumberOfSamples"])
SampleNumberSummaryAll <- rbind(SampleNumberSummaryAll, total)
rownames(SampleNumberSummaryAll)[nrow(SampleNumberSummaryAll)] <- "Total"

# make latex table
write_lines(xtable::xtable(SampleNumberSummaryAll,digits = 2), "data/metaGxStudies.tex")

# pData Variables
# the "T" phenotype variable doesn't work here, is it mixed up with "T"rue ?
pDataVars <- lapply(esets, function(x) names(pData(x)))
setdiff(names(pDataVars), names(unlist(lapply(pDataVars, function(x) grep("^T$", x, value = TRUE)))))
# [1] "GSE48091"
grep("GSE48091", names(esets))
# [1] 39

pDataID <- c("er","pgr", "her2", "age_at_initial_pathologic_diagnosis", "grade", "dmfs_days", "dmfs_status", "days_to_tumor_recurrence", "recurrence_status", "days_to_death", "vital_status", "sample_type", "treatment", "batch", "duplicates", "tumor_size", "T", "N")



pDataPercentSummaryTable <- NULL
pDataSummaryNumbersTable <- NULL

pDataSummaryNumbersList = lapply(esets[-39], function(x)
  sapply(pDataID, function(y) sum(!is.na(pData(x)[,y]))))

pDataPercentSummaryList = lapply(esets[-39], function(x)
  vapply(pDataID, function(y)
    sum(!is.na(pData(x)[,y]))/nrow(pData(x)), numeric(1))*100)

pDataSummaryNumbersTable = sapply(pDataSummaryNumbersList, function(x) x)
pDataPercentSummaryTable = sapply(pDataPercentSummaryList, function(x) x)

rownames(pDataSummaryNumbersTable) <- pDataID
rownames(pDataPercentSummaryTable) <- pDataID
colnames(pDataSummaryNumbersTable) <- names(esets)
colnames(pDataPercentSummaryTable) <- names(esets)

pDataSummaryNumbersTable <- rbind(pDataSummaryNumbersTable, total)
rownames(pDataSummaryNumbersTable)[nrow(pDataSummaryNumbersTable)] <- "Total"

# Generate a heatmap representation of the pData
pDataPercentSummaryTable<-t(pDataPercentSummaryTable)
pDataPercentSummaryTable<-cbind(Name=(rownames(pDataPercentSummaryTable))
                                ,pDataPercentSummaryTable)

nba<-pDataPercentSummaryTable
gradient_colors = c("#ffffff","#ffffd9","#edf8b1","#c7e9b4","#7fcdbb",
                    "#41b6c4","#1d91c0","#225ea8","#253494","#081d58")

library(lattice)
nbamat<-as.matrix(nba)
rownames(nbamat)<-nbamat[,1]
nbamat<-nbamat[,-1]
Interval<-as.numeric(c(10,20,30,40,50,60,70,80,90,100))

# rstudio export with width = 850
levelplot(nbamat,col.regions=gradient_colors,
          main="Available Clinical Annotation",
          scales=list(x=list(rot=90, cex=0.8),
                      y= list(cex=0.8),key=list(cex=0.2)),
          at=seq(from=0,to=100,length=10),
          cex=0.2, ylab="", xlab="", lattice.options=list(),
          colorkey=list(at=as.numeric(factor(c(seq(from=0, to=100, by=10)))),
                        labels=as.character(c( "0","10%","20%","30%", "40%","50%",
                                               "60%", "70%", "80%","90%", "100%"),
                                            cex=0.2,font=1,col="brown",height=1,
                                            width=1.4), col=(gradient_colors)))

# make variable description table
# descriptions added by hand in spreadsheet
varTab <- read_csv("data/metaGxBreast/coVarsDescription.csv")
varCount <- summarize(pheno)
varTab <- mutate(varTab, present = !is.na(pheno[[variable]]))
present <- enframe(apply(!is.na(pheno), 2, sum), name = "variable", value = "n")
varTab <- left_join(varTab, present)
write_csv(varTab, "data/metaGxBreast/coVarsDescription.csv")

# convert to markdown for pasting in to readme
knitr::kable(varTab, "pipe")
