## add covariate table column for channel count
library(tidyverse)

# use geometadb to find channel count for geo studies
# GEOmetadb::getSQLiteFile("/mnt/biodata/GEO")
con <- DBI::dbConnect(RSQLite::SQLite(), "/mnt/biodata/GEO/GEOmetadb.sqlite")
gsm <- tbl(con, "gsm") %>%
  select(series_id, gpl, channel_count)

# get bc data info
ct <- read_csv("data/curatedBreastData/bcClinicalTable.csv", guess_max = 10000)
bcFiles <- list.files("~/R/cancer.old/data/bcDump/example15bmc/", pattern = "^study", full.names = TRUE)
bc15 <- map(bcFiles, read_csv)
names(bc15) <- paste0("GSE",
                str_remove(bcFiles, "^/home/mozi/R/cancer.old/data/bcDump/example15bmc//study_") %>%
                str_remove("_all-bmc15.csv.xz|-bmc15.csv.xz"))

# geoquery check confirms only one of 15 studies in merged dataset bmc15 is two channel (GSE22226)
# compared to 
round(as_tibble(map(bc15[1:6], ~ summary(colMeans(select(., -1))))), digits = 3)
# GSE12093_GPL96 GSE1379_GPL1223 GSE16391_GPL570 GSE16446_GPL570 GSE17705_GPL96_JBI_1 GSE17705_GPL96_2
# 1  2.240          0.742           2.242           2.254           2.274                  2.233        
# 2  2.450          8.991           2.478           2.661           2.410                  2.360        
# 3  4.725          9.676           5.434           5.441           4.382                  4.546        
# 4  4.970          9.610           5.498           5.503           4.761                  4.931        
# 5  6.827         10.217           7.642           7.519           6.477                  6.921        
# 6 14.840         16.727          15.793          15.397          14.195                 14.930        
round(as_tibble(map(bc15[7:12], ~ summary(colMeans(select(., -1))))), digits = 3)
# GSE19615_GPL570 GSE20181_GPL96 GSE20194_GPL96 GSE2034_GPL96 GSE22226_GPL1708 GSE22358_GPL5325
# 1  2.251           2.209          2.279          2.212        -6.110            3.173          
# 2  2.556           2.262          2.431          2.394        -0.465            9.496          
# 3  5.329           3.679          4.021          5.061         0.032            9.967          
# 4  5.340           4.999          4.628          5.194        -0.018            9.942          
# 5  7.375           7.552          6.331          7.287         0.397           10.358          
# 6 15.197          15.460         14.651         14.985         5.958           16.583          
round(as_tibble(map(bc15[13:17], ~ summary(colMeans(select(., -1))))), digits = 3)
# GSE25055_GPL96_MDACC_M GSE25065_GPL96_MDACC GSE25065_GPL96_USO GSE32646_GPL570 GSE9893_GPL5049
# 1  2.270                  2.327                2.287              4.921           1.211         
# 2  2.424                  2.445                2.420              9.458           2.486         
# 3  4.035                  3.445                3.877             11.290           3.393         
# 4  4.662                  4.214                4.506             11.520           3.629         
# 5  6.405                  5.543                6.129             13.386           4.509         
# 6 14.776                 15.044               14.397             20.137          10.395         

bc <- select(ct, study_ID, GEO_platform_ID) %>%
  distinct() %>%
  mutate(study_ID = paste0("GSE", study_ID))
bc$study_ID[16:17] <- paste0(bc$study_ID[16:17], ",GSE25066")
bc$study_ID[24] <- "GSE16716,GSE20194"

bcChan <- left_join(bc, gsm, by = c("study_ID" = "series_id", "GEO_platform_ID" = "gpl"), copy = T) %>%
  distinct()

DBI::dbDisconnect(con)
table(bcChan$channel_count, useNA = "always")
#    1    2 <NA> 
#   16   11    0 

# write_csv(bcChan, "data/curatedBreastData/bcStudyChannelCount.csv")
# set up bcChan for ct join
bcChan$GEO_platform_ID[3:4] <- "GPL1708,GPL4133"
bcChan$GEO_platform_ID[13:15] <- "GPL1390,GPL5325,GPL7504"
bcChan <- distinct(bcChan) %>%
  transmute(series_id = study_ID, gpl = GEO_platform_ID, channel_count = channel_count, study_ID = str_remove(series_id, "GSE"))
bcChan$study_ID[13:14] <- c("25055", "25065")
bcChan$study_ID[21] <- "20194"
bcChan <- mutate(bcChan, study_ID = as.numeric(study_ID))

# update cbd covariate table with channel count and geometadb study labels
bc <- left_join(bcChan, ct)

# replace covariate table
write_csv(bc, "data/curatedBreastData/bcClinicalTable.csv")

# do the same for metagx studies
# get data
mgxSet <- readRDS("data/metaGxBreast/mgxSet.rds")
mgxSet <- map(mgxSet, ungroup)
pheno <- read_csv("data/metaGxBreast/metaGXcovarTable.csv.xz", guess_max = 4000)
mgx <- read_csv("data/metaGxBreast/MetaGxMetaData.csv") %>%
  select(2, 4, 6) %>%
  filter(!is.na(Dataset))

# requires a bunch of tweaking to maximize matches
mgx <- mgx[c(1:33, 33:39),]
mgx$Dataset_accession[33:34] <- c("GSE2034", "GSE5327")
mgx$Dataset_accession[37] <- "GSE25055,GSE25066"
mgx$Dataset_accession[17] <- "GSE16716,GSE20194"
mgx$Dataset_accession[25] <- "GSE20711,GSE20713"
mgx$Dataset_accession[27] <- "GSE61,GSE3193,GSE3281,GSE4268,GSE4335,GSE4382"
mgx$Platform[31] <- "GPL96"
mgx$Dataset_accession[31] <- "GSE2990,GSE6532"
mgx <- mgx[c(1:26, 26:27, 27, 27:30, 30, 30:40),]
mgx$Platform[26:30] <- c("GPL96", "GPL97", "GPL180", "GPL2776", "GPL2777")
mgx$Platform[33:35] <- c("GPL885", "GPL887", "GPL1390")

mgxChan <- left_join(mgx, gsm, by = c("Dataset_accession" = "series_id"), copy = TRUE) %>%
  distinct()

table(mgxChan$channel_count, useNA = "always")
#    1    2 <NA> 
#   47   18    7 

# check non geo data sets
filter(mgxChan, is.na(channel_count))
#   Dataset Dataset_accession    Platform         gpl   channel_count
# 1 CAL     E-TABM-158           Affymetrix HGU   NA               NA
# 2 HLP     E-TABM-543           Illumina         NA               NA
# 3 KOO     Authors' website     Affymetrix HGU95 NA               NA
# 4 MDA4    MDACC DB             Affymetrix HGU   NA               NA
# 5 NCI     Authors' website     In-house cDNA    NA               NA
# 6 NKI     Rosetta Inpharmatics Agilent          NA               NA
# 7 UCSF    Authors' website     In-house cDNA    NA               NA

round(as_tibble(map(mgxSet[c(1, 11, 13, 19, 23, 24, 29)], ~ summary(rowMeans(select(., -1))))), dig = 3)
#    CAL     HLP     KOO     MDA4    NCI     NKI     UCSF   
# 1  4.031   4.794   3.576   2.928  -3.499  -0.986  -3.335 
# 2  5.158   5.342   6.223   4.886  -0.219  -0.025  -0.438 
# 3  6.693   6.432   7.386   5.922   0.014  -0.001  -0.059 
# 4  6.886   6.915   7.254   5.980  -0.010  -0.014  -0.101 
# 5  8.229   8.130   8.465   6.914   0.209   0.016   0.279 
# 6 14.216  13.649  11.085  13.688   3.224   0.080   3.114 

# expression level distribution summaries for un-normalized vital_status and recurrence_status subsets
# requires lists from "qsnorm.R" to run
# round(as_tibble(map(qsSet_recur, ~ summary(colMeans(select(., -1))))), digits = 3)
# #    CAL     NCI     NKI     PNC     STK     STNO2   TRANSBIG UCSF    UNC4    UNT     UPP    
# # 1  4.032  -3.451  -0.966   2.148   1.865  -5.379   1.544   -3.144  -0.898  -0.696   1.423 
# # 2  5.164  -0.218  -0.024   4.023   5.302  -0.476   5.562   -0.438  -0.004  -0.003   5.048 
# # 3  6.691   0.013  -0.001   5.837   6.378  -0.098   7.416   -0.059   0.024   0.014   6.233 
# # 4  6.887  -0.010  -0.014   5.826   6.308  -0.069   7.431   -0.101   0.036   0.028   6.174 
# # 5  8.229   0.213   0.016   7.351   7.285   0.311   9.130    0.276   0.057   0.041   7.278 
# # 6 14.214   3.226   0.084  14.114  11.677   4.573  15.889    3.013   0.683   1.244  11.925 
# 
# select(mgxChan, Dataset, channel_count) %>% 
#   filter(Dataset %in% names(qsSet_recur)) %>% distinct() %>%
#   deframe()
# # CAL      NCI      NKI      PNC      STK    STNO2 TRANSBIG     UCSF     UNC4      UNT      UPP 
# #  NA       NA       NA        1        1        2        1       NA        2        1        1 
# 
# round(as_tibble(map(qsSet_death, ~ summary(colMeans(select(., -1))))), digits = 3)
# #    CAL     DUKE    METABRIC NKI     PNC     STNO2   TCGA    TRANSBIG UCSF    UNC4   
# # 1  4.032   3.305   5.083   -0.973   2.152  -5.379   0.029   1.544   -3.129  -0.904 
# # 2  5.165   5.303   5.492   -0.024   4.024  -0.476   3.138   5.562   -0.433  -0.004 
# # 3  6.690   6.578   6.175   -0.001   5.836  -0.098   7.782   7.416   -0.065   0.023 
# # 4  6.886   6.596   6.758   -0.013   5.826  -0.069   6.606   7.431   -0.101   0.036 
# # 5  8.227   7.747   7.648    0.017   7.351   0.311   9.689   9.130    0.277   0.058 
# # 6 14.207  12.799  14.240    0.082  14.110   4.573  16.501  15.889    3.003   0.679 
# 
# select(mgxChan, Dataset, channel_count) %>% 
#   filter(Dataset %in% names(qsSet_death)) %>% distinct() %>%
#   deframe()
# # CAL     DUKE      NKI      PNC    STNO2 TRANSBIG     UCSF     UNC4 METABRIC     TCGA 
# #  NA        1       NA        1        2        1       NA        2        1        1 

# first four are most likely 1 channel, remaining three are 2 channel
mgxChan$channel_count[c(1, 11, 13, 20)] <- 1
mgxChan$channel_count[c(24, 25, 35)] <- 2
mgxChan <- select(mgxChan, - Platform) %>%
  distinct()
# write_csv(mgxChan, "data/metaGxBreast/mgxStudyChannelCount.csv")
# add geometadb info to covariate table
mgxChan <- group_by(mgxChan, Dataset) %>%
  transmute(study = Dataset, gpl = paste0(gpl, collapse = ","), channel_count = channel_count) %>%
  distinct() %>%
  ungroup() %>%
  select(- Dataset)
mgx <- left_join(pheno, mgxChan)

write_csv(mgx, "data/metaGxBreast/metaGXcovarTable.csv.xz")

## code to integrate inokenty's platform data originally in "check data mgxb.R"
# add inokenty's platform data
uniquePlatfroms <- lapply(list(MAQC2 = "MAQC2", STK = "STK", STNO2 = "STNO2", UNC4 = "UNC4"),
                          function(x) read_tsv(paste0("data/metaGxBreast/Studies/", x, ".txt"), 
                                               col_names = c("gsm", "sample_name", "Platform")))

lapply(uniquePlatfroms, function(x) table(x$Platform))
# $MAQC2
# GPL96 
# 278 
# 
# $STK
# GPL96 GPL97 
#   159   159 
# 
# $STNO2
# GPL180 GPL2776 GPL2777 GPL2778 GPL3045 GPL3047 GPL3147 GPL3507 
#     66      43       1       8       3      24      21       1 
# 
# $UNC4
## GPL1390 GPL1708 GPL5325 GPL6607 GPL7504  GPL885  GPL887 
#      200      11      19       2      28      20      92 

sapply(uniquePlatfroms, dim)
#      MAQC2 STK STNO2 UNC4
# [1,]   278 318   167  372
# [2,]     3   3     3    3
sapply(mgxSet[names(uniquePlatfroms)], dim)
#      MAQC2   STK STNO2 UNC4
# [1,] 12536 17446  3197 4966
# [2,]    45   160   119  306

# none of names match!
intersect(pheno$sample_name, purrr::reduce(map(uniquePlatfroms, pull, sample_name), c))
# character(0)

intersect(pheno$sample_name, reduce(map(uniquePlatfroms, pull, gsm), c))
# character(0)

# how to fix?
# MAQC2 alternate_sample_name is 3 digit number, match with digits in inokenty's sample names
# TODO: follow up probable replicate samples, ie BR_FNA_M264 & BR_FNA_M264R1
# TODO: confirm which one is retained by MetaGxBreast::loadBreastEsets
uniquePlatfroms$MAQC2 <- mutate(uniquePlatfroms$MAQC2,
                                sample_name = str_replace(sample_name, "BR_FNA_M", "MAQC2_"))

# STK alternate_sample_name is 1-3 digit number, match with digits in inokenty's sample names
# 94 are still missing, can we match GSMs ?
# no, it appears gpl96 & 97 were merged into one eset for each patient
uniquePlatfroms$STK <- mutate(uniquePlatfroms$STK,
                              sample_name = paste0("STK_", str_remove(substr(sample_name, 2, 4), "^00|^0")))

# STNO2 sample_name has prefix "STNO2_", add it to inokenty's sample names
# TODO: 34 are still missing
uniquePlatfroms$STNO2 <- mutate(uniquePlatfroms$STNO2, sample_name = paste0("STNO2_", sample_name))

# UNC4 gsm ids & sample names have no apparent relationship to inokenty's gsm ids or sample names

setdiff(filter(pheno, study %in% names(uniquePlatfroms)[-4]) %>% pull(sample_name),
        purrr::reduce(map(uniquePlatfroms, pull, sample_name), c))

