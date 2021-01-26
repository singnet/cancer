### mis-classification analysis
library(tidyverse)
raw <- as.character(read_lines("ml/embedding vectors/raw_genes_misclassified.txt"))
moses <- as.character(read_lines("ml/embedding vectors/moses50_misclassified.txt"))
pam35 <- as.character(read_lines("ml/embedding vectors/pam35_misclassified.txt"))
xgb <- as.character(read_lines("ml/embedding vectors/xg50_misclassified.txt"))
igan <- as.character(read_lines("ml/embedding vectors/infogan_misclassified.txt"))

misses <- list(raw = raw, moses = moses, xgb = xgb, pam35 = pam35, igan = igan)
missIndex <- matrix(nrow = 5, ncol = 5)
for(i in 1:5) {
  for(j in 1:5) {
    missIndex[i, j] <- length(intersect(misses[[i]], misses[[j]])) / length(union(misses[[i]], misses[[j]]))
  }
}
rownames(missIndex) <- names(misses)
colnames(missIndex) <- names(misses)
missIndex
# raw                 moses       xgb     pam35      igan
# raw   1.0000000 0.1544715 0.1900452 0.4162437 0.1199262
# moses 0.1544715 1.0000000 0.6795367 0.1980769 0.3245125
# xgb   0.1900452 0.6795367 1.0000000 0.2129436 0.2948718
# pam35 0.4162437 0.1980769 0.2129436 1.0000000 0.1654930
# igan  0.1199262 0.3245125 0.2948718 0.1654930 1.0000000

# any relationship to study?
library(janitor)
pheno <- read_csv("data/curatedBreastData/bcClinicalTable.csv", guess_max = 4000)
gexp <- read_csv("data/curatedBreastData/microarray data/merged-combat15.csv.xz")
studies <- map(misses, ~filter(pheno, patient_ID %in% .) %>% tabyl(series_id) %>% arrange(desc(n)))
map(studies, ~select(., -n) %>% slice(1:5))
# $raw
#         series_id    percent
#           GSE2034 0.18750000
# GSE16716,GSE20194 0.12500000
#          GSE22358 0.12500000
#          GSE22226 0.10714286
#          GSE17705 0.08928571
# 
# $moses
#         series_id    percent
#           GSE2034 0.20394737
#          GSE17705 0.15570175
# GSE16716,GSE20194 0.12061404
# GSE25055,GSE25066 0.09429825
#          GSE22226 0.06140351
# 
# $xgb
#         series_id    percent
#           GSE2034 0.22222222
#          GSE17705 0.15700483
# GSE16716,GSE20194 0.11111111
# GSE25055,GSE25066 0.10386473
#          GSE16446 0.06038647
# 
# $pam35
#         series_id    percent
#           GSE2034 0.18562874
#          GSE22358 0.13173653
#          GSE17705 0.12574850
# GSE16716,GSE20194 0.10778443
#          GSE22226 0.06586826
# 
# $igan
#         series_id    percent
# GSE16716,GSE20194 0.27070707
#           GSE2034 0.13535354
#          GSE17705 0.09898990
#          GSE22358 0.09292929
# GSE25055,GSE25066 0.07474747

map(studies, ~select(., -n) %>% slice(1:5) %>% pull(series_id)) %>%
  reduce(intersect)
# [1] "GSE2034"           "GSE16716,GSE20194" "GSE17705"

# refine clinical variables for embedding
cvar <- read_lines("data/curatedBreastData/clinical variables.scm") %>%
  str_remove_all("^.+ \"|\".$")
gexp <- read_csv("data/curatedBreastData/microarray data/merged-combat15.csv.xz") %>%
  mutate(patient_ID = as.character(patient_ID))

dim(pheno)
# [1] 2719  142
pheno15 <- filter(pheno, patient_ID %in% gexp$patient_ID)
dim(pheno15)
# [1] 2237  142

# TOD:  make sure extra files aren't propogating errors
outcomes <- read_csv("data/curatedBreastData/combat15outcomes.csv")
pheno2 <- read_csv("data/curatedBreastData/bmc15mldata1.csv")
pheno3 <- read_csv("reports/coincideClinDat.csv", guess_max = 3000)

# where are extra 12 patients from?
extras <- setdiff(pheno15$patient_ID, pheno2$patient_ID)
extras
# [1] 249619 249623 249659 249670 249682 249683 250009 491205 491228 491285 491185 491188

filter(pheno, patient_ID %in% extras) %>%
  tabyl(series_id)
# series_id n   percent
#  GSE19615 5 0.4166667
#   GSE9893 7 0.5833333

# clean up
pheno15 <- select(pheno15, names(outcomes)[c(-8, -9)], cvar[-28])

summary(select(pheno15, RFS, DFS, pCR))
#            RFS              DFS              pCR        
# Min.   :0.0000   Min.   :0.0000   Min.   :0.0000  
# 1st Qu.:0.0000   1st Qu.:1.0000   1st Qu.:0.0000  
# Median :1.0000   Median :1.0000   Median :0.0000  
# Mean   :0.7146   Mean   :0.7805   Mean   :0.2293  
# 3rd Qu.:1.0000   3rd Qu.:1.0000   3rd Qu.:0.0000  
# Max.   :1.0000   Max.   :1.0000   Max.   :1.0000  
# NA's   :1207     NA's   :1581     NA's   :1103    

# get rid of variables with mostly missing values
embed15 <- select(pheno15, 1:7, ER = ER_preTrt, HER2 = HER2_preTrt, PR = PR_preTrt, tumor_stage_preTrt, tumor_size_cm_preTrt_preSurgery, preTrt_lymph_node_status) %>%
  mutate(node = recode(preTrt_lymph_node_status, "N0" = "0", "N1" = "1", "N2" = "1", "N3" = "1",    "positive" = "1", .default = NA_character_),
         tumor = coalesce(tumor_stage_preTrt,
          paste0("T", cut(tumor_size_cm_preTrt_preSurgery, breaks = c(-1, 2, 5, 999), labels = FALSE)))) %>%
  select(1:10, node, tumor) %>%
  left_join(select(pheno3, patient_ID, pam_coincide, p5)) %>%
  mutate(node = as.numeric(node), tumor = na_if(tumor, "TNA"), posOutcome = coalesce(pCR, RFS, DFS), posOutcome2 = coalesce(RFS, DFS)) 

write_csv(embed15, "data/curatedBreastData/embedding_vector_state_and_outcome.csv")
