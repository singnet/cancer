### check data for abdu's experiment
library(tidyverse)

# expression data
tm <- read_csv("ml/asmoses/tamox4expSet.csv.xz")

# https://github.com/singnet/cancer/blob/master/ml/asmoses/tamoxBinary.csv.xz
tmb <- read_csv("ml/asmoses/tamoxBinary.csv.xz")

dim(tmb)
# [1]  642 8833

table(tmb$posOutcome)
#   0   1 
# 171 471 

# gene lists for background network
# get current gene symbols (code from "data check bc.R")
symbolMap <- read_tsv("https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&col=gd_prev_sym&col=gd_aliases&col=gd_pub_eg_id&col=gd_pub_ensembl_id&status=Approved&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit") %>%
  mutate(`Previous symbols` = str_split(`Previous symbols`, ", "),
         `Alias symbols` = str_split(`Alias symbols`, ", "))

# combine previous and alias list columns, unnest, and add current/alt symbol identity row
symbolMap <- mutate(symbolMap, alt = map2(`Previous symbols`, `Alias symbols`, c), .keep = "unused") %>%
  mutate(alt = map(alt, ~ if(identical(., c(NA, NA))) NA_character_ else na.omit(.))) %>%
  unnest(cols = alt)
symbolMap <- bind_rows(symbolMap, mutate(symbolMap, alt = `Approved symbol`) %>% distinct()) %>%
  arrange(`Approved symbol`)

write_csv(symbolMap, "data/moses/hgncSymbolMap6jul22long.csv")


# https://pubchem.ncbi.nlm.nih.gov/pathway/PathBank:SMP0000606
tamPath <- read_csv("data/PathwayID_1185075_tamoxifen-metabolism.csv")

length(pull(tamPath, genesymbol))
# [1] 10

length(intersect(tamPath$genesymbol, names(tmb)))
# [1] 7

tamPathMissing <- setdiff(tamPath$genesymbol, names(tmb))
tamPathOutdated
# [1] "CYP3A5"  "UGT1A10" "UGT1A4"

# update symbols?  they are just missing
filter(symbolMap, `Approved symbol` %in% tamPathMissing) %>%
  filter(alt %in% names(tmb))
# A tibble: 0 × 4
# … with 4 variables: Approved symbol <chr>, NCBI Gene ID <dbl>, Ensembl gene ID <chr>,

# estrogen signaling
# https://reactome.org/PathwayBrowser/#/R-HSA-8939211&DTAB=MT
  esSig <- read_tsv("data/moses/Participating Molecules [R-HSA-8939211].tsv")
esSig <-  select(esSig, MoleculeName) %>%
  filter(str_detect(MoleculeName, "UniProt")) %>%
  separate(MoleculeName, c("uniprot", "symbol"), " ") %>%
  pull(symbol)

length(esSig)
# [1] 196

length(intersect(esSig, names(tmb)))
# [1] 115

# update symbols
esSigMissing <- setdiff(esSig, names(tmb))
length(esSigMissing)
# [1] 75

tmbOutdated <- filter(symbolMap, `Approved symbol` %in% esSigMissing) %>%
  filter(alt %in% names(tmb))

esSigBackdated <- c(intersect(esSig, names(tmb)), tmbOutdated$alt)
setdiff(esSigBackdated, names(tmb))  
# character(0)

length(esSigBackdated)
# [1] 139
length(unique(esSigBackdated))
# [1] 127

esSigBackdated <- unique(esSigBackdated)

intersect(tamPath$genesymbol, esSigBackdated)
# [1] "ESR1"

# ERBB2 signaling
# https://reactome.org/PathwayBrowser/#/R-HSA-1227986&DTAB=MT
erbb2 <- read_tsv("data/moses/Participating Molecules [R-HSA-1227986].tsv") %>%
  select(MoleculeName) %>%
  filter(str_detect(MoleculeName, "UniProt")) %>%
  separate(MoleculeName, c("uniprot", "symbol"), " ") %>%
  pull(symbol)
length(erbb2)
# [1] 56

length(unique(erbb2))
# [1] 50

erbb2 <- unique(erbb2)
erbb2missing <- setdiff(erbb2, names(tmb))
length(erbb2missing)
# [1] 15

tmbOutdated2 <- filter(symbolMap, `Approved symbol` %in% erbb2missing) %>%
  filter(alt %in% names(tmb))

erbb2Backdated <- c(intersect(erbb2, names(tmb)), tmbOutdated2$alt)
setdiff(erbb2Backdated, names(tmb))  
# character(0)

length(erbb2Backdated)
# [1] 38

# save gene lists
write_lines(tamPath$genesymbol, "data/moses/tamoxifen")
write_lines(erbb2Backdated, "data/moses/erbb2")
write_lines(esSigBackdated, "data/moses/estrogen")
