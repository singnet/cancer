### EGSEA

options(connectionObserver = NULL)

# load MASS first so it doesn't hide dplyr::select
library(MASS)
library(tidyverse)

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

msigdb <- bind_rows(msig74)
diffExp <- map(list(cb15 = "data/curatedBreastData/diffExp/top120genes15studies.csv",
                    cb15f = "data/curatedBreastData/diffExp/top100genes15studiesFiltered.csv",
                    tam4 = "data/curatedBreastData/diffExp/top2000genes4studies.csv",
                    tam4f = "data/curatedBreastData/diffExp/top1100genes4studiesFiltered.csv"), read_csv)

bestGeneSets <- map(diffExp, ~ pull(., logFC, symbol))


# TODO: fix
libarary(pathwayPCA)
msigdbCB <- pathwayPCA::read_gmt("/mnt/biodata/msigdb/msigdb.v7.4.CBsymbol.gmt")
pathwayPCA::write_gmt(msigdbCB, "/mnt/biodata/msigdb/msigdb74_CBsymbol.gmt")
map(msigdbCB$pathways, length) %>%
  unlist() %>%
  summary()
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 1.0    13.0    45.0   106.2   169.0  1810.0 

msigdbCBtab <- as_tibble(msigdbCB) %>%
  mutate(geneCount = map_int(pathways, ~ length(unlist(.))))
table(map_lgl(msigdbCBtab$pathways, ~ has_element(., "")))
# FALSE 
# 32280 

table(map_lgl(append(msigdbCBtab$pathways, list(c("GENE1", "", "GENE2"))), ~ has_element(., "")))
# FALSE  TRUE 
# 32280     1 

cd15idx <- buildGMTIdx(bestGeneSets$cb15, "/mnt/biodata/msigdb/msigdb74_CBsymbol.gmt", label = "msigDB_7.4", name = "current msigDB gene sets", min.size = 2)
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

# TODO: convert protein ids to gene symbols
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