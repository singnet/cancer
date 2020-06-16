## check out bc data
# get everything
library(MetaGxBreast)
esets <- loadBreastEsets("majority")
majority <- names(esets$esets)
esets <- loadBreastEsets(c(majority, "METABRIC", "TCGA"))
# Ids with missing data: CAL, EORTC10994, FNCLCC, LUND, LUND2, NCI, NKI, UCSF, METABRIC
save(esets, file = "data/metaGx/metaGxBCesets.rdata")

# make sample count table from vignette code
numSamples <- vapply(seq_along(esets), FUN=function(i, esets){
  length(sampleNames(esets[[i]]))
}, numeric(1), esets=esets)
SampleNumberSummaryAll <- data.frame(NumberOfSamples = numSamples,
                                     row.names = names(esets))
total <- sum(SampleNumberSummaryAll[,"NumberOfSamples"])
SampleNumberSummaryAll <- rbind(SampleNumberSummaryAll, total)
rownames(SampleNumberSummaryAll)[nrow(SampleNumberSummaryAll)] <- "Total"
print(xtable::xtable(SampleNumberSummaryAll,digits = 2), floating = FALSE)

# get datasets matching metabric
mb <- c(1, 6, 28, 30:33, 38)
duplicates <- esets$duplicates[mb]
esets <- esets$esets[mb]

# turn into csvs and save
library(tidyverse)
mbMat <- lapply(esets, exprs)
names(cbcMat) <- names(cbc)
sapply(mbMat, dim)

metabric <- read_csv("data/metaGx/METABRIC.csv.xz")
