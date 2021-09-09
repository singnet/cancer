## asmoses testing
library(tidyverse)
source("~/R/Rmoses/R/moses2.R")

tamExp <- read_csv("ml/asmoses/tamox4expSet.csv.xz")
tamMedNorm <- select(tamExp, -patient_ID:-DFS) %>%
  mutate(across(-1, ~ as.numeric(. > median(.))))

write_csv(tamMedNorm, "ml/asmoses/tamoxBinary.csv.xz")
