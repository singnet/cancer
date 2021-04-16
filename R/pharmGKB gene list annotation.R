### pharmGKB gene sets
library(dplyr)

# make pharmagkb set
pgkb <- read_csv("~/mozi/knowledge-import/cancer/pharmgkb_pathway.csv") %>%
  group_by(`Pathway ID`) %>%
  nest() %>%
  ungroup(`Pathway ID`) %>%
  transmute(pgkbID = `Pathway ID`, genes = map(data, pull))

pgkbAnn <- list.files("/mnt/biodata/pharmgkb/pathways-tsv/") %>%
  str_subset("^PA") %>%
  str_split("-", 2) %>%
  transpose() %>%
  map(flatten_chr) %>%
  set_names(c("pgkbID", "name")) %>%
  as_tibble() %>%
  mutate(name = str_remove(name, ".tsv")) %>%
  left_join(pgkb)

#' @michael I have got the following drug pathways (Pharmgkb pathways) from the PLN result of the mrmr100 genes with absolute gene expression.

pgkbSet1 <- c('PA2042',
  'PA166181140',
  'PA150642262',
  'PA152241951',
  'PA165959313',
  'PA150653776')

pgkbSet1 <- filter(pgkbAnn, pgkbID %in% pgkbSet1) %>%
  dplyr::select(- genes)

write_csv(pgkbSet1, "data/curatedBreastData/diffExp/PLNpgkbSets.csv")

# The following are from the relative median norm

pgkbSet2 <- str_split(c('PA152530845_overexp',
  'PA154444041_overexp',
  'PA2042_overexp',
  'PA152530845_underexp',
  'PA166181140_overexp',
  'PA154444041_underexp',
  'PA166181140_underexp',
  'PA165959313_overexp',
  'PA152241951_underexp',
  'PA165959313_underexp',
  'PA152241951_overexp',
  'PA2042_underexp'), "_") %>%
  transpose() %>%
  map(unlist) %>%
  set_names(c("pgkbID", "direction")) %>%
  as_tibble() %>%
  left_join(pgkbAnn) %>%
  dplyr::select(- genes)

write_csv(pgkbSet2, "data/curatedBreastData/diffExp/PLNpgkbSetsRelative.csv")

# The postfix _overexp or _underexp are dueto the normalization, which tells that the member genes are overexpressed or under expressed. (edited) 

