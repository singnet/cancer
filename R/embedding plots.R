### embedding plots
library(tidyverse)

# get embedding vectors
vs <- list.files("ml/embedding vectors", "json", full.names = TRUE)[-1]
em <- map(vs, read_tsv)

# clean up
em <- set_names(em, str_replace_all(str_sub(vs, 38, -17), "-", "_")) %>%
  map(~ rename_with(., function(x) paste0("L", x), !patient_ID)) %>%
  map(~ mutate(., patient_ID = as.character(patient_ID)))

names(em)[1] <- "eight_genes_only"

map(em, ~ summary(c(as.matrix(select(., -1)))))
# $eight_genes_only
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# -0.573877 -0.002273 -0.000002  0.000000  0.002245  0.691049 
# 
# $moses50_OnlyGenexpr
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.4880615 -0.0047729 -0.0000013  0.0000000  0.0047069  0.7463042 
# 
# $moses50_withoutpatientsdata
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.3981823 -0.0032196 -0.0000016  0.0000000  0.0031838  0.7341006 
# 
# $moses50_withoutplnresult
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.4652644 -0.0044198 -0.0000006  0.0000000  0.0043406  0.8276420 
# 
# $moses50_all
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.5085798 -0.0041406 -0.0000011  0.0000000  0.0040958  0.6580318 
# 
# $moses50_genexp_plus_BP
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.3981857 -0.0032190 -0.0000011  0.0000000  0.0031819  0.7341005 
# 
# $normalized_moses50_all
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.5085798 -0.0041406 -0.0000011  0.0000000  0.0040958  0.6580318 
# 
# $normalized_V2_moses50_all
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.5085798 -0.0041406 -0.0000011  0.0000000  0.0040958  0.6580318 
# 
# $xgb50_OnlyGenexpr
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.5465492 -0.0045296 -0.0000016  0.0000000  0.0044889  0.6613333 
# 
# $xgb50_withoutpatientsdata
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.5446278 -0.0030901 -0.0000016  0.0000000  0.0030505  0.6036319 
# 
# $xgb50_withoutplnresult
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.4540623 -0.0043378 -0.0000005  0.0000000  0.0042571  0.8182119 
# 
# $xgb50_all
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# -0.487704 -0.003996 -0.000001  0.000000  0.003945  0.548485 
# 
# $xgb50_genexp_plus_BP
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.5446319 -0.0030913 -0.0000014  0.0000000  0.0030471  0.6036362 

pheno <- read_csv("data/curatedBreastData/embedding_vector_state_and_outcome.csv") %>%
  mutate(patient_ID = as.character(patient_ID))

# tSNE plots
library(Rtsne)

# column to name
em <- map(em, ~ column_to_rownames(., "patient_ID"))

tem <- map(em, Rtsne)

# make tibbles for plotting
em <- map(em, ~ rownames_to_column(., "patient_ID"))

temPlot <- map2(em, tem, ~ bind_cols(select(.x, 1), as_tibble(.y$Y))) %>%
  map(~ left_join(., pheno)) %>%
  map(~ mutate(., posOutcome = as.factor(posOutcome))) %>%
  map(~ ggplot(., aes(x = V1, y = V2, color = series_id, shape = posOutcome)))
       
for(n in names(temPlot)) {
  temPlot[[n]] <- temPlot[[n]]  + scale_shape_manual(values = c(1, 4)) + labs(title = n)
  ggsave(paste0(n, ".png"), temPlot[[n]] + geom_point(), "png", "ml/embedding vectors/tsne/")
}


# # mpm template
# sm <- map(em, mpm::mpm,
#           logtrans = TRUE,
#           logrepl = ,
#           closure = ,
#           center = ,
#           normal = ,
#           row.weight = "constant",
#           col.weight = "constantz",
#           RW = 1,
#           CW = 1,
#           pos.row = FALSE,
#           pos.column = FALSE)

# transpose so columns are samples
emt <- map(em, ~ pivot_longer(., - patient_ID) %>% pivot_wider(names_from = patient_ID, values_from = value))

# make pca plots
library(mpm)
mpm1 <- map(emt, mpm, logtrans = FALSE, center = "none", normal = "none")

# fix name error
mpm1 <- map(mpm1, ~ modify_at(., "row.names", function(x) str_sub(x, 3, -2)))
mpm1 <- map(mpm1, ~ modify_at(., "row.names", function(x) str_remove_all(x, "\"")))
mpm1 <- map(mpm1, ~ modify_at(., "row.names", function(x) str_remove_all(x, "\n")))
mpm1 <- map(mpm1, ~modify_at(., "row.names", function(x) unlist(str_split(x, ", "))))
for (i in seq_along(mpm1)) {
  row.names(mpm1[[i]]$TData) <- mpm1[[i]]$row.names
}

mpm1sum <- map(mpm1, summary, 10)

# map(mpm1sum, print, what = "all")
map(mpm1sum, `[[`, "VPF")

# plotting
# get labels
study <- map(mpm1, ~ pluck(., "col.names")) %>%
  map(recode, !!!pull(pheno, series_id, patient_ID))

outcome <-  map(mpm1, ~ pluck(., "col.names")) %>%
  map(recode, !!!pull(pheno, posOutcome, patient_ID))

for(n in names(mpm1)) {
  png(paste0("./ml/embedding vectors/plots/", n, "-study.png"))
  plot.mpm(mpm1[[n]],
           scale = "eigen",
           dim = c(1, 2),
           zoom = c(1, 20),
           col.group = study[[n]],
           colors = c("orange1", "red", hcl.colors(15, "Dark 2", 0.5)),
           col.areas = FALSE,
           label.tol = 3,
           label.col.tol = 0,
           col.size = 1,
           do.smoothScatter = FALSE,
           main = paste0(n, " / study"),
           sub = "PCA of untransformed embedding vectors")
  dev.off()
}

for(n in names(mpm1)) {
  png(paste0("./ml/embedding vectors/plots/", n, "-outcome.png"))
  plot.mpm(mpm1[[n]],
           scale = "eigen",
           dim = c(1, 2),
           zoom = c(1, 20),
           col.group = outcome[[n]],
           colors = c("orange1", "blue", "#00000088", "#FF000088"),
           col.areas = FALSE,
           label.tol = 3,
           label.col.tol = 0,
           col.size = 1,
           do.smoothScatter = FALSE,
           main = paste0(n, " / outcome"),
           sub = "PCA of untransformed embedding vectors")
  dev.off()
}

map(sm, mpm::plot.mpm, label.tol = 10, label.col.tol = 10)
