### embedding plots
library(tidyverse)

# get embedding vectors
vs <- list.files("ml/embedding vectors/current/", "csv", full.names = TRUE)
em <- map(vs, read_tsv)

# clean up
em <- set_names(em, str_replace_all(str_sub(vs, 47, -16), "-", "_")) %>%
  map(~ rename_with(., function(x) paste0("L", x), !patient_ID)) %>%
  map(~ mutate(., patient_ID = as.character(patient_ID)))

map(em, ~ summary(c(as.matrix(select(., -1)))))
# $moses50_all
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.4531090 -0.0043808 -0.0000041  0.0000000  0.0043350  0.6384998 
# 
# $moses500_withoutpln
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.4583295 -0.0070055 -0.0000124  0.0000000  0.0069575  0.6134178 
# 
# $xgb50_all
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.5042805 -0.0039439 -0.0000018  0.0000000  0.0039143  0.6112367 

pheno <- read_csv("data/curatedBreastData/embedding_vector_state_and_outcome.csv") %>%
  mutate(patient_ID = as.character(patient_ID), posOutcome = as.factor(posOutcome), pam50 = pam_coincide) %>%
  select(- pam_coincide) %>%
  mutate(series_id = str_remove(series_id, ",GSE25066|GSE16716,"))

# tSNE plots
library(Rtsne)

# column to name
em <- map(em, ~ column_to_rownames(., "patient_ID"))

tem <- map(em, Rtsne)

# make tibbles for plotting
em <- map(em, ~ rownames_to_column(., "patient_ID"))

temPlot <- map2(em, tem, ~ bind_cols(select(.x, 1), as_tibble(.y$Y))) %>%
  map(~ left_join(., pheno)) %>%
  map(~ ggplot(., aes(x = V1, y = V2, color = series_id, shape = posOutcome)))
       
for(n in names(temPlot)) {
  temPlot[[n]] <- temPlot[[n]]  + scale_shape_manual(values = c(1, 4)) + labs(title = n)
  ggsave(paste0(n, ".png"), temPlot[[n]] + geom_point() + coord_fixed(), "png", "ml/embedding vectors/current/")
}

# color by pam50
temPlot <- map2(em, tem, ~ bind_cols(select(.x, 1), as_tibble(.y$Y))) %>%
  map(~ left_join(., pheno)) %>%
  map(~ ggplot(., aes(x = V1, y = V2, color = pam50, shape = posOutcome)))

for(n in names(temPlot)) {
  temPlot[[n]] <- temPlot[[n]]  + scale_shape_manual(values = c(1, 4)) + labs(title = n)
  ggsave(paste0(n, "_pam50.png"), temPlot[[n]] + geom_point() + coord_fixed(), "png", "ml/embedding vectors/current/")
}

# color by infogan clustering
temPlot <- map2(em, tem, ~ bind_cols(select(.x, 1), as_tibble(.y$Y))) %>%
  map(~ left_join(., pheno)) %>%
  map(~ ggplot(., aes(x = V1, y = V2, color = p5, shape = posOutcome)))

for(n in names(temPlot)) {
  temPlot[[n]] <- temPlot[[n]]  + scale_shape_manual(values = c(1, 4)) + labs(title = n)
  ggsave(paste0(n, "_infogan.png"), temPlot[[n]] + geom_point() + coord_fixed(), "png", "ml/embedding vectors/current/")
}

# make separate embedding sets by study
em <- map(em, ~ rownames_to_column(., "patient_ID"))

studies <- set_names(unique(pull(pheno, series_id))) %>%
  map(~ filter(pheno, series_id == .) %>% pull(patient_ID)) %>%
  map( ~ map(em, filter, patient_ID %in% .)) %>%
  map(~ map(., column_to_rownames, "patient_ID"))

# apply tsne
set.seed(1999)
# perp <- map_int(studies, ~ dim(.[[1]])[1])
stem <- map(studies, ~ map(.x, Rtsne, perplexity = 15, theta = 0, verbose = TRUE))   # .y / 3 - 3 
  
# make three copies of covariates & outcome for each study
studiesDat <- map(studies, ~ map(., rownames_to_column, "patient_ID")) %>%
  map(~ map(., select, "patient_ID")) %>%
  map(~ map(., left_join, select(pheno, - series_id)))
  
# make function to produce ggplot colored by input variable
library(rlang)
tsnePlot <- function(df, cvar, svar) ggplot(select(df, sym(cvar), sym(svar), sym("x"), sym("y")),
                                            aes(x, y, color = !! sym(cvar), shape = !! sym(svar))) +
                                            scale_shape_manual(values = c(1, 4)) + geom_point() +
                                            theme(axis.title = element_blank(), axis.text = element_blank(),
                                                  axis.ticks = element_blank(), legend.text = element_text(size = 8),
                                                  legend.title = element_text(size = 8))

tsnePlots <- tibble(study = names(studiesDat), data = studies, tsne = stem) %>%
  unnest(cols = c(data, tsne)) %>%
  mutate(data = map2(data, tsne, ~ mutate(.x, x = .y$Y[, 1], y = .y$Y[, 2]))) %>%
  mutate(embedding = names(data))

tsnePlots <- mutate(tsnePlots, pam50 = map(data, ~ tsnePlot(., "pam50", "posOutcome")))
tsnePlots <- mutate(tsnePlots, igan = map(data, ~ tsnePlot(., "p5", "posOutcome")))

# combine & save plots by study
library(ggpubr)

for(n in unique(tsnePlots$study)) {
  plot6 <- ggarrange(
    ggarrange(plotlist = filter(tsnePlots, study == n) %>% pull(pam50),
              labels = pull(tsnePlots, embedding)[1:3],
              font.label = list(size = 8),
              ncol = 3, legend = "right", common.legend = TRUE),
    ggarrange(plotlist = filter(tsnePlots, study == n) %>% pull(igan),
    ncol = 3, nrow = 1, legend = "right", common.legend = TRUE),
  nrow = 2)
ggsave(paste0(n, ".png"), annotate_figure(plot6, top = text_grob(n, face = "bold", size = 10)),
       "png", "ml/embedding vectors/current/plots/")
}

# calculate 3 dimensional tsne spce to assess clustering
# apply tsne
set.seed(1999)
stem50 <- map(studies, ~ map(.x, Rtsne, dims = 3, perplexity = 15, theta = 0, max_iter = 10000, verbose = TRUE))

# make three copies of covariates & outcome for each study
studies50Dat <- map(studies, ~ map(., rownames_to_column, "patient_ID")) %>%
  map(~ map(., select, "patient_ID")) %>%
  map(~ map(., left_join, select(pheno, - series_id)))

studies50Dat<- tibble(study = names(studies50Dat), data = studies50Dat, tsne = stem50) %>%
  unnest(cols = c(data, tsne)) %>%
  mutate(data = map2(data, tsne, ~ mutate(.x, x = .y$Y[, 1], y = .y$Y[, 2], z = .y$Y[, 3]))) %>%
  mutate(embedding = names(data))

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
