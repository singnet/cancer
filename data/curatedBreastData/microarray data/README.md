## large compressed expression matrices are available at the following links: 
### matrices with no between-study normalization
Expression levels are log 2 transformed original study data as published in GEO or other public source.

- [bc_noNorm.tar.xz](https://snet-bio-data.s3.us-west-2.amazonaws.com/example15bmc/bc_noNorm.tar.xz): expression matrices for 24 studies with 3 studies split into batches for a total of 31 files.  The gene symbols in this set are current as of August, 2020, unlike the expression sets linked below.  These have the original gene assignments from the curatedBreastData R package.  The ~10% of symbols without a current match based on this [HGNC file](https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&col=gd_prev_sym&col=gd_aliases&col=gd_pub_eg_id&col=gd_pub_ensembl_id&status=Approved&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit) are retained with "_obs" added to the original symbol.
### matrices with batch mean centered (BMC) normalization
Gene expression levels in each study batch are centered to each gene's mean expression value across all batch samples.  The gene symbols are updated to current HGNC values (as of 8/20) This procedure eliminates some of the originally included probes in each data set and chooses the probe with the highest variance among probes mapped to the same gene.
  
- [example15bmc.tar.xz](https://snet-bio-data.s3.us-west-2.amazonaws.com/example15bmc/example15bmc.tar.xz): 17 expression matrices from 15 studies from the complete set that have > 40 samples and > 10,000 gene expression levels.
- [ex15bmcMerged.csv.xz](https://snet-bio-data.s3.us-west-2.amazonaws.com/example15bmc/ex15bmcMerged.csv.xz): This is the avove 17 data sets merged into one matrix.

### matrices with smooth quantile normalization
Expression levels are normalized using a procedure that computes a weighted average of quantile normalization over all samples and quantile normaliztion within sample groups at each feature quantile, as implemented in the R Bioconductor [qsmooth](https://www.bioconductor.org/packages/release/bioc/html/qsmooth.html) package described [here](https://pubmed.ncbi.nlm.nih.gov/29036413/).  Batched studies are first normalized using empirical bayes normalization.
These expression sets include probes that don't map to current gene symbols, indicated by the suffix "_obs" at the end of the symbol string in the column heading; filter these out to get the same feature set as in the non-normalised and robust quantile normed sets above.  They also differ from the previous versions by including imputation of missing values.
- [in progress](https://snet-bio-data.s3.us-west-2.amazonaws.com/example15bmc/.tar.xz): These subsets of 10 studies are normalized within the groups defined by the `vital_status` outcome variable.
- [inprogress](https://snet-bio-data.s3.us-west-2.amazonaws.com/example15bmc/.tar.xz): The above data sets mean centered and standard deviation scaled by gene.
- [in progress](https://snet-bio-data.s3.us-west-2.amazonaws.com/example15bmc/.tar.xz):  These subsets of 11 studies are normalized within the groups defined by the `recurrence_status` outcome variable.
- [in pgrogress](https://snet-bio-data.s3.us-west-2.amazonaws.com/example15bmc/):  The above data sets mean centered and standard deviation scaled by gene.
