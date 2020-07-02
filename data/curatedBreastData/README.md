## large compressed expression matrices are available at the following links:  
- [example15bmc.tar.xz](https://snet-bio-data.s3.us-west-2.amazonaws.com/example15bmc/example15bmc.tar.xz): 17 expression matrices with batch mean centered (bmc) normalization
- [ex15bmcMerged.csv.xz](https://snet-bio-data.s3-us-west-2.amazonaws.com/example15bmc/ex15bmcMerged.csv.xz): intersection of 17 matrices with bmc normalization
- [merged-combat15.csv.xz](https://snet-bio-data.s3-us-west-2.amazonaws.com/example15bmc/merged-combat15.csv.xz):  intersection of 17 matrices with empirical bayes normalization
## curatedBreastData files
- **pam50** PAM50 centroid data for reproducing the Coincide paper results
- **bcTabs.ods** data dictionary and clinical data from the curatedBreastData R package used in Coincide paper

### bcTabs.ods
**bcVarDesc** -  description of all clinical variables with type annotations:
- **study:**  variables indicating membership in a specific study set or batch
- **patient:**  variables associated with individual patients
- **treatment:**  variables associated with treatment
- **state:**  variables associated with patient condition.  these have a **time** annotation indicating if the state was measured before or after treatment

**bcCovariateVars** - variables with patient demographics, medical history, miscellaneous clinical features and study information  
**bcTreatmentVars** - treatment drug and drug type variables  
**bcOutcomeVars** - treatment outcome variables including including time to event data  
**bcClinicalTable** - dump of clinical variables for all patients in breast cancer data set  

### Gene Expression data
 [**example15bmc**](https://s3.console.aws.amazon.com/s3/buckets/snet-bio-data/example15bmc/example15bmc/?region=us-west-2&tab=overview) - subset of patients from 15 studies (2 studies are divided into 2 batches so 17 expression level files) with at least 3 years of post-treatment follow up  
 [**merged-combat15.csv.xz**](https://snet-bio-data.s3-us-west-2.amazonaws.com/example15bmc/merged-combat15.csv.xz) - the 15 studies above combined using empirical bayes normalization as implemented by the `ComBat` function of the `sva` Bioconductor package of expression values leaving 8,832 overlapping genes (studies range from 12,722 to 18,260 genes).  

### Initial ML data set
For initial supervised ML pipeline experiments, a set of composite binary treatment and outcome variables [[bmc15mlData1.csv]](./bcDump/example15bmc/bmc15mlData1.csv) has been constructed to combine with the subset of expression data sets or the merged expression dataset described in the previous section.  
### `bmc15mldata`
`study` - csv file name containing sample expression data  
`patient_ID` - key to match clinical, expression and outcome data  
  
**binary treatment variables**
- `radio` - radiotherapy (442/2225)
- `surgery` - breast sparing vs mastectomy (144/2225)
- `hormone` - hormone therapy (855/2225)
- `chemo` - chemotherapy (1186/2225)
  
**outcome variables**
- `pCR` - complete remission by pathology (260/1134)
- `RFS` - relapse free survival (736/1030)
- `DFS` - disease free survival (512/656)
- `posOutcome` - any of the above are positive (1386/2225)  

One or more treatment variables and an outcome variable should be selected and combined with the individual or merged expression datasets using `patient_ID` as the key.

![heatmap of treatment variables](../plots/bmc15txHeatmap3.png)  
The heatmap of granular drug type variables shows hormone based treatment regimens clustering on the left and chemotherapy regimens on the right.  The color bar on the left shows which study the patient is from.
