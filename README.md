# OpenCog/SNet precision medicine for clinical trials poc project
## Questions to answer:
- what patterns of patient biomarkers characterize treatment success?
- what patterns of patient biomarkers characterize adverse events?
- what combination of biomarkers and treatment parameters characterize best patient outcomes?

## Breast, ovarian, and pancreatic cancer tumor transcriptomes and clinical data sets
Each cancer set has data from multiple studies with different gene sets and clincal variables that can be analyzed as an ensemble/meta-analysis or merged into one large matrix.  Meta-analysis is more powerful with standard statistical methods due to data loss when variables from different studies are aligned and merged.

## Data sources
### Clustering Intra and Inter DatasEts
[**curatedBreastData:**](https://bioconductor.org/packages/release/data/experiment/html/curatedBreastData.html)  4,923 breast tumor microarray expression sets from 2,613 patients in 20 studies published as a Bioconductor R package [[paper](https://www.ncbi.nlm.nih.gov/pubmed/24303324)] [[github](https://github.com/kplaney/curatedBreastCancer)].  Related files have **bc** prefix  

CoINcIDE is an unsupervised meta-graph clustering algorithm used to sub-type tumors from gene expression profiles from multiple patient study cohorts: [paper](https://www.ncbi.nlm.nih.gov/pubmed/26961683), author's  [github](https://github.com/kplaney/CoINcIDE).  
The author's github includes useful but outdated R code for processing and merging microarray data sets from [GEO](https://www.ncbi.nlm.nih.gov/geo/). Updating the code for current use is ongoing, see _cancer_ branch of fork of author's github repository in [_CoINcIDE_](https://github.com/mjsduncan/CoINcIDE/tree/cancer).  

[Link](https://www.ncbi.nlm.nih.gov/pubmed?linkname=pubmed_pubmed_citedin&from_uid=26961683) to papers citing CoINcIDE in pubmed for current related research in precision medicine.
  
### A Three-Gene Model to Robustly Identify Breast Cancer Molecular Subtypes
[**MetaGxBreast:**](https://www.bioconductor.org/packages/release/data/experiment/html/MetaGxBreast.html) in progress  
**MetaGxOvarian** in progress  
**MetaGxPancreas** in progress  
  
[MetaGxData: Clinically Annotated Breast, Ovarian and Pancreatic Cancer Datasets and Their Use in Generating a Multi-Cancer Gene Signature](https://pubmed.ncbi.nlm.nih.gov/31217513/)  

**pdf copies of papers are in the [lit](./lit) dircetory**  

## additional data and method sources

### TCGA
1073 samples already included in MetaGXBreast dataset.  These samples have other -omics assay data available for data integration analyses (whole genome sequencing, DNA methylation, proteomics, etc)  
[Link](https://www.nature.com/articles/nature11412) to multi-omics breast cancer sub-typing paper with analysis data available from TCGA.  This a good review for understanding current thinking about breast cancer.  
  
TCGA pan-cancer literature [index](https://www.cell.com/pb-assets/consortium/pancanceratlas/pancani3/index.html)  
163 normal tissue frome breast cancer patients [search table](https://portal.gdc.cancer.gov/exploration?facetTab=cases&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.primary_site%22%2C%22value%22%3A%5B%22breast%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.samples.sample_type%22%2C%22value%22%3A%5B%22solid%20tissue%20normal%22%5D%7D%7D%5D%7D)  
1,145 blood samples bc [search table](https://portal.gdc.cancer.gov/exploration?facetTab=cases&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.primary_site%22%2C%22value%22%3A%5B%22breast%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.samples.sample_type%22%2C%22value%22%3A%5B%22blood%20derived%20normal%22%5D%7D%7D%5D%7D)

### other data and methods links

state of the art tumor classification: [Dynamic Classification Using Case-Specific Training Cohorts Outperforms Static Gene Expression Signatures in Breast Cancer](https://pubmed.ncbi.nlm.nih.gov/25274406/)
- [Integration of RNA-Seq Data With Heterogeneous Microarray Data for Breast Cancer Profiling](https://pubmed.ncbi.nlm.nih.gov/29157215/)  
- [Mining Data and Metadata From the Gene Expression Omnibus](https://pubmed.ncbi.nlm.nih.gov/30594974/)  
- [Tree-Weighting for Multi-Study Ensemble Learners](https://pubmed.ncbi.nlm.nih.gov/31797618/)
- [OncoKB: A Precision Oncology Knowledge Base](https://pubmed.ncbi.nlm.nih.gov/28890946/)  
- [BioDataome: A Collection of Uniformly Preprocessed and Automatically Annotated Datasets for Data-Driven Biology](https://pubmed.ncbi.nlm.nih.gov/29688366/)
- [Microarray Meta-Analysis Database (M(2)DB): A Uniformly Pre-Processed, Quality Controlled, and Manually Curated Human Clinical Microarray Database](https://pubmed.ncbi.nlm.nih.gov/20698961/)
