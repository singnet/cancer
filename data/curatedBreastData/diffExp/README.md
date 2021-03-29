# gene lists and GO gene set lists

**genes15studies.txt -** the 8832  genes from the 15 study expression set intersection  

**filteredGenesXXstudies.txt -** above list minus genes with expression levels < 5.0  
  **15** for complete set, **4** for the four studies with tamoxifen treatment  
  
**topXXXgenesXXstudies[Filtered].csv -** the XXX genes with p value for differential expression between <=5 year relapse group (posOutcome = 0) and no relapse group (posOutcome = 1) less than 0.05  
**[cb, tam]XX[BP, MF, CC][01, 05].txt -** significant GO terms at p value alpha 0.01 or 0.05 for the 3 GO DAGs (biological process, molecular function, and cellular component) where **cb** is the 15 study set and **tam** is the 4 study subset
