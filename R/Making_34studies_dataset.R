library("curatedBreastData")
data("curatedBreastDataExprSetList")
cbc <- processExpressionSetList(exprSetList=curatedBreastDataExprSetList, outputFileDirectory = "data/")
library("Coincide")
dataMatrixList <- exprSetListToMatrixList(cbc,featureDataFieldName="gene_symbol")
names(dataMatrixList) <- names(cbc)
output <- mergeDatasetList(datasetList=dataMatrixList,minNumGenes = 1, minNumPatients = 1,batchNormalize = c('BMC')); 
# normalization availiable "BMC", "none", "combat". But combat doesn't work for some reason
library(readr)
write_csv(data.frame(patient_ID = colnames(output$mergedExprMatrix), t(output$mergedExprMatrix)), "./bmcMergedAND.csv")
#here we've saved merged 34 studies using intersection merge.

library(tidyverse)
result = {}
for (study in names(cbc))
{
	eset <- cbc[[study]]
	hormone <- as.numeric(as.character(eset$hormone_therapyClass))
	patient_id <- eset$patient_ID
	radio <- as.numeric(as.character(eset$radiotherapy))
	surgery <- eset$surgery_type
	chemo <- as.numeric(as.character(eset$chemotherapy))
	pCR <- eset$pCR
	RFS <- eset$RFS
	DFS <- eset$DFS
	posOutcome = if_else(is.na(pCR), 0, as.double(pCR)) + if_else(is.na(RFS), 0, as.double(RFS)) + if_else(is.na(DFS), 0, as.double(DFS))
	one_study <- cbind(c(study), patient_id, radio, surgery, chemo, hormone, pCR, RFS, DFS, posOutcome)
	colnames(one_study)[1] = "study"
	colnames(one_study)[2] = "patient_ID"
	result = rbind(result, one_study)
}
write.table(result, file="mldata34studies.csv", quote=FALSE, sep=",", row.names = FALSE)
#here we've saved mldata for ml studies

for (study in names(cbc))
{
	exprset2 <- cbc[[study]]
	teset <- exprs(exprset2)
	teset2 <- t(teset)
	rwnames <- rownames(teset2)
	teset3 <- cbind(rwnames, teset2)
	colnames(teset3)[1] = "patient_ID"
	write.table(teset3, paste0(study, ".csv"), sep=",", row.names = FALSE, quote = FALSE)
}
#here we've saved all 34 studies separately

data("clinicalData")
write.table(clinicalData$clinicalTable, file="clinicalTable.csv", sep=',', row.names=FALSE, quote=FALSE)
#here we've saved clinical table for all patients

dataMatrixList <- list();
for(d in 1:length(cbc)){
  dataMatrixList[[d]] <- data.matrix(cbc[[d]]) - rowMeans(data.matrix(cbc[[d]]),na.rm=TRUE,dims=1)
  names(dataMatrixList)[d] <- names(cbc)[d];
  rownames(dataMatrixList[[d]]) <- rownames(cbc[[d]])
}
#this code is from coincide's function mergeDatasetList which implements BMC normalization

merged_OR = {}
for (study in dataMatrixList)
{
  study <- t(study)
  rwnames <- rownames(study)
  study <- cbind(rwnames, study)
  colnames(study)[1] = "patient_ID"
  merged_OR <- bind_rows(merged_OR, data.frame(study))
}
write.table(merged_OR, sep=",", row.names=FALSE, quote=FALSE, file="bmcMergedOR.csv")
#here we've saved merged using union, not intersection, and applied BMC normalization, whic i've retrieved from Coincide