import pandas as pd

import glob
import os


#making ml data
def fillTreatment(treatment):
    hormone = "0"
    chemo = "0"
    if(treatment == "untreated"):
        return hormone, chemo #already filled that
    if(treatment == "NA"):
        hormone = "NA"
        chemo = "NA"
    elif(treatment == "chemo.plus.hormono"):
        hormone = "1"
        chemo = "1"
    elif(treatment == "hormonotherapy"):
        hormone = "1"
    elif(treatment == "chemotherapy"):
        chemo = "1"
    return hormone, chemo


path1 = "metaGXcovarTable.csv"
path2 = "mldata.csv"

f1 = open(path1, 'r')
f2 = open(path2, 'w')

line1 = f1.readline()
colnames = line1.split("\n")[0].split(",")

treatment_idx = 15
vital_status = 14
sample_name_idx = 1
recurrence_idx = 13


count = 0

f2.write("sample_name,alt_sample_name,recurrence,posOutcome,hormone,chemo\n")
full_count = 0
for line1 in f1:
    full_count += 1
    splitted = line1.split("\n")[0].split(",")
    hormone, chemo = fillTreatment(splitted[treatment_idx])
    if(splitted[recurrence_idx] == "norecurrence"):
        recur = "0"
    elif(splitted[recurrence_idx] == "recurrence"):
        recur = "1"
    elif(splitted[recurrence_idx] == "NA"):
        recur = "NA"

    if(splitted[vital_status] == "deceased"):
        posOutcome = "0"
    elif(splitted[vital_status] == "living"):
        posOutcome = "1"
    elif(splitted[vital_status] == "NA"):
        count += 1
        posOutcome = "NA"


    line_to_write = splitted[sample_name_idx] + "," + splitted[sample_name_idx] + "," + str(recur) + "," + posOutcome + "," + hormone + "," + chemo + "\n"
    f2.write(line_to_write)

print(str(count) + "/" + str(full_count))
f1.close()
f2.close()
