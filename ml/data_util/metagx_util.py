import os
import lzma
import pandas

import sys
import os
import lzma
import tarfile
import random
from collections import defaultdict
import math

import util

from util import *


treatment_columns_metagx = 'tumor_size', 'chemo', 'hormono', 'grade', 'N', 'age'


def ternary(column, nan):
    """
    make ternary data from column
    represent true, false, nan as 1, -1, 0
    """
    # zeros in column represent either false or nan
    # set them to -1
    column.loc[column == 0] -= 1
    # nan should be zero
    # invert nan so that it is 1 for not nan
    # thus multiplying by not-nan will preserve column values
    column *= (~nan * 1)
    return column


def ternary_column(column):
    assert len(column.unique()) == 3
    column += (column == 0) * -1
    column = column.fillna(0)
    return column


def process_continious(column):
    # check if there zeros, if so
    # add one and fill na with zeros
    zeros = bool(len(column[column == 0]))
    if zeros:
        column += 1
    column = column.fillna(0)
    return column


def convert_covars(merged):
    dummies = pandas.get_dummies(merged.treatment, prefix='t', dummy_na=True)
#Index(['treatment_chemo.plus.hormono', 'treatment_chemotherapy',
#       'treatment_hormonotherapy', 'treatment_untreated'],
    # dummies allways have valid value and can be added
    hormo = dummies.t_hormonotherapy + dummies['t_chemo.plus.hormono']
    hormo = (hormo > 0) * 1
    # merged.chemo might be nan
    chemo_orig = (merged.chemo > 0) * 1
    chemo = dummies.t_chemotherapy + dummies['t_chemo.plus.hormono'] + chemo_orig
    chemo = (chemo > 0) * 1
    untreated = dummies.t_untreated
    # need to update nan, based on chemo
    nan = (dummies.t_nan - chemo) > 0
    merged.chemo = ternary(chemo, nan)
    merged.insert(3, 'hormono', ternary(hormo, nan))
    # verify that chemo + hormo hasn't changed
    shape_chemo_hormono = merged[merged.treatment == 'chemo.plus.hormono'].shape
    tmp = merged[merged.chemo == 1]
    assert shape_chemo_hormono == tmp[tmp.hormono == 1].shape
    # create posOutcome from vital_status
    posOutcome = (merged.vital_status == 'living') * 1
    posOutcome += (merged.vital_status == 'deceased') * -1
    merged.insert(3, 'posOutcome', posOutcome)
    merged.N = ternary_column(merged.N)
    merged.grade = process_continious(merged.grade)
    age = process_continious(merged.age_at_initial_pathologic_diagnosis)
    merged.insert(4, 'age', age)
    merged.tumor_size = process_continious(merged.tumor_size)
    merged.insert(4, 'patient_ID', merged.sample_name)
    return merged



def load_metagx_dataset(base_path, min_genes=10000):
    """
    load and merge metagx dataset

    Parameters
    -------------
    base_path: str
        path to the directory with metaGXcovarTable.csv.xz and 'microarray data' directory
    min_genes: int
        minimal number of genes in study to be added to merged dataset
    """
    microarray_path = os.path.join(base_path, 'microarray data')
    norm_path = os.path.join(microarray_path, 'mGXrlNorm.tar.xz')
    norm_path1 = os.path.join(microarray_path, 'mGXrlNormNoBatch.tar.xz')
    not_norm_path = os.path.join(microarray_path, 'noNorm/DFHCC3_noNorm.csv.xz')
    covariates_path = os.path.join(base_path, 'metaGXcovarTable.csv.xz')

    dtype = {'hormonotherapy': pandas.Int64Dtype(),
             'chemo.plus.hormono': pandas.Int64Dtype(),
             'chemotherapy': pandas.Int64Dtype(),
             'untreated': pandas.Int64Dtype()}

    covariates_table = pandas.read_csv(lzma.open(covariates_path), dtype=dtype)
    study_tables = dict()
    tar1 = tarfile.open(norm_path, "r:xz")
    tar2 = tarfile.open(norm_path1, 'r:xz')
    for tar in (tar1, tar2):
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                study_name = member.name.split('/')[1].split('.')[0]
                assert study_name not in study_tables
                study_tables[study_name] = pandas.read_csv(lzma.open(f), encoding="utf-8")

    sets_dict = dict()
    for key, value in study_tables.items():
        sets_dict[key] = set(value.columns[1:].to_list())

    study_mapping = dict()
    i = 0
    for study, table in study_tables.items():
        study_mapping[study.split('_')[0]] = i
        table.insert(1, 'study', [i for _ in range(len(table))])
        i += 1

    # filter studies by size
    result = None
    filtered_studies = []
    i = 0
    for k, value in sets_dict.items():
        if len(value) < min_genes:
            continue
        if result is None:
            result = value
        else:
            result = result.intersection(value)
        filtered_studies.append(k)
        i += 1

    genes_list = list(result)
    genes_features = pandas.concat([study_tables[k][['study', 'sample_name'] + genes_list]  for k in filtered_studies])
    assert all(genes_features.sample_name.isin(covariates_table.sample_name))
    covariates_table = covariates_table[covariates_table.sample_name.isin(genes_features.sample_name)]
    tmp_study = [study_mapping[x] for x in covariates_table.study]
    covariates_table.study = tmp_study
    covariates_table.reset_index()

    merged = pandas.merge(genes_features, covariates_table.drop(columns=['study']), left_on='sample_name', right_on='sample_name')
    merged.insert(0, 'row_num', range(0,len(merged)))
    merged = merged.reset_index()
    merged = merged[merged.sample_type == 'tumor']

    result = dict(covariates=covariates_table, genes_features=genes_list,
            merged=merged)

    convert_covars(merged)
    return result


if __name__ == '__main__':
    data = load_metagx_dataset('/home/noskill/projects/cancer/data/metaGxBreast/', min_genes=5000)
    merged = data['merged']
    genes_list = data['genes_features']

    train_data, train_labels, val_data, val_labels = util.random_split(merged,
                                                              treatment_columns_metagx + genes_list,
                                                              ['posOutcome'],
                                                              balance_validation=True,
                                                              balance_by_study=False)

    train_data, train_labels, val_data, val_labels = util.random_split(merged,
                                                              treatment_columns_metagx + genes_list,
                                                              ['study'],
                                                              balance_validation=False,
                                                              balance_by_study=False)