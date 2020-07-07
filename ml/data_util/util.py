import sys
import os
import lzma
import math
import random
import string
from collections import defaultdict
from typing import *

import numpy
import pandas

from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix


def split_by_study(merged, bmc, study_name=None):
    """
    Split one study out for cross-validation

    merged: pandas.DataFrame
        Genes + treatments table
    bmc: pandas.DataFrame
        averaged treatment table
    study_name: str
        Optional param - process only singe study

    """
    for eval_study in set(bmc.study):
        if study_name:
            eval_study = study_name
        print(eval_study)
        bmc_train = bmc[bmc.study != eval_study]
        bmc_val = bmc[bmc.study == eval_study]
        assert (not set(bmc_train.patient_ID).intersection(set(bmc_val.patient_ID)))

        train_split = merged[merged.patient_ID.isin(bmc_train.patient_ID)]
        val_split = merged[merged.patient_ID.isin(bmc_val.patient_ID)]
        assert val_split.patient_ID.to_list() == bmc_val.patient_ID.to_list()
        train_data = train_split[feature_columns].to_numpy()
        train_labels = train_split[label_columns].to_numpy().astype(int)
        val_data = val_split[feature_columns].to_numpy()
        val_labels = val_split[label_columns].to_numpy().astype(int)
        yield train_data, train_labels, val_data, val_labels
        if study_name:
            break


def select_balanced_idx(study, num, balance_train=False):
    if not num % 2 == 0:
        num = num + 1
    validation = []
    pos_outcome = study[study.posOutcome == 1].patient_ID
    neg_outcome = study[study.posOutcome == 0].patient_ID
    pos_idx = numpy.arange(len(pos_outcome))
    neg_idx = numpy.arange(len(neg_outcome))
    random.shuffle(pos_idx)
    random.shuffle(neg_idx)
    i = 0
    while not (len(validation) >= num):
        validation.append(pos_outcome.iloc[pos_idx[i]])
        validation.append(neg_outcome.iloc[neg_idx[i]])
        i += 1
    train_pos = pos_idx[i:]
    train_neg = neg_idx[i:]
    len_diff = len(train_pos) - len(train_neg)
    if balance_train:
        if len_diff > 0: # train_pos is longer, sample train_neg
            mul = len_diff // len(train_neg) + 1
            train_neg = numpy.hstack([train_neg, ([x for x in train_neg] * mul)[:len_diff]])
        if len_diff < 0:
            len_diff *= -1
            mul = len_diff // len(train_pos) + 1
            train_pos = numpy.hstack([train_pos, ([x for x in train_pos] * mul)[:len_diff]])
    train_idx = pos_outcome.iloc[train_pos].index.to_list() \
            + neg_outcome.iloc[train_neg].index.to_list()
    train = study.loc[train_idx]
    # train = study[~study.patient_ID.isin(validation)]
    validation = study[study.patient_ID.isin(validation)]
    assert not set(train.patient_ID.to_list()).intersection(set(validation.patient_ID.to_list()))
    return train, validation


def resample_patients_by_study(study_patients: Dict[str, list]) -> Dict[str, list]:
    result = defaultdict(list)
    max_length = max([len(x) for x in study_patients.values()])
    for study, lst in study_patients.items():
        result[study] += lst
        to_upsample = max_length - len(lst)
        result[study] += [random.choice(lst) for i in range(to_upsample)]
    assert all([(len(x) == max_length) for x in result.values()])
    return result


def get_loc(patient_ID, frame):
    return frame[frame.patient_ID == patient_ID].row_num.to_list()[0]


def random_split(merged, bmc, feature_columns, label_columns, ratio=0.1, study_name=None, rand=False, to_numpy=True, balance_by_study=False, balance_train=False):
    """
    Split dataset into train and validation sets:

    Parameters
    balance_train: bool
        balance train set by posOutcome
    --------------
    Returns: train_data, train_labels, val_data, val_labels, expected
        expected - confusion matrix expected from classification by ratio of positive/negative for each study
    """
    val_dict = defaultdict(list)
    train_dict = defaultdict(list)
    expected = dict()
    expected['TN'] = 0
    expected['FN'] = 0
    expected['FP'] = 0
    expected['TP'] = 0

    bmc = merged
    for eval_study in set(bmc.study):
        if study_name is not None:
            if study_name != eval_study:
                continue
        study = bmc[bmc.study == eval_study]
        num_select = math.ceil(len(study) * ratio)
        study_patients = bmc[bmc.study == eval_study]
        bmc_train, bmc_val = select_balanced_idx(study_patients, num_select, balance_train)
        pos_prob_train = bmc_train.posOutcome.sum() / len(bmc_train)
        neg_prob_train = 1 - pos_prob_train
        P = bmc_val.posOutcome.sum()
        N = len(bmc_val) - P
        TN = N * neg_prob_train
        TP= P * pos_prob_train
        FP = N - TN
        FN = P - TP
        expected['TN'] += TN
        expected['TP'] += TP
        expected['FP'] += FP
        expected['FN'] += FN
        val_dict[eval_study] = bmc_val.index.to_list()
        train_dict[eval_study] = bmc_train.index.to_list()
        train_patients = []
        for patient_lst in train_dict.values():
            train_patients += patient_lst
        balance = merged.loc[train_patients].posOutcome.sum() / len(merged.loc[train_patients])
        print("study {0}, balance {1}".format(eval_study, balance))

    if balance_by_study:
        train_dict = resample_patients_by_study(train_dict)
        iloc = []
        for patient_lst in train_dict.values():
            iloc += [get_loc(p, merged) for p in patient_lst]
        train_split = merged.iloc[iloc]
    else:
        train_patients = []
        for patient_lst in train_dict.values():
            train_patients += patient_lst
        train_split = merged.loc[train_patients]
    val_patients = []
    for patient_lst in val_dict.values():
        val_patients += patient_lst
    val_split = merged.loc[val_patients]
    train_data = train_split[feature_columns]
    train_labels = train_split[label_columns]
    val_data = val_split[feature_columns]
    val_labels = val_split[label_columns]
    if rand:
        train_data = numpy.random.randn(*train_data.shape)
        val_data = numpy.random.randn(*val_data.shape)
    if to_numpy:
        train_data = train_data.to_numpy()
        train_labels = train_labels.to_numpy().astype(int).ravel()
        val_data = val_data.to_numpy()
        val_labels = val_labels.to_numpy().astype(int).ravel()
    return train_data, train_labels, val_data, val_labels, expected


def digitize_genes(col, bins=15):
    _, edges = numpy.histogram(col, bins=bins, density=True)
    res = numpy.digitize(col, edges)
    return res - res.min()


def digitize_genes_by_median(col):
    median = col.median()
    return col > median


res_ar = numpy.zeros((16, 4), dtype=numpy.bool_)
for i in range(16):
    ar0 = bin(i).split('0b')[1]
    ar0 = '0' * (4 - len(ar0)) + ar0
    res_ar[i] = [int(x) for x in ar0]



def binary_genes(merged, genes_columns, by_median=True):
    n_patients, n_genes = merged[genes_columns].shape
    result = dict()
    for gene_col in range(len(genes_columns)):
    # for gene_col in range(1000):
        if by_median:
            digitized = digitize_genes_by_median(merged[genes_columns[gene_col]])
            result[genes_columns[gene_col]] = digitized
        else:
            binary_genes = numpy.zeros((n_patients, 4), dtype=numpy.bool_)
            digitized = digitize_genes(merged[genes_columns[gene_col]])
            for i, digit in enumerate(digitized):
                binary_genes[i] = res_ar[digit]
            column_names = [genes_columns[gene_col] + '_{0}'.format(x) for x in range(4)]
            for i, col in enumerate(column_names):
                result[col] = binary_genes[:, i]
    return result



def digitize_non_genes_data(col):
    not_nan = numpy.nan_to_num(col, nan=-1)
    n_unique = len(set(not_nan))
    if 15 < n_unique:
        n_unique = 15
    edge = numpy.histogram_bin_edges(numpy.nan_to_num(col, nan=col.min()), bins=n_unique - 1)
    digits = numpy.digitize(not_nan, edge)
    digits = digits - digits.min()
    return digits


def categorical_non_genes(merged, genes_columns, feature_columns, to_letters=True, dtype=numpy.float16):
    other_columns = [x for x in merged.columns[~merged.columns.isin(genes_columns)] if x in feature_columns]
    non_genes_data = dict()
    for oth in other_columns:
        dig = digitize_non_genes_data(merged[oth])
        if to_letters:
            dtype=object
        ar1 = numpy.zeros(len(merged[oth]), dtype=dtype)
        for i, d in enumerate(dig):
            if to_letters:
                ar1[i] = string.ascii_letters[d]
            else:
                ar1[i] = dtype(d)
        non_genes_data[oth] = ar1
    return non_genes_data


def binary_non_genes(to_category=True):
    non_genes_data = dict()
    for oth in other_columns:
        dig = digitize_non_genes_data(merged[oth])
        set_size = len(set(dig))
        new_col = -1
        for x in range(1, 5):
            if set_size <= 2 ** x:
                new_col = x
                break
        assert new_col != -1
        ar1 = numpy.zeros((len(merged[oth]), new_col), dtype=numpy.bool_)
        for i, d in enumerate(dig):
            ar1[i] = res_ar[d][-new_col:]
        column_names = [oth + '_{0}'.format(x) for x in range(new_col)]
        for i, col_name in enumerate(column_names):
            non_genes_data[col_name] = ar1[:, i]
    return non_genes_data

def binarize_dataset(merged, genes_columns, feature_colum, to_letters):
    """
    merged: pandas.DataFrame
    genes_columns: List[str]
    feature_colum: List[str]
    to_letters: bool
        convert categorical variables to ascii letters
        if false numpy.float16 is used
    """
    patients_id = merged.patient_ID.to_list()
    binary_genes_dict = binary_genes(merged, genes_columns)
    binary_non_genes_dict = categorical_non_genes(merged, genes_columns, feature_colum, to_letters=to_letters)
    binary_non_genes_dict['patient_ID'] = patients_id
    binary_genes_dict['patient_ID'] = patients_id
    binary_non_genes_dict['posOutcome'] = merged.posOutcome.to_list()
    binary_genes_df = pandas.DataFrame(data=binary_genes_dict).sort_values(by='patient_ID') * 1
    binary_non_genes_df = pandas.DataFrame(data=binary_non_genes_dict).sort_values(by='patient_ID')
    result = pandas.merge(binary_genes_df, binary_non_genes_df, left_on='patient_ID', right_on='patient_ID').sort_values(by='patient_ID')
    return result


def compute_metrics(result, y_true, y_pred, x_true, x_pred):
    result['recall'].append(recall_score(y_true, y_pred))
    result['precision'].append( precision_score(y_true, y_pred))
    result['f1'].append(f1_score(y_true, y_pred))
    result['confusion'].append(confusion_matrix(y_true, y_pred))
    result['train_f1'].append(f1_score(x_true, x_pred))
    result['train_confusion'].append(confusion_matrix(x_true, x_pred))
    confusion = result['confusion'][-1]
    accuracy = (confusion[0][0] + confusion[1][1]) / (sum(confusion[0]) + sum(confusion[1]))
    result['accuracy'].append(accuracy)


# Convertors for mapping string data to numbers
def convert_surgery(x, surgery_mapping=dict()):
    if x not in surgery_mapping:
        surgery_mapping[x] = len(surgery_mapping)# + 1
    return surgery_mapping[x]


def convert_node_status(x, mapping=dict()):
    if x == 'NA' or x == 'NaN':
        return numpy.nan
    if not isinstance(x, str) and numpy.isnan(x):
        return x
    if x not in mapping:
        mapping[x] = len(mapping) + 1
    return mapping[x]


def convert_race(x, mapping=dict()):
    return convert_node_status(x, mapping)

def convert_menapause(x, mapping=dict()):
    return convert_node_status(x, mapping)

def convert_study_id(x, mapping=dict()):
    return convert_surgery(x, mapping)

converters=dict(preTrt_lymph_node_status=convert_node_status,
               race=convert_race,
               menopausal_status=convert_menapause,
               surgery_type=convert_surgery,
               surgery=convert_surgery,
               study=convert_study_id)

xgboost_top_100 = ['C9',
 'OR12D3',
 'TSPAN32',
 'SGK2',
 'preTrt_numPosLymphNodes',
 'DDN',
 'GABRQ',
 'GZMM',
 'COL11A1',
 'IGFBP4',
 'EIF1AY',
 'KCNH4',
 'F2R',
 'PSMD5',
 'PYCARD',
 'POU3F2',
 'taxaneGeneral',
 'DUOX2',
 'CLINT1',
 'SAG',
 'KIF17',
 'PPEF1',
 'ITPKC',
 'CCR4',
 'DTX2',
 'MAP1S',
 'C20orf43',
 'TDP1',
 'PLXNA1',
 'METAP2',
 'IQCG',
 'DLX6',
 'DAO',
 'CCNE2',
 'TNNI1',
 'PHF15',
 'RACGAP1',
 'MYBPC2',
 'RHOBTB1',
 'CES2',
 'CARD9',
 'TCFL5',
 'VPS8',
 'PPP1R12A',
 'NHLH2',
 'SERPINB2',
 'NAB1',
 'GPM6B',
 'PRPSAP2',
 'GPI',
 'CRMP1',
 'RPL36',
 'KLRF1',
 'ETV7',
 'ARHGAP15',
 'ICT1',
 'GP1BB',
 'KEL',
 'MGAT4A',
 'CREB3',
 'OSBP2',
 'IRS4',
 'TTC31',
 'ING2',
 'PEBP1',
 'MDM1',
 'CBX8',
 'ZNF14',
 'OTOR',
 'ST6GALNAC4',
 'ZNF613',
 'FBXO7',
 'CTNNA1',
 'KRT15',
 'ARID4B',
 'CBX4',
 'SLC6A8',
 'BTG1',
 'VPS39',
 'FOS',
 'MAP3K7',
 'GLI3',
 'RAD50',
 'RBM23',
 'C7orf10',
 'C16orf58',
 'GALNT8',
 'POLR2C',
 'NRGN',
 'SLCO4A1',
 'NME3',
 'F2RL3',
 'LY6H',
 'IFIT1',
 'MYO1A',
 'AZIN1',
 'PFN2',
 'CWF19L1',
 'PTGES',
 'C9orf38']


pam50 = [x.strip() for x in """ACTR3B
ANLN
BAG1
BCL2
BIRC5
BLVRA
CCNB1
CCNE1
CDC20
CDC6
NUF2
CDH3
CENPF
CEP55
CXXC5
EGFR
ERBB2
ESR1
EXO1
FGFR4
FOXA1
FOXC1
GPR160
GRB7
KIF2C
NDC80
KRT14
KRT17
KRT5
MAPT
MDM2
MELK
MIA
MKI67
MLPH
MMP11
MYBL2
MYC
NAT1
ORC6
PGR
PHGDH
PTTG1
RRM2
SFRP1
SLC39A6
TMEM45B
TYMS
UBE2C
UBE2T""".split()]

treatment_columns = ['tumor_size_cm_preTrt_preSurgery',
                     'preTrt_lymph_node_status',
                     'preTrt_totalLymphNodes',
                     'preTrt_numPosLymphNodes',
                     'hist_grade',
                     'nuclear_grade_preTrt',
                     'age', 'race', 'menopausal_status', 'surgery_type', 'intarvenous', 'intramuscular', 'oral',
                     'radiotherapyClass', 'chemotherapyClass', 'hormone_therapyClass', 'postmenopausal_only',
                     'anthracycline', 'taxane', 'anti_estrogen', 'aromatase_inhibitor',
                     'estrogen_receptor_blocker', 'estrogen_receptor_blocker_and_stops_production', 'anti_HER2',
                     'tamoxifen', 'doxorubicin',
                     'epirubicin', 'docetaxel', 'capecitabine', 'fluorouracil',
                     'paclitaxel', 'cyclophosphamide', 'anastrozole',
                     'trastuzumab', 'letrozole', 'chemotherapy',
                     'no_treatment', 'methotrexate', 'other', 'taxaneGeneral']


def load_merged_dataset(cancer_data_dir):
    dump_dir = os.path.join(cancer_data_dir, 'bcDump/example15bmc')
    clinical_table_path = os.path.join(cancer_data_dir, 'bcClinicalTable.csv')
    merged_path = os.path.join(dump_dir, 'ex15bmcMerged.csv.xz')
    bmc_all_path = os.path.join(dump_dir, 'bmc15mldata1.csv')

    dtype = {'DFS': pandas.Int64Dtype(),
             'pCR': pandas.Int64Dtype(),
             'RFS': pandas.Int64Dtype(),
             'DFS': pandas.Int64Dtype(),
             'posOutcome': pandas.Int64Dtype()}


    # load averaged treatment table
    bmc = pandas.read_csv(bmc_all_path, dtype=dtype, converters=converters)
    bmc = bmc.sort_values(by='patient_ID')

    # load detailed treatment
    treatment = pandas.read_csv(clinical_table_path, converters=converters).sort_values(by='patient_ID')
    treatment = treatment[treatment.patient_ID.isin(bmc.patient_ID)]


    # load genes expression data
    gene_expression = pandas.read_csv(lzma.open(merged_path))

    genes_features = gene_expression[gene_expression.patient_ID.isin(bmc.patient_ID)]
    genes_features = genes_features.sort_values(by='patient_ID')

    genes_columns = genes_features.columns.to_list()[1:]

    ## merge genes expression + averaged treatment + detailed treatment
    merged = pandas.merge(genes_features, bmc, left_on='patient_ID', right_on='patient_ID')
    merged = pandas.merge(merged, treatment, left_on='patient_ID', right_on='patient_ID')
    merged.insert(0, 'row_num', range(0,len(merged)))

    result = dict(bmc=bmc, genes_features=genes_features,
            genes_columns=genes_columns, merged=merged,
            treatment=treatment)
    return result
