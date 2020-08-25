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
from sklearn.metrics import accuracy_score


def split_by_study(merged, feature_columns, label_columns, study=None,
                   to_numpy=True):
    """
    Split one study out for cross-validation

    merged: pandas.DataFrame
        Genes + treatments table
    feature_columns: List[str]
        list of column names to extract from merged dataframe for use as input features
        in train and validation sets
    label_columns: List[str]
        list of column names to extract from merged dataframe for use as labels
        in train and validation sets
    study: str
        Optional param - process only singe study
    to_numpy: bool
        convert to numpy if True

    """
    for eval_study in set(merged.study):
        if study is not None:
            eval_study = study
        merged_train = merged[merged.study != eval_study]
        merged_val = merged[merged.study == eval_study]
        assert (not set(merged_train.patient_ID).intersection(set(merged_val.patient_ID)))

        train_split = merged[merged.patient_ID.isin(merged_train.patient_ID)]
        val_split = merged[merged.patient_ID.isin(merged_val.patient_ID)]
        assert val_split.patient_ID.to_list() == merged_val.patient_ID.to_list()
        if to_numpy:
            train_data = train_split[feature_columns].to_numpy()
            train_labels = train_split[label_columns].to_numpy().astype(int)
            val_data = val_split[feature_columns].to_numpy()
            val_labels = val_split[label_columns].to_numpy().astype(int)
            yield train_data, train_labels, val_data, val_labels
        else:
            train_data = train_split[feature_columns]
            train_labels = train_split[label_columns]
            val_data = val_split[feature_columns]
            val_labels = val_split[label_columns]
            yield train_data, train_labels, val_data, val_labels

        if study is not None:
            break


def select_random_idx(study, num):
    if not num % 2 == 0:
        num = num + 1
    validation = []
    idx = numpy.arange(len(study))
    random.shuffle(idx)
    validation = study.iloc[idx[:num]]
    train = study.iloc[idx[num:]]
    return train, validation


def select_balanced_idx(study, num, balance_train=False):
    """
    Split study to train and validation sets.
    For validation set sample unique rows balanced by posOutcome.

    Parameters:
    study: pandas.DataFrame
        study data
    num: int
        number of samples to draw
    balance_train: bool
        balance train set if true by upsampling
    """
    if not num % 2 == 0:
        num = num + 1
    validation = []
    outcomes = study.posOutcome.unique()
    if len(outcomes) == 1:
        print("can't balance by posOutcome: theres only one outcome for study {0}".format(study.study.iloc[0]))
        return study, study.iloc[0:0]
    pos = max(outcomes)
    neg = min(outcomes)
    pos_outcome = study[study.posOutcome == pos].patient_ID
    neg_outcome = study[study.posOutcome == neg].patient_ID
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
    other = study[(~study.patient_ID.isin(validation)) & (~study.patient_ID.isin(train.patient_ID))]
    train = study.loc[train_idx + other.index.to_list()]

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


def random_split(merged, feature_columns, label_columns, ratio=0.1, study_name=None, rand=False, to_numpy=True, balance_by_study=False, balance_train=False, balance_validation=True):
    """
    Split dataset into train and validation sets, that is
        two pairs of (features, labels)

    Parameters:
    merged: pandas.DataFrame
        frame #patients by #features
    feature_columns: List[str]
        list of column names to extract from merged dataframe for use as input features
        in train and validation sets
    label_columns: List[str]
        list of column names to extract from merged dataframe for use as labels
        in train and validation sets
    balance_train: bool
        balance train set by posOutcome
    balance_by_study: bool
        resample smaller studies for train data. As the result
        train data will contain equal number of rows for each study.
    to_numpy: bool
        convert to numpy.array if true
    --------------
    Returns: train_data, train_labels, val_data, val_labels
    """
    val_dict = defaultdict(list)
    train_dict = defaultdict(list)

    bmc = merged
    for eval_study in set(bmc.study):
        if study_name is not None:
            if study_name != eval_study:
                continue
        study = bmc[bmc.study == eval_study]
        num_select = math.ceil(len(study) * ratio)
        study_patients = bmc[bmc.study == eval_study]
        # balance validation set and possibly train set
        if balance_validation:
            bmc_train, bmc_val = select_balanced_idx(study_patients, num_select, balance_train)
        else:
            bmc_train, bmc_val = select_random_idx(study_patients, num_select)
        val_dict[eval_study] = bmc_val.index.to_list()
        train_dict[eval_study] = bmc_train.index.to_list()
    if balance_by_study:
        train_dict = resample_patients_by_study(train_dict)
        loc = []
        for patient_lst in train_dict.values():
            loc += patient_lst
        train_split = merged.loc[loc]
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
    return train_data, train_labels, val_data, val_labels


def digitize_genes(col, bins=15):
    _, edges = numpy.histogram(col, bins=bins, density=True)
    res = numpy.digitize(col, edges)
    return res - res.min()


def digitize_genes_by_median(col):
    return col > numpy.median(col)


res_ar = numpy.zeros((16, 4), dtype=numpy.bool_)
for i in range(16):
    ar0 = bin(i).split('0b')[1]
    ar0 = '0' * (4 - len(ar0)) + ar0
    res_ar[i] = [int(x) for x in ar0]


def binary_genes(merged, genes_columns):
    # raw improves the performance
    # multiplication by one convertes bool to int
    merged[genes_columns] = merged[genes_columns].apply(digitize_genes_by_median, raw=True) * 1
    return merged


def digitize_non_genes_data(col):
    not_nan = numpy.nan_to_num(col, nan=-1)
    n_unique = len(set(not_nan))
    if 15 < n_unique:
        n_unique = 15
    edge = numpy.histogram_bin_edges(numpy.nan_to_num(col, nan=col.min()), bins=n_unique - 1)
    digits = numpy.digitize(not_nan, edge)
    digits = digits - digits.min()
    return digits


def categorical_features(frame, feature_columns, dtype=numpy.float16, to_letters=False):
    """
    Replace features in frame by historgram bin id

    Parameters
    ---------------
    frame: pandas.DataFrame
    feature_columns: List[str]
    dtype: object
        type to use for categorical data
    to_letters: bool
        use ascii latters if true for categorical data
    """
    for col_name in feature_columns:
        tmp_dtype = dtype
        if len(frame[col_name].unique()) == 1:
            print('dropping column {0}: it has only one value'.format(col_name))
            frame = frame.drop(columns=[col_name])
            continue
        dig = digitize_non_genes_data(frame[col_name])
        if to_letters:
            tmp_dtype=object
        if len(set(dig)) <= 2:
            # it can be represented by bool
            tmp_dtype = numpy.uint8
        ar1 = numpy.zeros(len(dig), dtype=tmp_dtype)
        for i, d in enumerate(dig):
            if to_letters and tmp_dtype == object:
                ar1[i] = string.ascii_letters[d]
            else:
                ar1[i] = tmp_dtype(d)
        frame[col_name] = ar1
    return frame


def binarize_dataset(merged, genes_columns, feature_colum, to_letters):
    """
    merged: pandas.DataFrame
    genes_columns: List[str]
    feature_colum: List[str]
    to_letters: bool
        convert categorical variables to ascii letters
        if false numpy.float16 is used
    """
    copy = merged.copy()
    to_bin_columns = [x for x in feature_colum if x not in genes_columns]
    copy = categorical_features(copy, to_bin_columns, to_letters=False)
    binary_genes(copy, genes_columns)
    return copy


def compute_metrics(result, y_true, y_pred, x_true, x_pred, multilabel=False):
    if multilabel:
        result['train_accuracy'].append(accuracy_score(x_true, x_pred))
        result['accuracy'].append(accuracy_score(y_true, y_pred))
        return
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
def convert_surgery(x, surgery_mapping=dict(NA=0, NaN=0)):
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

study_mapping = dict()
def convert_study_id(x, mapping=study_mapping):
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

treatment_columns_bmc = ['radio', 'surgery', 'chemo', 'hormone']

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


dtype = {'DFS': pandas.Int64Dtype(),
         'pCR': pandas.Int64Dtype(),
         'RFS': pandas.Int64Dtype(),
         'DFS': pandas.Int64Dtype(),
         'posOutcome': pandas.Int64Dtype()}


def load_merged_dataset(cancer_data_dir):
    dump_dir = os.path.join(cancer_data_dir, 'bcDump/example15bmc')
    clinical_table_path = os.path.join(cancer_data_dir, 'bcClinicalTable.csv')
    merged_path = os.path.join(dump_dir, 'ex15bmcMerged.csv.xz')
    bmc_all_path = os.path.join(dump_dir, 'bmc15mldata1.csv')

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
    to_drop = []
    for col in bmc.columns:
        if col in treatment.columns:
            if col == 'patient_ID':
                continue
            to_drop.append(col)
    treatment = treatment.drop(columns=to_drop)
    merged = pandas.merge(merged, treatment, left_on='patient_ID', right_on='patient_ID')
    merged.insert(0, 'row_num', range(0,len(merged)))

    result = dict(bmc=bmc, genes_features=genes_features,
            genes_columns=genes_columns, merged=merged,
            treatment=treatment)
    return result


def zero_to_minus_one(frame, colname):
    """
    Replace zero with minus one in the given DataFrame and column name
    """
    assert len(frame[colname].unique()) == 2
    frame.loc[frame[colname] == 0, colname] = -1


def merge_metagx_curated(metagx_merged, mergedCurated):
    """
    Merge curatedBreastCancer and metagx dataframes
    resulting dataset has -1 for false 1 for true and 0 for 'unknown'
    """
    # there two surgery types, merge them
    mergedCurated.surgery = (mergedCurated.surgery > 0) * 1
    mergedCurated.loc[mergedCurated.posOutcome == 0, 'posOutcome'] = -1
    zero_to_minus_one(mergedCurated, 'surgery')
    zero_to_minus_one(mergedCurated, 'chemo')
    zero_to_minus_one(mergedCurated, 'hormone')
    zero_to_minus_one(mergedCurated, 'radio')
    mergedCurated.study += (max(metagx_merged.study.unique()) + 1)

    # 0 is undefined in metagx
    mergedCurated.age += 1
    mergedCurated.insert(2, 'tumor_size', mergedCurated.tumor_size_cm_preTrt_preSurgery.fillna(-1) + 1)
    metagx_merged.insert(2, 'radio', numpy.zeros(len(metagx_merged)))
    metagx_merged.insert(2, 'surgery', numpy.zeros(len(metagx_merged)))
    metagx_merged = metagx_merged.drop(columns=['row_num'])
    mergedCurated = mergedCurated.drop(columns=['row_num'])
    columns_common = mergedCurated.columns[mergedCurated.columns.isin(metagx_merged.columns)]
    merged_common = pandas.concat([mergedCurated[columns_common], metagx_merged[columns_common]])
    merged_common.insert(0, 'row_num', numpy.arange(len(merged_common)))
    merged_common.reset_index(inplace=True)
    return merged_common
