import subprocess
import argparse
import shlex
import pandas
from data_util.util import compute_metrics
from collections import defaultdict


def estimate(program_path, csv_path, target_field='posOutcome'):
    line = "aseval-table -i {0} -C {1} -u {2} --labels=1".format(csv_path,
            program_path, target_field)

    out = subprocess.run(shlex.split(line), stdout=subprocess.PIPE)
    # split by target variable
    splitted = [x for x in out.stdout.decode().split(target_field) if bool(x)]
    result = list()
    for line in splitted:
        estimated = [int(x) for x in line.split()]
        result.append(estimated)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate combo program on \
            train and validation tables')
    parser.add_argument('--train-path',
                        help="path to train csv file")
    parser.add_argument('--validation-path',
                        help="path to validation csv file")
    parser.add_argument('--program-file',
                        help="path to a file with combo program")

    args = parser.parse_args()
    return args


def main():
    dtype = {'posOutcome': pandas.Int64Dtype()}
    args = parse_args()
    val_path = args.validation_path
    train_path = args.train_path
    program_path = args.program_file
    # load averaged treatment table
    val_data = pandas.read_csv(val_path, dtype=dtype)
    train_data = pandas.read_csv(train_path, dtype=dtype)


    val_est_list = estimate(program_path, val_path)

    train_est_list = estimate(program_path, train_path)

    i = 0
    for val_est, train_est in zip(val_est_list, train_est_list):
        i += 1
        print('program #{0}'.format(i))
        result = defaultdict(list)
        compute_metrics(result, val_data.posOutcome.to_list(), val_est,
                train_data.posOutcome.to_list(), train_est)
        for k, v in result.items():
            print(k, v)

main()
