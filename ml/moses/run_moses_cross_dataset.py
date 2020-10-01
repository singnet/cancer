import re
from numpy import array
import numpy
from collections import defaultdict
import time
import subprocess
import os.path
import shlex
import argparse
import concurrent.futures
from eval_moses import evaluate_moses


s = "asmoses -i {train_path} --log-file {log_path} --hc-fraction-of-nn 0.01 -j10 --balance 1 --result-count 100 --reduct-knob-building-effort=1 --hc-crossover-min-neighbors=500 --fs-focus=all --fs-seed=init -m {max_eval} --hc-max-nn-evals=100000 -l debug -q 0.05 -u {target} --output-with-labels=1 --logical-perm-ratio=-0.95 --complexity-ratio=1 --max-time={max_time}"


def get_train_val_path(data_dir, study):
    val_path = None
    train_path = None
    for data_f in os.listdir(data_dir):
        if '_val_' in data_f and study in data_f:
            val_path = os.path.join(data_dir, data_f)
        if '_train_' in data_f and study in data_f:
            train_path = os.path.join(data_dir, data_f)
    return train_path, val_path


def get_results(logs_dir, data_dir, target):
    accuracy = []
    i = 0
    for f in os.listdir(logs_dir):
        m = re.match('log_(.+?).out', f)
        if m:
            i += 1
            print('{0} study'.format(i))
            out_path = os.path.join(logs_dir, f)
            study = m.group(1)
            train_path, val_path = get_train_val_path(data_dir, study)
            study_res = get_best_result(evaluate_moses(train_path, val_path, out_path, target=target))
            if len(study_res):
                print(study)
            for k, value in study_res.items():
                print(k, value)
                if k == 'accuracy':
                    accuracy.append(value)
            print('\n')
    print('mean accuracy {0}'.format(str(numpy.mean(accuracy))))


def get_best_result(results):
    pairs = sorted([(value['accuracy'], key) for (key, value) in results.items()])
    if pairs:
        return results[pairs[-1][1]]
    return dict()


def process_results(future, study, logs_dir):
    proc: subprocess.CompletedProcess = future.result()
    if proc.returncode != 0:
        print("process failed with exit code {0}".format(proc.returncode))
    print("done study {0}".format(study))
    with open(os.path.join(logs_dir, 'log_{0}.out'.format(study)), 'wt') as f:
        # stderr was captured in stdout
        f.write(proc.stdout)


def parse_args():
    parser = argparse.ArgumentParser(description='run asmoses on datasets')
    parser.add_argument('--data-dir', default='',
                        help="path to a files datasets")
    parser.add_argument('--logs-dir', default='',
                        help='path to store logs')
    parser.add_argument('--target', default='',
                        help='target variable name')
    parser.add_argument('--max-evaluations', default=60000, type=int,
                        help="max evaluations")
    parser.add_argument('--max-time', default=56800, type=int,
                        help="max running time in seconds")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir = os.path.expanduser(args.data_dir)
    logs_dir = os.path.expanduser(args.logs_dir)
    max_time = args.max_time
    max_eval = args.max_evaluations
    target = args.target
    if not target:
        print('target must be provided')
        return

    train_val_path = []
    for f in os.listdir(data_dir):
        m = re.match("(.+?)_val_leave_study_(.+?)\.csv", f)
        if m:
            study = m.group(2)
            train_f = m.group(1) + '_train_leave_study_' + study + '.csv'
            train_val_path.append((study,
                os.path.join(data_dir, train_f),
                os.path.join(data_dir, f)))
    executor = concurrent.futures.ThreadPoolExecutor()
    futures = []
    future_to_study = dict()
    while (train_val_path):
        if len(futures) <= 5:
            study, train_path, val_path = train_val_path.pop()
            log_path = os.path.join(logs_dir, 'log_{0}.txt.log'.format(study))
            if os.path.exists(log_path):
                print('not running study {0} - log exists'.format(study))
            s_tmp = s.format(train_path=train_path, log_path=log_path, target=target,
                    max_time=str(max_time), max_eval=str(max_eval))
            future = executor.submit(subprocess.run, shlex.split(s_tmp),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding='utf-8')
            futures.append(future)
            future_to_study[future] = study
            print('started study {0}'.format(study))
        time.sleep(10)
        done = [x for x in futures if x.done()]
        futures = [x for x in futures if x not in done]
        for f in done:
            process_results(f, future_to_study[f], logs_dir)
    concurrent.futures.wait(futures)
    for f in futures:
        process_results(f, future_to_study[f], logs_dir)

    get_results(logs_dir, data_dir, target)

if __name__ == '__main__':
    #get_results('./logs.dfs', '/home/noskill/projects/cancer.old/bin/leave_one.DFS', 'DFS')
    main()
