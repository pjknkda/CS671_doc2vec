import csv
import functools
import json
import os
import random
import re
import sys
import time
from multiprocessing.pool import Pool

import msgpack
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.svm import SVC, LinearSVC

WORKER_NUM_PROCESS = os.cpu_count()


def do_classification(train_docs, train_labels, test_docs, test_labels):
    train_docs = np.array(train_docs)
    train_labels = np.array(train_labels)

    test_docs = np.array(test_docs)
    test_labels = np.array(test_labels)

    classifier = GradientBoostingClassifier(verbose=2, n_estimators=300)
    classifier.fit(train_docs, train_labels)

    reports = []
    best_report_f, best_report = -1, None
    for y_pred in classifier.staged_predict(test_docs):
        accuracy = accuracy_score(test_labels, y_pred)
        precision, recall, f_measure, _ = precision_recall_fscore_support(test_labels, y_pred, average='weighted')
        report = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f_measure': f_measure
        }
        reports.append(report)

        if best_report_f < f_measure:
            best_report_f, best_report = f_measure, report
    print(best_report)
    return best_report


def handle_one_exp(train_labels, test_labels, f_name):
    print('Handle', f_name)

    if os.path.exists('outputs/%s_result.json' % f_name):
        with open('outputs/%s_result.json' % f_name, 'r') as f:
            return (f_name, json.load(f))
    else:
        with open('outputs/%s_train.msgpack' % f_name, 'rb') as f:
            train_docs = msgpack.load(f)

        with open('outputs/%s_test.msgpack' % f_name, 'rb') as f:
            test_docs = msgpack.load(f)

        report = do_classification(train_docs, train_labels, test_docs, test_labels)

        with open('outputs/%s_result.json' % f_name, 'w') as f:
            json.dump(report, f)

        return f_name, report


def format_result_to_csv(filepath):
    rows = []
    with open(filepath, 'r') as f:
        results = json.load(f)
        for filename, report in results.items():
            m = re.match('(?P<inf_mode>.*)'
                         '_cell_(?P<cell_type>.*)'
                         '_dir_(?P<rnn_direction>.*)'
                         '_bat_(?P<batch_size>\d+)'
                         '_maxlen_(?P<max_length>\d+)'
                         '_unit_(?P<num_unit>\d+)'
                         '_layer_(?P<num_layer>\d+)'
                         '_epoch_(?P<epoch>\d+)'
                         '_(?:\d+)', filename)

            row = {}
            row.update(m.groupdict())
            row.update(report)
            rows.append(row)

    pd.DataFrame(rows,
                 columns=['inf_mode', 'cell_type', 'rnn_direction',
                          'batch_size', 'sentence_length',
                          'num_unit', 'num_layer', 'epoch',
                          'accuracy', 'precision', 'recall', 'f_measure']).to_csv(filepath + '.csv')


if __name__ == '__main__':
    csv.field_size_limit(2 * 2 ** 20)
    train = pd.read_csv('20NEWS_data/train_v2.tsv', header=0, delimiter='\t')
    test = pd.read_csv('20NEWS_data/test_v2.tsv', header=0, delimiter='\t')
    all_doc = pd.read_csv('20NEWS_data/all_v2.tsv', header=0, delimiter='\t')

    train_labels = [int(i) for i in list(train['class'])]
    test_labels = [int(i) for i in list(test['class'])]

    files = [f[:-len('_train.msgpack')] for f in os.listdir('outputs/') if f.endswith('_train.msgpack')]

    worker_pool = Pool(processes=WORKER_NUM_PROCESS)
    reports = worker_pool.map(
        functools.partial(handle_one_exp, train_labels, test_labels),
        files
    )

    results = {filename: report for filename, report in reports}

    result_filename = 'results/%d.json' % time.time()
    with open(result_filename, 'w') as f:
        json.dump(results, f, indent=2)
    format_result_to_csv(result_filename)
