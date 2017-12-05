import csv
import functools
import json
import os
import random
import time
from multiprocessing.pool import Pool

import msgpack
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC, LinearSVC

WORKER_NUM_PROCESS = 6


class CLF:
    def __init__(self):
        self.model = None
        self.score = 'accuracy'
        #self.param_grid = [{'C': np.arange(0.1, 7, 0.2)}]
        self.param_grid = [{'C': np.array([1.0])}]

    def get_doc_vec(self, sentences):
        return np.mean(np.array(sentences), axis=0)

    def train(self, docs, labels, score):
        print('start train')
        self.score = score
        doc_vecs = np.array(docs)

        #self.model = GridSearchCV(LinearSVC(C = 1), self.param_grid, n_jobs = 20, cv = 2, scoring = '%s' % self.score, verbose = 1)
        # self.model = RandomForestClassifier(max_depth=None, random_state=0, n_estimators = 200)
        self.model = GradientBoostingClassifier(verbose=2)
        #self.model = KNeighborsClassifier(3, n_jobs = 20)
        #self.model = MLPClassifier(alpha = 1, hidden_layer_sizes = [256, 128, 128, 128, 128], verbose = True, max_iter = 100, early_stopping = False, tol = 0.0)

        self.model.fit(doc_vecs, labels)

    def predict(self, docs, true_labels):
        if self.model == None:
            return [], -1

        Y_true, Y_pred = true_labels, self.model.predict(docs)
        report = classification_report(Y_true, Y_pred, digits=6)
        print(report)
        score = self.model.score(docs, true_labels)
        return Y_pred, score, report

    def do_exp(self, train_docs, train_labels, test_docs, test_labels, score):
        '''
        train_doc_vecs = np.array([self.get_doc_vec(doc) for doc in train_docs])
        test_doc_vecs = np.array([self.get_doc_vec(doc) for doc in test_docs])
        '''
        train_doc_vecs = train_docs
        test_doc_vecs = test_docs

        self.train(train_doc_vecs, train_labels, score)
        return self.predict(test_doc_vecs, test_labels)


def handle_one_exp(train_labels, test_labels, f_name):
    print('Handle', f_name)

    def parse(s):
        s = s.split('\n')
        idx = [0, 1, 22]
        r = [s[i][14:].split() for i in range(len(s)) if i not in idx]
        return r

    with open('outputs/' + f_name + '_train.msgpack', 'rb') as f:
        train_docs = msgpack.load(f)

    with open('outputs/' + f_name + '_test.msgpack', 'rb') as f:
        test_docs = msgpack.load(f)

    clf = CLF()
    report = clf.do_exp(train_docs, train_labels, test_docs, test_labels, 'accuracy')

    return (f_name, parse(report[2]))


if __name__ == '__main__':
    csv.field_size_limit(2 * 2 ** 20)
    train = pd.read_csv('20NEWS_data/train_v2.tsv', header=0, delimiter='\t')
    test = pd.read_csv('20NEWS_data/test_v2.tsv', header=0, delimiter='\t')
    all_doc = pd.read_csv('20NEWS_data/all_v2.tsv', header=0, delimiter='\t')

    train_labels = [int(i) for i in list(train['class'])]
    test_labels = [int(i) for i in list(test['class'])]

    files = [f[:-14] for f in os.listdir('outputs/') if f.endswith('train.msgpack')]

    worker_pool = Pool(processes=WORKER_NUM_PROCESS)
    reports = worker_pool.map(
        functools.partial(handle_one_exp, train_labels, test_labels),
        files
    )

    results = {kv[0]: kv[1] for kv in reports}

    with open('results/%d.json' % time.time(), 'w') as f:
        json.dump(results, f, indent=2)
