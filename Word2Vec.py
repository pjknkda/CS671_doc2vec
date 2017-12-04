#!/usr/bin/env python
import csv
import logging
import sys
import time

import nltk.data
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

from KaggleWord2VecUtility import KaggleWord2VecUtility

if __name__ == '__main__':

    start = time.time()
    # The csv file might contain very huge fields, therefore set the field_size_limit to maximum.
    csv.field_size_limit(2 * 2 ** 20)
    # Read train data.
    train_word_vector = pd.read_csv('20NEWS_data/all_v2.tsv', header=0, delimiter="\t")
    # Use the NLTK tokenizer to split the paragraph into sentences.
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []
    print("Parsing sentences from training set...")

    # Loop over each news article.
    for review in train_word_vector["news"]:
        try:
            # Split a review into parsed sentences.
            sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        except Exception:
            continue

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    num_features = int(sys.argv[1])     # Word vector dimensionality
    min_word_count = 20   # Minimum word count
    num_workers = 40       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    print("Training Word2Vec model...")
    # Train Word2Vec model.
    model = Word2Vec(sentences, workers=num_workers, hs=0, sg=1, negative=10, iter=25,
                     size=num_features, min_count=min_word_count,
                     window=context, sample=downsampling, seed=1)

    model_name = str(num_features) + "features_" + str(min_word_count) + \
        "minwords_" + str(context) + "context_len2alldata"
    model.init_sims(replace=True)
    # Save Word2Vec model.
    print("Saving Word2Vec model...")
    model.save(model_name)
    endmodeltime = time.time()

    print("time : ", endmodeltime - start)
