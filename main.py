import csv
import functools
import json
import logging
import os
import os.path
import pickle
import random
import re
import time
from contextlib import contextmanager
from multiprocessing.pool import Pool

import msgpack
import nltk.data
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords

from KaggleWord2VecUtility import KaggleWord2VecUtility

try:
    from tqdm import tqdm
except Exception:
    def tqdm(lst): return lst

WORKER_NUM_PROCESS = 6
SPECIAL_WORDS = ['<UNK>', '<EOS>', '<PAD>', '<GO>']

INFERENCE_MODE = os.getenv('INFERENCE_MODE', 'next')  # 'self' or 'next'
RNN_DIRECTION = os.getenv('RNN_DIRECTION', 'bi')  # 'uni' or 'bi'
CELL_TYPE = os.getenv('CELL_TYPE', 'gru')  # 'gru' or 'lstm'
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '256'))
TOTAL_EPOCH = int(os.getenv('TOTAL_EPOCH', '50'))
MAX_SENTENCE_LENGTH = int(os.getenv('MAX_SENTENCE_LENGTH', '30'))
RNN_SIZE = int(os.getenv('RNN_SIZE', '256'))
RNN_LAYERS = int(os.getenv('RNN_LAYERS', '3'))
EVAL_GAP = int(os.getenv('EVAL_GAP', '10'))

print('INFERENCE_MODE', INFERENCE_MODE)
print('RNN_DIRECTION', RNN_DIRECTION)
print('CELL_TYPE', CELL_TYPE)
print('BATCH_SIZE', BATCH_SIZE)
print('TOTAL_EPOCH', TOTAL_EPOCH)
print('MAX_SENTENCE_LENGTH', MAX_SENTENCE_LENGTH)
print('RNN_SIZE', RNN_SIZE)
print('RNN_LAYERS', RNN_LAYERS)
print('EVAL_GAP', EVAL_GAP)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


@contextmanager
def task_step(descrption):
    st_time = time.time()
    logging.info('Start [%s]', descrption)
    yield
    logging.info('Finish [%s] (takes %.2lfs)', descrption, time.time() - st_time)


def _read_corpus_work(tokenizer, doc):
    return KaggleWord2VecUtility.review_to_sentences(doc, tokenizer,
                                                     remove_stopwords=True)


def read_corpus(stage):
    '''
        Read train corpus
    '''
    if os.path.exists('cache/read_corpus_%s' % stage):
        with open('cache/read_corpus_%s' % stage, 'rb') as f:
            labels, docs = pickle.load(f)
        return labels, docs

    csv.field_size_limit(2 * 2 ** 20)
    train_word_vector = pd.read_csv('20NEWS_data/%s_v2.tsv' % stage, header=0, delimiter='\t')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    labels, docs = [], []

    labels = [int(label) for label in train_word_vector['class']]

    worker_pool = Pool(processes=WORKER_NUM_PROCESS)
    docs = worker_pool.map(
        functools.partial(_read_corpus_work, tokenizer),
        train_word_vector['news']
    )

    with open('cache/read_corpus_%s' % stage, 'wb') as f:
        pickle.dump((labels, docs), f)

    return labels, docs


def read_word2vec(filename='100features_20minwords_10context_len2alldata'):
    from gensim.models import Word2Vec  # Import here to reduce warnings in process pool

    w2v_model = Word2Vec.load(filename)
    vocab_words = SPECIAL_WORDS + list(w2v_model.wv.vocab.keys())

    return w2v_model, vocab_words


def _transform_corpus_work(vocab_words, doc):
    transformed_doc = []
    for sentence in doc:
        transformed_sentence = []
        for word in sentence:
            try:
                word_idx = vocab_words.index(word)
            except ValueError:
                word_idx = vocab_words.index('<UNK>')
            transformed_sentence.append(word_idx)
        transformed_doc.append(transformed_sentence)
    return transformed_doc


def transform_corpus(docs, vocab_words, stage):
    if os.path.exists('cache/transform_corpus_%s' % stage):
        with open('cache/transform_corpus_%s' % stage, 'rb') as f:
            transformed_docs = pickle.load(f)
        return transformed_docs

    worker_pool = Pool(processes=WORKER_NUM_PROCESS)
    transformed_docs = worker_pool.map(
        functools.partial(_transform_corpus_work, vocab_words),
        docs
    )

    with open('cache/transform_corpus_%s' % stage, 'wb') as f:
        pickle.dump(transformed_docs, f)

    return transformed_docs


def make_batch_corpus(docs):
    eos_idx = SPECIAL_WORDS.index('<EOS>')
    pad_idx = SPECIAL_WORDS.index('<PAD>')
    go_idx = SPECIAL_WORDS.index('<GO>')

    in_setences_by_docs = []
    in_sentences = []
    out_sentences = []
    out_sentences_len = []
    target_sentences = []
    for doc in tqdm(docs):
        in_setences_by_doc = []
        for i in range(len(doc)):
            in_sentence = doc[i][:MAX_SENTENCE_LENGTH]

            in_sentence_np = np.full(MAX_SENTENCE_LENGTH + 1, pad_idx)
            out_sentence_np = np.full(MAX_SENTENCE_LENGTH + 1, pad_idx)
            target_sentence_np = np.full(MAX_SENTENCE_LENGTH + 1, pad_idx)

            in_sentence_np[:len(in_sentence)] = in_sentence
            in_setences_by_doc.append(in_sentence_np)

            if INFERENCE_MODE == 'self':
                out_sentence = doc[i][:MAX_SENTENCE_LENGTH]
            elif INFERENCE_MODE == 'next':
                if i == len(doc) - 1:
                    continue
                out_sentence = doc[i + 1][:MAX_SENTENCE_LENGTH]
            else:
                raise ValueError('Unknown inference mode : %s' % INFERENCE_MODE)

            out_sentence_np[0] = go_idx
            out_sentence_np[1:len(out_sentence) + 1] = out_sentence

            target_sentence_np[:len(out_sentence)] = out_sentence
            target_sentence_np[len(out_sentence)] = eos_idx

            in_sentences.append(in_sentence_np)
            out_sentences.append(out_sentence_np)
            out_sentences_len.append(len(out_sentence) + 1)  # 1: <GO> or <EOS>
            target_sentences.append(target_sentence_np)

        in_setences_by_docs.append(np.array(in_setences_by_doc))

    train_corpus = {
        'doc': in_setences_by_docs,
        'in': np.array(in_sentences),
        'out': np.array(out_sentences),
        'out_len': np.array(out_sentences_len),
        'target': np.array(target_sentences)
    }

    return train_corpus


def build_graph(w2v_model, vocab_words):
    vocab_size = len(vocab_words)
    drop_keep_prob = 0.8
    learning_rate = 0.001

    # Shape: [batch_size, max_length]
    enc_input = tf.placeholder(tf.int64, [None, None], name='X')

    # Shape: [batch_size, max_length]
    dec_input = tf.placeholder(tf.int64, [None, None])
    dec_length_input = tf.placeholder(tf.int64, [None])

    # Shape: [batch_size, max_length]
    targets = tf.placeholder(tf.int64, [None, None])

    placeholders = {
        'enc_input': enc_input,
        'dec_input': dec_input,
        'dec_length_input': dec_length_input,
        'targets': targets
    }

    # TODO initialize embedding with w2v_model
    with tf.variable_scope('embed'):
        embedding = tf.get_variable('embedding', [vocab_size, RNN_SIZE], tf.float32)
        enc_embed = tf.nn.embedding_lookup(embedding, enc_input)
        dec_embed = tf.nn.embedding_lookup(embedding, dec_input)

    with tf.variable_scope('encoder'):
        def _make_cell():
            enc_cell_list = []
            for i in range(RNN_LAYERS):
                if CELL_TYPE == 'gru':
                    enc_cell = tf.contrib.rnn.GRUCell(RNN_SIZE)
                elif CELL_TYPE == 'lstm':
                    enc_cell = tf.contrib.rnn.BasicLSTMCell(RNN_SIZE)

                enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=drop_keep_prob)
                enc_cell_list.append(enc_cell)

            return tf.contrib.rnn.MultiRNNCell(enc_cell_list)

        if RNN_DIRECTION == 'uni':
            enc_cell = _make_cell()
            enc_outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_embed,
                                                        dtype=tf.float32)
        elif RNN_DIRECTION == 'bi':
            fw_cell = _make_cell()
            bw_cell = _make_cell()

            enc_outputs, (enc_fw_states, enc_bw_states) = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, enc_embed,
                dtype=tf.float32)
            enc_outputs = tf.concat(enc_outputs, -1)

        enc_states_ident = tf.identity(enc_outputs, name='sentence_vector')

    with tf.variable_scope('decoder'):
        def _make_cell():
            dec_cell_list = []
            for i in range(RNN_LAYERS):
                if CELL_TYPE == 'gru':
                    dec_cell = tf.contrib.rnn.GRUCell(RNN_SIZE)
                elif CELL_TYPE == 'lstm':
                    dec_cell = tf.contrib.rnn.BasicLSTMCell(RNN_SIZE)

                dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=drop_keep_prob)

                dec_cell_list.append(dec_cell)

            return tf.contrib.rnn.MultiRNNCell(dec_cell_list)

        if RNN_DIRECTION == 'uni':
            dec_cell = _make_cell()
            dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_embed,
                                                        initial_state=enc_states,
                                                        dtype=tf.float32)
        else:
            fw_cell = _make_cell()
            bw_cell = _make_cell()

            dec_outputs, (dec_fw_states, dec_bw_states) = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, dec_embed,
                initial_state_fw=enc_fw_states,
                initial_state_bw=enc_bw_states,
                dtype=tf.float32)
            dec_outputs = tf.concat(dec_outputs, -1)

    model = tf.layers.dense(dec_outputs, vocab_size, activation=None)

    mask = tf.sequence_mask(dec_length_input, MAX_SENTENCE_LENGTH + 1, dtype=tf.float32)
    cross_entropy = tf.contrib.seq2seq.sequence_loss(model, targets, mask)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    return model, cross_entropy, optimizer, placeholders


def train_model(vocab_words, train_corpus, test_corpus, model, cross_entropy, optimizer, placeholders):
    logging.info('# of samples: %d', train_corpus['in'].shape[0])

    def _get_batches():
        vocab_size = len(vocab_words)

        shuffled_indexes = list(range(train_corpus['in'].shape[0]))
        random.shuffle(shuffled_indexes)

        for batch_start_idx in range(0, len(shuffled_indexes), BATCH_SIZE):
            batch_indexes = shuffled_indexes[batch_start_idx:batch_start_idx + BATCH_SIZE]

            batch_enc_input = train_corpus['in'][batch_indexes]
            batch_enc_input = batch_enc_input[:, ::-1]

            batch_dec_input = train_corpus['out'][batch_indexes]

            batch_dec_length_input = train_corpus['out_len'][batch_indexes]

            batch_targets_input = train_corpus['target'][batch_indexes]

            yield batch_start_idx, batch_enc_input, batch_dec_input, batch_dec_length_input, batch_targets_input

    model_name = '%s_cell_%s_dir_%s_bat_%d_maxlen_%d_unit_%d_layer_%d' % (
        INFERENCE_MODE, CELL_TYPE, RNN_DIRECTION,
        BATCH_SIZE, MAX_SENTENCE_LENGTH, RNN_SIZE, RNN_LAYERS)

    model_saver = tf.train.Saver(max_to_keep=None)
    model_save_dir = 'cache/%s' % model_name

    try:
        os.mkdir(model_save_dir)
    except FileExistsError:
        logging.info('Found pre-trained model directory')

    try:
        os.mkdir('outputs')
    except FileExistsError:
        logging.info('Found output directory')

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        epoch_start = 0
        for epoch in reversed(range(TOTAL_EPOCH)):
            if os.path.exists('%s/epoch_%d.ckpt.index' % (model_save_dir, epoch)):
                epoch_start = epoch + 1
                break

        if 0 < epoch_start:
            save_path = '%s/epoch_%d.ckpt' % (model_save_dir, epoch_start - 1)
            model_saver.restore(session, save_path)

        for epoch in range(epoch_start, TOTAL_EPOCH):
            for batch_start_idx, batch_enc_input, batch_dec_input, batch_dec_length_input, batch_targets in _get_batches():
                _, loss = session.run(
                    [optimizer, cross_entropy],
                    feed_dict={
                        placeholders['enc_input']: batch_enc_input,
                        placeholders['dec_input']: batch_dec_input,
                        placeholders['dec_length_input']: batch_dec_length_input,
                        placeholders['targets']: batch_targets
                    }
                )

                logging.info('Epoch: %04d, Batch: %06d, loss: %.6f', epoch + 1, batch_start_idx, loss)

            save_path = model_saver.save(session, '%s/epoch_%d.ckpt' % (model_save_dir, epoch))
            logging.info('Model is saved to %s', save_path)

            if (epoch + 1) % EVAL_GAP == 0:
                logging.info('Store document vectors')

                graph = tf.get_default_graph()
                X = graph.get_tensor_by_name('X:0')
                sentence_vector = graph.get_tensor_by_name('encoder/sentence_vector:0')

                train_results, test_results = [], []
                for corpus, results in [[train_corpus['doc'], train_results],
                                        [test_corpus['doc'], test_results]]:
                    for doc in tqdm(corpus):
                        enc_sentence = session.run(sentence_vector, feed_dict={X: doc[:, ::-1]})
                        enc_sentence = enc_sentence[:, -1, :]
                        doc_vec = np.mean(enc_sentence, axis=0)
                        results.append(doc_vec.tolist())

                output_name = '%s_epoch_%d' % (model_name, epoch + 1)
                curr_ts = time.time()

                with open('outputs/%s_%d_train.msgpack' % (output_name, curr_ts), 'wb') as f:
                    msgpack.dump(train_results, f)

                with open('outputs/%s_%d_test.msgpack' % (output_name, curr_ts), 'wb') as f:
                    msgpack.dump(test_results, f)

    return save_path


def main():
    try:
        os.mkdir('cache')
        logging.info('Make new `cache` directory')
    except FileExistsError:
        logging.info('Use `cache` directory')

    with task_step('Read corpus'):
        train_labels, train_docs = read_corpus('train')
        test_labels, test_docs = read_corpus('test')

    with task_step('Read pretrained word2vec model'):
        w2v_model, vocab_words = read_word2vec()

    with task_step('Transform train/test corpus to be index-based'):
        train_transformed_docs = transform_corpus(train_docs, vocab_words, 'train')
        test_transformed_docs = transform_corpus(test_docs, vocab_words, 'test')

    with task_step('Make train/test corpus to batch-friendly'):
        train_corpus = make_batch_corpus(train_transformed_docs)
        test_corpus = make_batch_corpus(test_transformed_docs)

    with task_step('Build RNN graph'):
        model, cross_entropy, optimizer, placeholders = build_graph(w2v_model, vocab_words)

    with task_step('Train RNN model'):
        model_path = train_model(vocab_words, train_corpus, test_corpus,
                                 model, cross_entropy, optimizer, placeholders)


if __name__ == '__main__':
    main()
