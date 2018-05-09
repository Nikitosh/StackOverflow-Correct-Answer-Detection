import datetime
import logging
import re
import time

import gensim
import os.path

import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from keras.layers import Bidirectional, LSTM

from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, \
    precision_recall_curve, average_precision_score

stops = set(stopwords.words('english'))

IGNORED_TAGS = ['del', 'strike', 's']
CODE_TAG = 'code'


def lower_text(text):
    text = text.lower()
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = gensim.parsing.preprocessing.strip_punctuation2(text)
    text = gensim.parsing.preprocessing.strip_numeric(text)
    text = gensim.parsing.preprocessing.strip_multiple_whitespaces(text)
    text = text.strip()
    return text


def transform_text(text):
    text = lower_text(text)
    filtered_words = [word for word in text.split() if word not in stops and len(word) >= 3]
    return ' '.join(filtered_words)


def stem_text(text):
    return gensim.parsing.preprocessing.stem_text(transform_text(text))


def draw_roc_curve(y_tests, y_preds):
    fpr, tpr, _ = roc_curve(y_tests, y_preds)
    logging.info('AUC: {}'.format(auc(fpr, tpr)))

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    plt.savefig('plots/roc.png')


def draw_precision_recall_curve(y_tests, y_preds):
    precision, recall, _ = precision_recall_curve(y_tests, y_preds)
    average_precision = average_precision_score(y_tests, y_preds)
    logging.info('Average precision-recall score: {0:0.2f}'.format(average_precision))

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig('plots/precision_recall.png')


def print_metrics(y_tests, y_preds):
    y_preds_binary = np.round(y_preds)
    logging.info('Accuracy: {}'.format(accuracy_score(y_tests, y_preds_binary)))
    logging.info('Precision: {}'.format(precision_score(y_tests, y_preds_binary)))
    logging.info('Recall: {}'.format(recall_score(y_tests, y_preds_binary)))
    logging.info('F1: {}'.format(f1_score(y_tests, y_preds_binary)))
    draw_precision_recall_curve(y_tests, y_preds)
    draw_roc_curve(y_tests, y_preds)


def generate_unit_vector(dim):
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)


def transform_sentence_batch_to_vector(word_vectors, sentences, document_max_num_words, num_features):
    X = np.zeros((len(sentences), document_max_num_words, num_features))
    for i in range(len(sentences)):
        words = sentences[i].split()
        for j, word in enumerate(words):
            if j == document_max_num_words:
                break
            if word in word_vectors:
                X[i, j, :] = word_vectors[word]
            else:
                X[i, j, :] = generate_unit_vector(num_features)
    return X


def get_word2vec_model_path(csv_file_name):
    file_name = os.path.splitext(os.path.basename(csv_file_name))[0]
    if file_name.find('_') != -1:
        file_name = file_name[file_name.find('_') + 1:]
    return 'word2vec/models/{}_model.bin'.format(file_name)


def add_lstm_to_model(model, input_shape, embed_size, bidirectional, dropout, return_sequences=False):
    if bidirectional:
        model.add(Bidirectional(LSTM(embed_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences), input_shape=input_shape))
    else:
        model.add(LSTM(embed_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences, input_shape=input_shape))


def string_to_timestamp(date):
    return int(time.mktime(datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f').timetuple()))


def process_code(code):
    return 'CODELEXEM'


def get_child_text(node):
    name = getattr(node, 'name', None)
    if name in IGNORED_TAGS:
        return ''
    if name == CODE_TAG:
        return process_code(get_node_text(node))
    return get_node_text(node)


def get_node_text(node):
    if 'childGenerator' in dir(node):
        return ' '.join([get_child_text(child) for child in node.childGenerator()])
    return '' if node.isspace() else node


def process_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return ' '.join(list(filter(None, [get_node_text(node).strip() for node in soup.childGenerator()])))
