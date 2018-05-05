import re
import gensim
import os.path

import numpy as np

from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

stops = set(stopwords.words('english'))


def lower_text(text):
    text = text.lower()
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = gensim.parsing.preprocessing.strip_punctuation2(text)
    text = gensim.parsing.preprocessing.strip_numeric(text)
    text = gensim.parsing.preprocessing.strip_multiple_whitespaces(text)
    return text


def transform_text(text):
    text = lower_text(text)
    filtered_words = [word for word in text.split() if word not in stops and len(word) >= 3]
    return ' '.join(filtered_words)


def stem_text(text):
    return gensim.parsing.preprocessing.stem_text(transform_text(text))


def print_metrics(y_tests, y_preds):
    print('Accuracy: {}'.format(accuracy_score(y_tests, y_preds)))
    print('Precision: {}'.format(precision_score(y_tests, y_preds)))
    print('Recall: {}'.format(recall_score(y_tests, y_preds)))
    print('F1: {}'.format(f1_score(y_tests, y_preds)))


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
                print(word)
                X[i, j, :] = generate_unit_vector(num_features)
    return X


def get_word2vec_model_path(csv_file_name):
    file_name = os.path.splitext(os.path.basename(csv_file_name))[0]
    if file_name.find('_') != -1:
        file_name = file_name[file_name.find('_') + 1:]
    return 'word2vec/models/{}_model.bin'.format(file_name)

