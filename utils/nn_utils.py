import logging

import numpy as np
from gensim.models import FastText
from keras import backend as K
from keras.layers import Bidirectional, LSTM, Embedding, Lambda

from utils.other_utils import get_fasttext_model_path


def get_lstm(embed_size, bidirectional, dropout, return_sequences=False):
    if bidirectional:
        return Bidirectional(
            LSTM(embed_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences))
    else:
        return LSTM(embed_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences)


def get_embedding(data_reader, ids):
    model = FastText.load(get_fasttext_model_path(data_reader.csv_file_name))

    all_words = set()
    for words in data_reader.get_processed_questions_and_answers_as_lists(ids):
        all_words.update(words)

    words_count = len(all_words) + 1
    num_features = 300
    embedding_weights = np.zeros((words_count, num_features))
    words_index = {}
    for index, word in enumerate(all_words):
        words_index[word] = index + 1
        if word in model:
            embedding_weights[index + 1, :] = model[word]
        else:
            logging.info(word)

    embedding = Embedding(output_dim=num_features, input_dim=words_count, trainable=False)
    embedding.build((None,))
    embedding.set_weights([embedding_weights])
    return embedding, words_index


def axis(a):
    return len(a._keras_shape) - 1


def dot(a, b):
    return K.batch_dot(a, b, axes=axis(a))


def l2_norm(a, b):
    return K.sqrt(K.sum((a - b) ** 2))


def get_lambda_layer(mode):
    if mode == 'cosine':
        return Lambda(cosine, dot_output_shape)
    if mode == 'gesd':
        return Lambda(gesd, dot_output_shape)
    if mode == 'aesd':
        return Lambda(aesd, dot_output_shape)
    if mode == 'sum':
        return Lambda(sum, sum_output_shape)
    if mode == 'concat':
        return Lambda(concat, concat_output_shape)
    raise NotImplementedError


def cosine(x):
    return dot(x[0], x[1]) / K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1]))


def gesd(x):
    euclidean = 1 / (1 + l2_norm(x[0], x[1]))
    sigmoid = 1 / (1 + K.exp(-1 * (dot(x[0], x[1]) + 1)))
    return euclidean * sigmoid


def aesd(x):
    euclidean = 0.5 / (1 + l2_norm(x[0], x[1]))
    sigmoid = 0.5 / (1 + K.exp(-1 * (dot(x[0], x[1]) + 1)))
    return euclidean + sigmoid


def sum(x):
    return x[0] + x[1]


def concat(x):
    return K.concatenate([x[0], x[1]], axis=1)


def dot_output_shape(input_shape):
    shape = list(input_shape)
    return tuple((shape[0][0], 1))


def sum_output_shape(input_shape):
    return list(input_shape)[0]


def concat_output_shape(input_shape):
    shape = list(input_shape)
    return tuple((shape[0][0], shape[0][1] + shape[1][1]))
