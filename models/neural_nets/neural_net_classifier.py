import logging

import numpy as np
from keras.utils import plot_model

from utils.other_utils import get_logging_filename
from utils.word_utils import get_embedding
from word2vec.fasttext_model_trainer import FasttextModelTrainer


class NeuralNetClassifier:
    OTHER_FEATURES_COUNT = 65
    THREAD_FEATURES_COUNT = 2

    def __init__(self):
        self.num_features = 300
        self.model = None
        self.embedding = None
        self.words_index = None
        self.class_weight = {0: 0.5, 1: 0.5}

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary(print_fn=logging.info)
        plot_model(self.model, to_file='outputs/models_images/model-{}.png'.format(get_logging_filename()))

    def pretrain(self, data_reader, ids):
        trainer = FasttextModelTrainer()
        trainer.train(data_reader, ids)
        self.embedding, self.words_index = get_embedding(data_reader, ids)
        labels_number = [0, 0]
        for X, y in data_reader.get_raw_data_labels_batch(ids, batch_size=64):
            for y_i in y:
                labels_number[y_i] += 1
        all_labels_number = labels_number[0] + labels_number[1]
        self.class_weight = {0: labels_number[1] / all_labels_number, 1: labels_number[0] / all_labels_number}
        logging.info('Labels number: {}'.format(labels_number))
        self.create_model()

    def create_model(self):
        raise NotImplementedError

    def transform_input(self, X):
        raise NotImplementedError

    def train_on_batch(self, X, y):
        return self.model.train_on_batch(self.transform_input(X), y, class_weight=self.class_weight)

    def evaluate(self, X, y):
        return self.model.evaluate(self.transform_input(X), y)

    def predict(self, X):
        return self.model.predict(self.transform_input(X))

    def save(self, epoch):
        self.model.save('outputs/models/model-{}-{}.h5'.format(get_logging_filename(), epoch))

    def transform_sentence_batch_to_vector(self, sentences, document_max_num_words):
        X = np.zeros((len(sentences), document_max_num_words))
        for i in range(len(sentences)):
            words = sentences[i].split()
            for j, word in enumerate(words):
                if j == document_max_num_words:
                    break
                if word in self.words_index:
                    X[i, j] = self.words_index[word]
                else:
                    logging.info('Unknown word: {}'.format(word))
        return X

    def get_other_features(self, data):
        return data[[
            'answer_count',
            'a_count_n1', 'code_count_n1', 'p_count_n1', 'upper_count_n1', 'lower_count_n1', 'space_count_n1',
            'length_n1', 'longest_sentence_char_count_n1', 'longest_sentence_word_count_n1', 'average_words_n1',
            'average_chars_n1', 'sentence_count_n1', 'ari_n1', 'fre_n1', 'si_n1', 'fkg_n1', 'cli_n1', 'gf_n1',
            'lix_n1', 'age_n1',

            'a_count_n2', 'code_count_n2', 'p_count_n2', 'upper_count_n2', 'lower_count_n2', 'space_count_n2',
            'length_n2', 'longest_sentence_char_count_n2', 'longest_sentence_word_count_n2', 'average_words_n2',
            'average_chars_n2', 'sentence_count_n2', 'ari_n2', 'fre_n2', 'si_n2', 'fkg_n2', 'cli_n2', 'gf_n2',
            'lix_n2', 'age_n2',

            'a_count_pos', 'code_count_pos', 'p_count_pos', 'upper_count_pos', 'lower_count_pos', 'space_count_pos',
            'length_pos', 'longest_sentence_char_count_pos', 'longest_sentence_word_count_pos', 'average_words_pos',
            'average_chars_pos', 'sentence_count_pos', 'ari_pos', 'fre_pos', 'si_pos', 'fkg_pos', 'cli_pos', 'gf_pos',
            'lix_pos', 'age_pos',

            'qa_overlap', 'qa_idf_overlap', 'qa_filtered_overlap', 'qa_filtered_idf_overlap'
        ]].as_matrix()
