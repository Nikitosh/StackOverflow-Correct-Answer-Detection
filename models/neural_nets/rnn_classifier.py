import logging

from keras import Input, Model
from keras.layers import Dropout, Dense, concatenate

from models.neural_nets.neural_net_classifier import NeuralNetClassifier
from utils.html_utils import process_html
from utils.nn_utils import get_lstm
from utils.word_utils import lower_text


class RnnClassifier(NeuralNetClassifier):
    def __init__(self, answer_body_words_count, lstm_embed_size, hidden_layer_size, bidirectional, dropout):
        super().__init__()
        logging.info('RNN Word2Vec classifier')
        logging.info('answer_body_words_count = {}, lstm_embed_size = {}, bidirectional = {}, dropout = {}'.format(
            answer_body_words_count, lstm_embed_size, bidirectional, dropout))

        self.answer_body_words_count = answer_body_words_count
        self.lstm_embed_size = lstm_embed_size
        self.hidden_layer_size = hidden_layer_size
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.lstm_layer = None
        self.dropout_layer = None

    def create_model(self):
        self.lstm_layer = get_lstm(self.lstm_embed_size, self.bidirectional, self.dropout)
        self.dropout_layer = Dropout(self.dropout)

        answer_body_input = Input(shape=(self.answer_body_words_count,), name='answer_body_input')
        other_features = Input(shape=(NeuralNetClassifier.OTHER_FEATURES_COUNT,), name='other_features')

        answer_body_features = self.embedding(answer_body_input)
        rnn_features = self.lstm_layer(answer_body_features)
        rnn_features = self.dropout_layer(rnn_features)
        features = concatenate([rnn_features, other_features], name='features')
        fc = Dense(self.hidden_layer_size, activation='relu')(features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(inputs=[answer_body_input, other_features], outputs=[output])
        self.compile_model()

    def transform_input(self, data):
        answer_body = self.transform_sentence_batch_to_vector([lower_text(process_html(X)) for X in data['body']],
                                                              self.answer_body_words_count)
        other_features = self.get_other_features(data)
        return [answer_body, other_features]
