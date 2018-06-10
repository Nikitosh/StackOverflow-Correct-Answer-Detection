import logging

from keras import Input, Model
from keras.layers import Dense

from models.neural_nets.neural_net_classifier import NeuralNetClassifier


class ThreadClassifier(NeuralNetClassifier):
    def __init__(self, hidden_layer_size):
        super().__init__()
        logging.info('Thread classifier')
        logging.info('hidden_layer_size = {}'.format(hidden_layer_size))

        self.hidden_layer_size = hidden_layer_size

    def create_model(self):
        thread_features = Input(shape=(NeuralNetClassifier.THREAD_FEATURES_COUNT,), name='thread_features')

        fc = Dense(self.hidden_layer_size, activation='relu')(thread_features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(inputs=[thread_features], outputs=[output])
        self.compile_model()

    def pretrain(self, data_reader, ids):
        self.create_model()

    def transform_input(self, data):
        #thread_features = data[['answer_count', 'age_pos']].as_matrix()
        thread_features = data[['answer_count', 'age_n1', 'score_n1', 'age_n2', 'score_n2', 'age_pos',
                                'score_pos', 'qa_overlap', 'qa_idf_overlap', 'qa_filtered_overlap',
                                'qa_filtered_idf_overlap']].as_matrix()
        return [thread_features]
