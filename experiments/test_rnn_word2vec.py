import argparse
import logging
from datetime import datetime

from models.neural_nets.rnn_word2vec_classifier import RnnWord2VecClassifier
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    logging.basicConfig(filename='outputs/logs/{}.log'.format(datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')),
                        level=logging.INFO)

    classifier = RnnWord2VecClassifier(
        answer_body_words_count=100,
        lstm_embed_size=128,
        hidden_layer_size=64,
        bidirectional=True,
        dropout=0.1
    )
    train(classifier, args.csv_path)