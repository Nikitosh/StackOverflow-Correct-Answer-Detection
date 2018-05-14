import argparse
import logging
from datetime import datetime

from models.neural_nets.rnn_word2vec_classifier import RnnWord2VecClassifier
from run import run
from utils.utils import get_dataset_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    logging.basicConfig(filename='outputs/logs/{}-{}-rnn_word2vec.log'.format(
        datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), get_dataset_name(args.csv_path)),
        level=logging.INFO)

    classifier = RnnWord2VecClassifier(
        answer_body_words_count=100,
        lstm_embed_size=128,
        hidden_layer_size=64,
        bidirectional=True,
        dropout=0.1
    )
    run(classifier, args.csv_path)
