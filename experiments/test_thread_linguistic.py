import argparse
import logging
from datetime import datetime

from models.neural_nets.thread_linguistic_classifier import ThreadLinguisticClassifier
from run import run
from utils.other_utils import get_dataset_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    logging.basicConfig(
        filename='outputs/logs/{}-{}-thread_linguistic.log'.format(datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                                                                   get_dataset_name(args.csv_path)), level=logging.INFO)

    classifier = ThreadLinguisticClassifier(hidden_layer_size=256)
    run(classifier, args.csv_path, epochs=20)
