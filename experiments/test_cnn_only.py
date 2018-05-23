import argparse
import logging
from datetime import datetime

from models.neural_nets.cnn_only_classifier import CnnOnlyClassifier
from run import run
from utils.other_utils import get_dataset_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    logging.basicConfig(filename='outputs/logs/{}-{}-cnn_only.log'.format(
        datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), get_dataset_name(args.csv_path)),
                        level=logging.INFO)

    classifier = CnnOnlyClassifier(
        question_body_words_count=300,
        answer_body_words_count=500,
        filters_count=32,
        kernel_sizes=[2, 3, 5, 7],
        mode='aesd'
    )
    run(classifier, args.csv_path, epochs=20)
