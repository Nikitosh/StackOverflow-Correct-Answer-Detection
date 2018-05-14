import argparse
import logging
from datetime import datetime

from models.fasttext.fasttext_classifier import FasttextClassifier
from run import run
from utils.utils import get_dataset_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    logging.basicConfig(filename='outputs/logs/{}-{}-fasttext.log'.format(
        datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), get_dataset_name(args.csv_path)),
                        level=logging.INFO)

    logging.info('Fasttext classifier')
    classifier = FasttextClassifier()
    run(classifier, args.csv_path)
