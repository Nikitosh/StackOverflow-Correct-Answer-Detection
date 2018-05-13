import argparse
import logging
from datetime import datetime

from models.fasttext.fasttext_classifier import FasttextClassifier
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    logging.basicConfig(filename='outputs/logs/{}.log'.format(datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')),
                        level=logging.INFO)

    logging.info('Fasttext classifier')
    classifier = FasttextClassifier()
    train(classifier, args.csv_path)
