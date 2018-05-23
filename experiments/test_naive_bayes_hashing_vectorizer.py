import argparse
import logging
from datetime import datetime

from sklearn.naive_bayes import MultinomialNB

from models.sklearn.hashing_vectorizer_adapter import HashingVectorizerAdapter
from models.sklearn.sklearn_classifier import SKLearnClassifier
from run import run
from utils.other_utils import get_dataset_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    logging.basicConfig(filename='outputs/logs/{}-{}-naive_bayes_hashing_vectorizer.log'.format(
        datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), get_dataset_name(args.csv_path)),
                        level=logging.INFO)

    logging.info('Naive Bayes classifier')
    classifier = SKLearnClassifier(MultinomialNB(alpha=0.01),
                                   HashingVectorizerAdapter(decode_error='ignore', n_features=2 ** 18,
                                                            alternate_sign=False))
    run(classifier, args.csv_path)
