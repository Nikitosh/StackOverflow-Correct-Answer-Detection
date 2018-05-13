import argparse
import logging
from datetime import datetime

from sklearn.naive_bayes import MultinomialNB

from models.sklearn.sklearn_classifier import SKLearnClassifier
from models.sklearn.tfidf_vectorizer_adapter import TfIdfVectorizerAdapter
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    logging.basicConfig(filename='outputs/logs/{}.log'.format(datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')),
                        level=logging.INFO)

    logging.info('Naive Bayes classifier')
    classifier = SKLearnClassifier(MultinomialNB(alpha=0.01), TfIdfVectorizerAdapter())
    train(classifier, args.csv_path)
