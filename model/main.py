import argparse

from model.naive_bayes_classifier import NaiveBayesClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.process(args.csv_path)