import argparse
import logging
from datetime import datetime

from models.neural_nets.cnn_word2vec_with_question_classifier import CnnWord2VecWithQuestionClassifier
from run import run
from utils.utils import get_dataset_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    logging.basicConfig(filename='outputs/logs/{}-{}-cnn_word2vec_with_question.log'.format(
        datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), get_dataset_name(args.csv_path)),
                        level=logging.INFO)

    classifier = CnnWord2VecWithQuestionClassifier(
        question_body_words_count=500,
        answer_body_words_count=500,
        filters_count=128,
        kernel_size=3,
        hidden_layer_size=256,
        dropout=0.5,
    )
    run(classifier, args.csv_path, epochs=20)
