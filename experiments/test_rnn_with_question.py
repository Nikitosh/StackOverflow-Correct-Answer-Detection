import argparse
import logging
from datetime import datetime

from models.neural_nets.rnn_with_question_classifier import RnnWord2VecWithQuestionClassifier
from run import run
from utils.other_utils import get_dataset_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    logging.basicConfig(filename='outputs/logs/{}-{}-rnn_word2vec_with_question.log'.format(
        datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), get_dataset_name(args.csv_path)),
                        level=logging.INFO)

    classifier = RnnWord2VecWithQuestionClassifier(
        question_title_words_count=50,
        question_body_words_count=300,
        answer_body_words_count=500,
        lstm_embed_size=128,
        hidden_layer_size=128,
        bidirectional=True,
        dropout=0.5,
        mode='cosine'
    )
    run(classifier, args.csv_path, epochs=20)
