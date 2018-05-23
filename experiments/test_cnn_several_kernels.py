import argparse
import logging
from datetime import datetime

from models.neural_nets.cnn_several_kernels_classifier import CnnSeveralKernelsClassifier
from run import run
from utils.other_utils import get_dataset_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    logging.basicConfig(filename='outputs/logs/{}-{}-cnn_several_kernels.log'.format(
        datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), get_dataset_name(args.csv_path)),
                        level=logging.INFO)

    classifier = CnnSeveralKernelsClassifier(
        question_body_words_count=500,
        answer_body_words_count=500,
        filters_count=32,
        kernel_sizes=[2, 3, 5, 7],
        hidden_layer_size=128,
        dropout=0.5,
    )
    run(classifier, args.csv_path, epochs=20)
