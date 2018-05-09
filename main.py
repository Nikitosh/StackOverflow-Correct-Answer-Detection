import argparse
import logging

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from models.attention_rnn_word2vec_with_question_classifier import AttentionRnnWord2VecWithQuestionClassifier
from data_reader.data_reader import DataReader
from models.fasttext_classifier import FasttextClassifier
from models.hashing_vectorizer_adapter import HashingVectorizerAdapter
from models.rnn_word2vec_classifier import RnnWord2VecClassifier
from models.rnn_word2vec_with_question_classifier import RnnWord2VecWithQuestionClassifier
from models.sklearn_classifier import SKLearnClassifier
from models.tfidf_vectorizer_adapter import TfIdfVectorizerAdapter
from train import train


def test_tfidf(csv_path):
    vectorizer = TfIdfVectorizerAdapter(min_df=2, max_df=0.5)
    data_reader = DataReader(csv_path)
    ids = data_reader.get_ids()
    vectorizer.fit(data_reader, ids)


def test_hashing_naive_bayes(csv_path):
    logging.info('Hashing vectorizer, Naive Bayes classifier')
    classifier = SKLearnClassifier(MultinomialNB(alpha=0.01),
                                      HashingVectorizerAdapter(decode_error='ignore', n_features=2 ** 18,
                                                               alternate_sign=False))
    train(classifier, csv_path)


def test_tfidf_naive_bayes(csv_path):
    logging.info('TF-IDF vectorizer, Naive Bayes classifier')
    classifier = SKLearnClassifier(MultinomialNB(alpha=0.01), TfIdfVectorizerAdapter())
    train(classifier, csv_path)


def test_tfidf_bigrams_naive_bayes(csv_path):
    logging.info('TF-IDF bigrams vectorizer, Naive Bayes classifier')
    classifier = SKLearnClassifier(MultinomialNB(alpha=0.01), TfIdfVectorizerAdapter(ngram_range=(1, 2)))
    train(classifier, csv_path)


def test_tfidf_sgd(csv_path):
    logging.info('TF-IDF vectorizer, SGD classifier')
    classifier = SKLearnClassifier(SGDClassifier(), TfIdfVectorizerAdapter())
    train(classifier, csv_path)


def test_fasttext(csv_path):
    logging.info('Fasttext classifier')
    classifier = FasttextClassifier()
    train(classifier, csv_path)


def test_rnn_word2vec(csv_path):
    logging.info('RNN Word2Vec classifier')
    classifier = RnnWord2VecClassifier(
        answer_body_words_count=100,
        lstm_embed_size=128,
        bidirectional=False,
        dropout=0.1
    )
    train(classifier, csv_path, epochs=5)


def test_rnn_word2vec_with_question(csv_path):
    logging.info('RNN Word2Vec with question classifier')
    classifier = RnnWord2VecWithQuestionClassifier(
        question_title_words_count=50,
        question_body_words_count=100,
        answer_body_words_count=100,
        lstm_embed_size=128,
        bidirectional=False,
        dropout=0.1,
        mode='sum'
    )
    train(classifier, csv_path, epochs=5)


def test_attention_rnn_word2vec_with_question(csv_path):
    logging.info('Attention RNN Word2Vec with question classifier')
    classifier = AttentionRnnWord2VecWithQuestionClassifier()
    train(classifier, csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    logging.basicConfig(filename='logs/log.out', level=logging.INFO)

    test_rnn_word2vec(args.csv_path)
    #test_rnn_word2vec_with_question(args.csv_path)
    #test_attention_rnn_word2vec_with_question(args.csv_path)
    #test_hashing_naive_bayes(args.csv_path)
    #test_tfidf_naive_bayes(args.csv_path)
    #test_tfidf_bigrams_naive_bayes(args.csv_path)
    #test_tfidf_sgd(args.csv_path)
    # test_fasttext(args.csv_path)
    # test_tfidf(args.csv_path)
