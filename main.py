import argparse

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from models.data_reader import DataReader
from models.fasttext_classifier import FasttextClassifier
from models.hashing_vectorizer_adapter import HashingVectorizerAdapter
from models.rnn_word2vec_classifier import RnnWord2VecClassifier
from models.sklearn_classifier import SKLearnClassifier
from models.tfidf_vectorizer_adapter import TfIdfVectorizerAdapter
from word2vec.word2vec_model_trainer import Word2VecModelTrainer


def test_hashing_naive_bayes(csv_path):
    nb_classifier = SKLearnClassifier()
    classifier = MultinomialNB(alpha=0.01)
    vectorizer = HashingVectorizerAdapter(decode_error='ignore', n_features=2 ** 18, alternate_sign=False)
    print('Hashing vectorizer, Naive Bayes classifier')
    nb_classifier.process(csv_path, classifier, vectorizer)


def test_tfidf_naive_bayes(csv_path):
    nb_classifier = SKLearnClassifier()
    classifier = MultinomialNB(alpha=0.01)
    vectorizer = TfIdfVectorizerAdapter()
    print('TF-IDF vectorizer, Naive Bayes classifier')
    nb_classifier.process(csv_path, classifier, vectorizer)


def test_tfidf_bigrams_naive_bayes(csv_path):
    nb_classifier = SKLearnClassifier()
    classifier = MultinomialNB(alpha=0.01)
    vectorizer = TfIdfVectorizerAdapter(ngram_range=(1, 2))
    print('TF-IDF bigrams vectorizer, Naive Bayes classifier')
    nb_classifier.process(csv_path, classifier, vectorizer)


def test_tfidf_sgd(csv_path):
    sgd_classifier = SKLearnClassifier()
    classifier = SGDClassifier()
    vectorizer = TfIdfVectorizerAdapter()
    print('TF-IDF vectorizer, SGD classifier')
    sgd_classifier.process(csv_path, classifier, vectorizer)


def test_tfidf(csv_path):
    vectorizer = TfIdfVectorizerAdapter(min_df=2, max_df=0.5)
    data_reader = DataReader(csv_path)
    ids = data_reader.get_ids()
    vectorizer.fit(data_reader, ids)
    print(len(vectorizer.vectorizer.vocabulary_))


def test_fasttext(csv_path):
    classifier = FasttextClassifier()
    print('Fasttext classifier')
    classifier.process(csv_path)


def test_rnn_word2vec(csv_path):
    classifier = RnnWord2VecClassifier()
    print('RNN Word2Vec classifier')
    classifier.process(csv_path)


def train_word2vec(csv_path):
    trainer = Word2VecModelTrainer()
    trainer.train(csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to .csv file')
    args = parser.parse_args()
    train_word2vec(args.csv_path)
    #test_rnn_word2vec(args.csv_path)
    #test_fasttext(args.csv_path)
    #test_tfidf(args.csv_path)
    #test_hashing_naive_bayes(args.csv_path)
    #test_tfidf_naive_bayes(args.csv_path)
    #test_tfidf_bigrams_naive_bayes(args.csv_path)
    #test_tfidf_sgd(args.csv_path)
