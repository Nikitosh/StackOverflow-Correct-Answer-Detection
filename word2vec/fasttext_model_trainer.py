import os.path

from gensim.models import FastText

from utils.other_utils import get_fasttext_model_path
from word2vec.wiki_sentences import WikiSentences


class FasttextModelTrainer:
    WIKI_MODEL_PATH = 'word2vec/fasttext_models/wiki_model.bin'

    def train_on_wiki(self):
        sentences = WikiSentences('wiki-english-20171001')
        fasttext_model = FastText(sentences, size=300)
        fasttext_model.save(FasttextModelTrainer.WIKI_MODEL_PATH)

    def train(self, data_reader, ids):
        model_path = get_fasttext_model_path(data_reader.csv_file_name)
        if os.path.isfile(model_path):
            return
        if not os.path.isfile(FasttextModelTrainer.WIKI_MODEL_PATH):
            self.train_on_wiki()
        fasttext_model = FastText.load(FasttextModelTrainer.WIKI_MODEL_PATH)
        fasttext_model.min_count = 1
        fasttext_model.build_vocab(data_reader.get_processed_texts_as_lists(ids), update=True)
        fasttext_model.train(data_reader.get_processed_texts_as_lists(ids), epochs=fasttext_model.iter,
                             total_examples=fasttext_model.corpus_count)
        fasttext_model.save(model_path)
