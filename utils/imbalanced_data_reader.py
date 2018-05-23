import pandas as pd
from sklearn.utils import shuffle

from utils.html_utils import process_html
from utils.word_utils import lower_text, stem_text


class ImbalancedDataReader:
    def __init__(self, csv_file_name, column_name, chunk_size=10 ** 5):
        self.csv_file_name = csv_file_name
        self.column_name = column_name
        self.chunk_size = chunk_size

    def get_ids(self):
        ids = set()
        for chunk in pd.read_csv(self.csv_file_name, chunksize=self.chunk_size):
            for id in chunk[self.column_name]:
                ids.add(id)
        return ids

    def _get_data_labels(self, ids):
        for chunk in pd.read_csv(self.csv_file_name, chunksize=self.chunk_size):
            chunk = chunk[chunk[self.column_name].isin(ids)]
            yield chunk.drop('is_accepted', axis=1), chunk['is_accepted']

    def get_raw_data_labels_batch(self, ids, batch_size):
        X = pd.DataFrame()
        y = []
        for X_i, y_i in self._get_data_labels(ids):
            X = X.append(X_i)
            y.extend(y_i)
            while len(X) >= batch_size:
                yield shuffle(X.iloc[:batch_size], y[:batch_size])
                X = X.iloc[batch_size:]
                y = y[batch_size:]

    def get_stemmed_texts(self, ids):
        for X, y in self._get_data_labels(ids):
            while not X.empty:
                yield stem_text(process_html(X['body'].iloc[0]))
                X = X.iloc[1:]

    def get_processed_texts_as_lists(self, ids):
        for X, y in self._get_data_labels(ids):
            while not X.empty:
                yield lower_text(process_html(X['body'].iloc[0])).split()
                X = X.iloc[1:]

    def get_processed_questions_and_answers_as_lists(self, ids):
        for X, y in self._get_data_labels(ids):
            while not X.empty:
                yield lower_text(process_html(X['question_title'].iloc[0])).split()
                yield lower_text(process_html(X['question_body'].iloc[0])).split()
                yield lower_text(process_html(X['body'].iloc[0])).split()
                X = X.iloc[1:]
