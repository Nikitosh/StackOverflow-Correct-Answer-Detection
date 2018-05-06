import pandas as pd

from models.utils import lower_text
from sklearn.utils import shuffle


class DataReader:
    def __init__(self, csv_file_name, chunk_size=10 ** 5):
        self.csv_file_name = csv_file_name
        self.chunk_size = chunk_size

    def get_ids(self):
        ids = []
        for chunk in pd.read_csv(self.csv_file_name, chunksize=self.chunk_size):
            for id in chunk['id']:
                ids.append(id)
        return ids

    def _get_data_labels(self, ids):
        for chunk in pd.read_csv(self.csv_file_name, chunksize=self.chunk_size):
            chunk = chunk[chunk['id'].isin(ids)]
            yield chunk[chunk['is_accepted'] == 0].drop('is_accepted', axis=1), \
                  chunk[chunk['is_accepted'] == 0]['is_accepted'], \
                  chunk[chunk['is_accepted'] == 1].drop('is_accepted', axis=1), \
                  chunk[chunk['is_accepted'] == 1]['is_accepted']

    def get_data_labels_batch(self, ids, batch_size):
        X = [pd.DataFrame(), pd.DataFrame()]
        y = [[], []]
        for X0_i, y0_i, X1_i, y1_i in self._get_data_labels(ids):
            X[0] = X[0].append(X0_i)
            X[1] = X[1].append(X1_i)
            y[0].extend(y0_i)
            y[1].extend(y1_i)
            size = batch_size // 2
            while len(X[0]) >= size and len(X[1]) >= size:
                yield shuffle(pd.concat([X[0].iloc[:size], X[1].iloc[:size]]), y[0][:size] + y[1][:size])
                for i in range(2):
                    X[i] = X[i].iloc[size:]
                    y[i] = y[i][size:]

    def get_texts(self, ids):
        X = pd.DataFrame()
        for X0_i, y0_i, X1_i, y1_i in self._get_data_labels(ids):
            X = pd.concat([X, X0_i, X1_i])
            while not X.empty:
                yield lower_text(X['body'].iloc[0])
                X = X.iloc[1:]

    def get_texts_as_lists(self, ids):
        X = pd.DataFrame()
        for X0_i, y0_i, X1_i, y1_i in self._get_data_labels(ids):
            X = pd.concat([X, X0_i, X1_i])
            while not X.empty:
                yield lower_text(X['body'].iloc[0]).split()
                X = X.iloc[1:]
