import pandas as pd

from model.utils import transform_text
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

    def get_texts(self, ids):
        for chunk in pd.read_csv(self.csv_file_name, chunksize=self.chunk_size):
            chunk = chunk[chunk['id'].isin(ids)]
            yield chunk[chunk['is_accepted'] == 0]['body'].map(transform_text), \
                  chunk[chunk['is_accepted'] == 0]['is_accepted'], \
                  chunk[chunk['is_accepted'] == 1]['body'].map(transform_text), \
                  chunk[chunk['is_accepted'] == 1]['is_accepted']

    def get_texts_batch(self, ids, batch_size):
        X = [[], []]
        y = [[], []]
        for X0_i, y0_i, X1_i, y1_i in self.get_texts(ids):
            X[0].extend(X0_i)
            X[1].extend(X1_i)
            y[0].extend(y0_i)
            y[1].extend(y1_i)
            size = batch_size // 2
            while len(X[0]) >= size and len(X[1]) >= size:
                yield shuffle(X[0][:size] + X[1][:size], y[0][:size] + y[1][:size])
                for i in range(2):
                    X[i] = X[i][size:]
                    y[i] = y[i][size:]
