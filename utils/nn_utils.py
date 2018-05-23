from keras.layers import Bidirectional, LSTM


def get_lstm(embed_size, bidirectional, dropout, return_sequences=False):
    if bidirectional:
        return Bidirectional(
            LSTM(embed_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences))
    else:
        return LSTM(embed_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences)