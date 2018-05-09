import subprocess

from preprocessors.csv_to_fasttext_preprocessor import CsvToFasttextPreprocessor


class FasttextWrapper:

    def fit(self, file_name, data_reader, ids, lr=0.05, epoch=5):
        preprocessor = CsvToFasttextPreprocessor()
        preprocessor.process_train(file_name, data_reader, ids)
        subprocess.call('fasttext supervised -input {} -output {} -lr {} -epoch {}'
                        .format(file_name, 'fasttext/models/models', lr, epoch))

    def predict(self, file_name, data_reader, ids):
        preprocessor = CsvToFasttextPreprocessor()
        preprocessor.process_test(file_name, data_reader, ids)
        output = subprocess.check_output('fasttext predict fasttext/models/models.bin {}'.format(file_name))\
            .decode("utf-8")
        return list(int(label[-1]) for label in output.split('\r\n') if label)
