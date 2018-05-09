import argparse

from preprocessors.xml_to_csv_preprocessor import XmlToCsvPreprocessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_path', type=str, help='Path to .xml file')
    args = parser.parse_args()
    preprocessor = XmlToCsvPreprocessor()
    preprocessor.process(args.xml_path)
