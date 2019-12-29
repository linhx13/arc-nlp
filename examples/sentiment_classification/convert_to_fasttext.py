# -*- coding: utf-8 -*-

import logging
import csv

logger = logging.getLogger(__name__)


def run_convert(input_path, output_path, text_column, label_column):
    with open(input_path) as fin, open(output_path, 'w') as fout:
        reader = csv.DictReader(fin, delimiter='\t')
        for row in reader:
            label = '__label__%s' % row[label_column]
            text = row[text_column]
            fout.write('%s %s\n' % (label, text))


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--text_column", required=True)
    parser.add_argument("--label_column", required=True)
    args = parser.parse_args()
    logger.info('args: %s' % args)

    run_convert(args.input_path, args.output_path,
                args.text_column, args.label_column)