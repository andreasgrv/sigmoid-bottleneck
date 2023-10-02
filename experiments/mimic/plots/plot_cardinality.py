import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from ast import literal_eval


def plot_label_cardinality(counter):
    xx, counts = zip(*counter.most_common())
    plt.bar(xx, counts)
    plt.ylabel('# Examples')
    plt.xlabel('# Active Labels')
    plt.title('MIMIC-III Training Set')
    plt.tight_layout()
    # plt.savefig('example.pdf')
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to openimages csv file (train/valid/test)')

    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df[~df.LABELS.isnull()]

    df['LABELS'] = df['LABELS'].str.split(';')
    print('Label coverage: %d' % len(df.LABELS.explode().unique()))

    # Label freq
    all_labels = Counter(df.LABELS.explode())
    for k, v in all_labels.most_common()[:20]:
        print(k, v)

    # Label combinations
    # all_labels = Counter(df.LABELS)
    # for k, v in all_labels.most_common()[:20]:
    #     print(k, v)

    # df['cardinality'] = df.label_idxs.apply(len)
    # for i, row in df.nlargest(200, 'cardinality').iterrows():
    #     print(row.imagefile, row.cardinality, row.label_strs)
    # for r in df.label_idxs:
    #     print(r, len(r))
    # Label cardinalities
    all_labels_card = Counter([len(r) for r in df.LABELS])
    for k, v in sorted(all_labels_card.most_common(), key=lambda x: x[0]):
        print(k, v)
    plot_label_cardinality(all_labels_card)
    print('Max cardinality: %d' % max(all_labels_card.keys()))
