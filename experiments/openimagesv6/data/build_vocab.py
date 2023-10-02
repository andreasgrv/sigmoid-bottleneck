import os
import csv
import argparse
import pandas as pd

from collections import defaultdict


def load_vocab(f):
    vocab = set()
    with open(f, 'r') as f:
        for line in f:
            vocab.add(line.rstrip())
    return vocab


def load_name_dict(f):
    mid_to_class = {}
    with open(f, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Skip header
        next(reader)
        for k, v in reader:
            mid_to_class[k] = v
    return mid_to_class


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, help='Path to labels folder', required=True)
    parser.add_argument('--downloaded', type=str, help='Path to folder with names of downloaded images', required=True)
    parser.add_argument('--trainable', type=str, help='Path to list of trainable labels', required=True)
    parser.add_argument('--classnames', type=str, help='Path to label idx to label names csv', required=True)

    args = parser.parse_args()

    # Mapping from label hashes to label names
    print('Loading name lookup...')
    vocab_to_name = load_name_dict(args.classnames)

    # The paper created a list of labels
    # that occur in the training data at least 100 times
    print('Loading trainable labels...')
    trainable = load_vocab(args.trainable)

    dfs = dict()
    for split in ['train', 'valid', 'test']:
        print('Loading %s annotations...' % split)
        labels_path = os.path.join(args.labels, '%s.csv' % split)
        df = pd.read_csv(labels_path)
        # Confidence = 1 means the label is active (ignore non-active)
        df = df[df.Confidence == 1]
        # Just keep the image and labels columns
        df = df[['ImageID', 'LabelName']]
        df = df.rename(columns={'ImageID': 'imagefile', 'LabelName': 'labels'})
        df = df.groupby('imagefile', as_index=False).agg(list)

        # Filter to downloaded images
        dl_path = os.path.join(args.downloaded, '%s.csv' % split)
        dl = load_vocab(dl_path)
        df['downloaded'] = df.imagefile.apply(lambda x: x in dl)
        df = df[df.downloaded == True]
        df = df.drop(columns=['downloaded'])

        dfs[split] = df

    # We filted to the filenames starting with 0
    dfs['train'] = dfs['train'][dfs['train'].imagefile.str.startswith('1')]
    print(dfs['train'])

    print('Building label vocab...')
    train_vocab = set(dfs['train'].labels.explode().unique())
    # We narrow down to "trainable" - sort ids to avoid any randomness in order
    train_vocab = train_vocab.intersection(trainable)
    vocab = tuple(sorted(train_vocab))

    # valid_vocab = set(dfs['valid'].labels.explode().unique())
    # valid_vocab = valid_vocab.intersection(trainable)
    # print(dfs['valid'])
    #
    # test_vocab = set(dfs['test'].labels.explode().unique())
    # test_vocab = test_vocab.intersection(trainable)
    # print(dfs['test'])
    #
    # print('Train vocab size: %d' % len(train_vocab))
    # print('Valid vocab size: %d' % len(valid_vocab))
    # print('Test vocab size : %d' % len(test_vocab))
    #
    # print('Train overlap valid: %d' % len(train_vocab.intersection(valid_vocab)))
    # print('Train overlap test:  %d' % len(train_vocab.intersection(test_vocab)))

    # Only keep examples that are covered by train_vocab
    for split in ['train', 'valid', 'test']:
        dfs[split]['covered'] = dfs[split].labels.apply(lambda ls: all(l in train_vocab for l in ls))
        dfs[split] = dfs[split][dfs[split].covered == True]
        dfs[split] = dfs[split].drop(columns=['covered'])
        if split == 'train':
            # Now that we got rid of some training examples
            # some labels may not show up in any training examples, so recalibrate vocab
            vocab = tuple(sorted(dfs['train'].labels.explode().unique()))
            train_vocab = set(vocab)

    valid_vocab = set(dfs['valid'].labels.explode().unique())
    valid_vocab = valid_vocab.intersection(trainable)
    print(dfs['valid'])

    test_vocab = set(dfs['test'].labels.explode().unique())
    test_vocab = test_vocab.intersection(trainable)
    print(dfs['test'])

    print('Train vocab size: %d' % len(train_vocab))
    print('Valid vocab size: %d' % len(valid_vocab))
    print('Test vocab size : %d' % len(test_vocab))

    print('Train overlap valid: %d' % len(train_vocab.intersection(valid_vocab)))
    print('Train overlap test:  %d' % len(train_vocab.intersection(test_vocab)))

    vocab_to_idx = {v: k for k, v in enumerate(vocab)}


    for split in ['train', 'valid', 'test']:
        dfs[split]['label_strs'] = dfs[split].labels.apply(lambda row: list(sorted(vocab_to_name[l] for l in row)))
        # Convert label name hashes to idxs
        dfs[split]['label_idxs'] = dfs[split].labels.apply(lambda row: list(sorted(vocab_to_idx[l] for l in row)))
        dfs[split] = dfs[split].drop(columns=['labels'])
        print('Saving %d examples to %s.csv...' % (len(dfs[split]), split))
        # Shuffle the rows
        dfs[split] = dfs[split].sample(frac=1, random_state=41)
        # For valid only keep 5k examples
        if split == 'valid':
            dfs[split].head(5000).to_csv('%s.csv' % split)
        else:
            dfs[split].to_csv('%s.csv' % split)

    print('Saving label vocabulary to vocab.csv...')
    # Save label vocabulary
    vlines = ['%d,%s,%s\n' % (vocab_to_idx[v], vocab_to_name[v], v) for v in vocab]
    with open('vocab.csv', 'w') as f:
        f.writelines(vlines)
