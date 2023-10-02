# Based on:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import os
import json
import glob
import torch
import numpy as np
import pandas as pd
import datasets

from itertools import groupby
from pathos.multiprocessing import ProcessingPool as Pool
from collections import defaultdict
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
# from torchvision.io import read_image, ImageReadMode
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer


class MeSHEncoder(object):
    """docstring for MeSHEncoder"""
    def __init__(self, filename):
        super(MeSHEncoder, self).__init__()
        self.filename = filename
        self.index = dict()
        self.inv_index = dict()
        self._build_index()

    def _build_index(self):
        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                name, meshidx = line.rsplit('=', 1)
                self.index[name] = meshidx
                self.inv_index[meshidx] = name

    def encode(self, names):
        meshidxs = [self.index[name] for name in names]
        return meshidxs

    def decode(self, meshidxs):
        names = [self.inv_index[meshidx] for meshidx in meshidxs]
        return names


class BPELabelEncoder(object):
    """docstring for BPELabelEncoder"""
    def __init__(self, tokenizer):
        super(BPELabelEncoder, self).__init__()
        self.tokenizer = tokenizer
        # self.str_index = defaultdict(set) 
        self.index = defaultdict(set) 
        # The subset of bpe token idxs that are being used
        self.idx_vocab = set()
        self.inv_vocab = dict()
        self.vocab = dict()

    @property
    def num_labels(self):
        # return len(self.index)
        return len(self.idx_vocab)

    @property
    def num_classes(self):
        return self.num_labels + 2

    def build_vocab(self, labelfile):
        with open(labelfile, 'r') as f:
            for label in f:
                label = label.strip()
                # self.str_index[label] = set(self.tokenizer.tokenize(label))
                bpe_parts =  set(self.tokenizer.encode(label, add_special_tokens=False))
                self.index[label] = bpe_parts
                self.idx_vocab.update(bpe_parts)
        self.idx_vocab = tuple(sorted(self.idx_vocab))
        self.inv_vocab = dict(enumerate(self.idx_vocab))
        self.vocab = {v:k for k, v in self.inv_vocab.items()}

    def str2int(self, label):
        if label == '<EOS>':
            return self.num_labels
        elif label == '<PAD>':
            return self.num_labels + 1
        else:
            raise AttributeError('Unknown label %s' % label)

    def str2idxs(self, label):
        idxs = self.tokenizer.encode(label, add_special_tokens=False)
        cls_idxs = [self.vocab[idx] for idx in idxs]
        return tuple(sorted(set(cls_idxs)))

    def idxs2str(self, cls_idxs):
        # Map from prediction idxs to BPE idxs
        token_idxs = [self.inv_vocab[idx] for idx in cls_idxs]
        uniq = set(token_idxs)
        labels, added = [], []
        for k, v in sorted(self.index.items(), key=lambda x: len(x[1]), reverse=True):
            if len(uniq.intersection(v)) == len(v):
                found = False
                for prev in added:
                    if len(prev.intersection(v)) == len(v):
                        found = True
                        break
                if not found:
                    labels.append(k)
                    added.append(v)
        return tuple(sorted(labels))


def encode_label_list(encoder,
                      labels,
                      max_cardinality,
                      include_EOS=False):

    PAD_IDX = encoder.str2int('<PAD>')
    EOS_IDX = encoder.str2int('<EOS>')

    # TODO: Consider traversing MeSH tree DFS
    # Max number of labels that can be assigned
    num_active = len(labels)

    ids = []
    # Convert string labels to int ids
    for label in labels:
        ids.append(encoder.str2int(label))

    # Add EOS if we are doing seq prediction
    if include_EOS:
        ids.append(EOS_IDX)
    # Check we have not passed truncation threshold
    if len(ids) > max_cardinality:
        print('Warning: Truncating labels!')
        ids = ids[:max_cardinality]
        if include_EOS:
            ids[-1] = EOS_IDX
    # Pad to max_cardinality if needed
    if len(ids) < max_cardinality:
        num_pad = max_cardinality - len(ids)
        ids = tuple(list(ids) + [PAD_IDX,] * num_pad)

    return {'labels': ids}


def encode_bpe_label_list(encoder,
                          labels,
                          max_cardinality,
                          include_EOS=False):

    PAD_IDX = encoder.str2int('<PAD>')
    EOS_IDX = encoder.str2int('<EOS>')

    # TODO: Consider traversing MeSH tree DFS
    # Max number of labels that can be assigned
    num_active = len(labels)

    ids = []
    # Convert string labels to int ids
    for label in labels:
        token_idxs = encoder.str2idxs(label)
        ids.extend(token_idxs)
    ids = tuple(sorted(set(ids)))

    # Add EOS if we are doing seq prediction
    if include_EOS:
        ids.append(EOS_IDX)
    # Check we have not passed truncation threshold
    if len(ids) > max_cardinality:
        print('Warning: Truncating labels!')
        ids = ids[:max_cardinality]
        if include_EOS:
            ids[-1] = EOS_IDX
    # Pad to max_cardinality if needed
    if len(ids) < max_cardinality:
        num_pad = max_cardinality - len(ids)
        ids = tuple(list(ids) + [PAD_IDX,] * num_pad)

    return {'bpe_labels': ids}


def encode_batch_labels(labels, label_encoder, method):
    # labels: batch_dim x max_active_labels
    # the tensor contains the indices of active labels
    # tensor second dim is always max_active_labels size
    # <PAD> token used to pad to max size

    PAD = label_encoder.str2int('<PAD>')
    # label_encoder.num_classes has all labels + placeholders e.g. <PAD> <EOS>
    ALL_LABELS = label_encoder.num_classes
    # label_encoder.num_labels only includes labels that are valid entities
    ENTITY_LABELS = label_encoder.num_labels
    # We assume that placeholder labels come after entity labels
    assert (PAD + 1) > ENTITY_LABELS
    if method in ('sigmoid', 'blocksigmoid'):
        targets = one_hot_encode(labels, ALL_LABELS)
        # Handle padding token
        # Drop last two dimensions
        targets = targets[:, :ENTITY_LABELS]
    return targets


def one_hot_encode(labels, num_labels):
    batch_size, max_active_labels = labels.shape
    # ============ Convert labels to torch BCELoss format =================
    # [1, 4, 5, 5] -> [0, 1, 0, 0, 1] (if 5 was <PAD>)
    # Include all classes, this includes invalid classes like <PAD>
    # We do so to index without errors - we remove them afterwards
    targets = torch.zeros(batch_size,
                          num_labels,
                          dtype=torch.long,
                          device=labels.device)
    # We need to index the targets vector and set active indices to 1
    batch_indices = torch.arange(batch_size,
                                 device=labels.device).view(-1, 1)
    batch_indices = torch.tile(batch_indices, [max_active_labels])
    targets[batch_indices, labels] = 1
    return targets


def one_hot_decode(preds):
    labels = []
    # Input is of size batch_size x num_labels
    lbl_idxs = preds.nonzero().tolist()
    # Group results by batch index
    batch_labels = defaultdict(list)
    batch_labels.update([(k, tuple(v)) for k, v 
                          in groupby(lbl_idxs, lambda x: x[0])])
    for batch_idx in range(len(preds)):
        lbls = [row[1] for row in batch_labels[batch_idx]]
        labels.append(lbls)
    return labels


def decode_batch_labels(preds, label_encoder, method):
    str_labels = []
    if method in ('sigmoid', 'blocksigmoid'):
        labels = one_hot_decode(preds)
        for lbls in labels:
            if isinstance(label_encoder, datasets.ClassLabel):
                str_lbls = label_encoder.int2str(lbls)
            elif isinstance(label_encoder, BPELabelEncoder):
                # Decode BPE idxs into active labels
                str_lbls = label_encoder.idxs2str(lbls)
            else:
                raise ValueError('Unknown encoder type %r' % type(encoder))
            str_labels.append(str_lbls)
    return str_labels


def process_shard(dataset, shard_idx, conf):

    tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer_model)

    has_labels = 'meshMajor' in dataset.features
    shard = dataset.shard(conf.num_proc, shard_idx)
    # Encode text
    # TODO: Include title
    # TODO: Increase max_length
    shard = shard.map(lambda x: tokenizer('%s %s %s' % (x['journal'], x['title'], x['abstractText']),
                                          truncation=True,
                                          padding='max_length',
                                          max_length=conf.data.max_length))
    # Encode labels
    if has_labels:
        shard = shard.remove_columns(['year'])
        shard = shard.map(lambda x: encode_label_list(conf.label_encoder,
                                    x['meshMajor'],
                                    max_cardinality=conf.data.max_cardinality))
        if conf.use_bpe_labels:
            shard = shard.map(lambda x: encode_bpe_label_list(conf.bpe_label_encoder,
                                        x['meshMajor'],
                                        max_cardinality=conf.data.max_cardinality))
        # We use output_all_columns since we want to keep pmid

        cols = ['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'doc_embedding']
        if conf.use_bpe_labels:
            cols.append('bpe_labels')
        shard.set_format(type='torch',
                         columns=cols,
                         output_all_columns=True)
        shard = shard.remove_columns(['meshMajor', 'journal', 'abstractText', 'title'])
    else:
        shard.set_format(type='torch',
                         columns=['input_ids', 'token_type_ids', 'attention_mask'],
                         output_all_columns=True)
        shard = shard.remove_columns(['journal', 'abstractText', 'title'])
    return shard


def load_label_encoder(conf):
    print('Loading labels from "%s"...' % conf.data.labels_file)
    labels_file = conf.data.labels_file
    label_encoder = datasets.ClassLabel(names_file=labels_file)
    label_encoder.num_labels = label_encoder.num_classes - 2
    conf.num_labels = label_encoder.num_labels
    assert label_encoder.str2int('<EOS>') == (conf.num_labels)
    assert label_encoder.str2int('<PAD>') == (conf.num_labels + 1)
    return label_encoder


def load_bpe_label_encoder(conf):
    print('Loading labels from "%s"...' % conf.data.labels_file)
    labels_file = conf.data.labels_file
    label_encoder = BPELabelEncoder(conf.tokenizer)
    # Create vocabulary from labels_file
    label_encoder.build_vocab(labels_file)
    conf.num_labels = label_encoder.num_labels
    assert label_encoder.str2int('<EOS>') == (conf.num_labels)
    assert label_encoder.str2int('<PAD>') == (conf.num_labels + 1)
    return label_encoder


def create_dataloader(conf, split='train'):
    assert split in ['train', 'valid']
    shuffle = (split == 'train')
    # ================== PREPARE LABELS =======================================
    conf.tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer_model)
    conf.label_encoder = load_label_encoder(conf)
    if conf.use_bpe_labels:
        conf.bpe_label_encoder = load_bpe_label_encoder(conf)

    # ================== PROCESS DATASET  =====================================
    data_path = conf.data.train_path if split == 'train' else conf.data.valid_path
    # Always pass 'train' as split - since we only want this dataset to be
    # returned when using this dataloader - use separate dataloader for valid
    dataset = datasets.load_dataset('json',
                                    data_files=glob.glob(data_path),
                                    split='train',
                                    streaming=False)

    shards = []
    # Process shards in parallel
    with Pool(processes=conf.num_proc) as p:
        for shard in p.imap(lambda x: process_shard(dataset, x, conf),
                            range(conf.num_proc)):
            shards.append(shard)
    dataset = datasets.interleave_datasets(shards)
    # dataset = dataset.shuffle(buffer_size=10000, seed=13)
    if shuffle:
        dataset = dataset.shuffle(seed=conf.seed)
    # collate_fn can be used if we want to return a batch_size list of
    # dicts instead of a dict of concatenated tensors
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=shuffle,
                                             batch_size=conf.batch_size,
                                             # collate_fn=lambda x: x
                                             )
    return dataloader


def bioasq_submission(pmids, labels, name_id_mapping, system='ediranknn'):
    # labels have MeSH names not ids
    # Format for submission
    # {"username":"your_username", "password":"your_password", "system":"your_system",
    # "documents": [{"labels":["label1", "label2",...,"labelN"], "pmid": 22511223},
    #                   {"labels":["label1", "label2",...,"labelM"],"pmid":22511224},
    #                                             .
    #                                             .
    #                   {"labels":["label1", "label2",..., "labelK"], "pmid":22511225}]}

    assert len(pmids) == len(labels)
    encoder = MeSHEncoder(name_id_mapping)

    sub = dict()
    sub['username'] = os.environ['BIOASQ_USERNAME']
    sub['password'] = os.environ['BIOASQ_PASSWORD']
    sub['system'] = system
    docs = []
    for pmid, lbls in zip(pmids, labels):
        doc = dict()
        # print(pmid, lbls)
        meshidxs = encoder.encode(lbls)
        doc['labels'] = meshidxs
        doc['pmid'] = pmid
        docs.append(doc)
    sub['documents'] = docs

    return json.dumps(sub, indent=4)


# # S is the BOS token
# BOS = 'S'
# CHARS = 'abcdefghij01%s ' % BOS
# CHARS_TO_IDX = {c:i for c, i in zip(CHARS, range(len(CHARS)))}
# IDX_TO_CHARS = {i:c for c, i in zip(CHARS, range(len(CHARS)))}
#
#
# class SynthMultiLabelDataset(torch.utils.data.Dataset):
#
#     """Docstring for SynthMultiLabelDataset. """
#
#     def __init__(self, filename, transform=None):
#         self.filename = filename
#         self.transform = transform
#         self.load_data()
#
#     def load_data(self):
#         with open(self.filename, 'r') as f:
#             X, y = [], []
#             for line in f:
#                 xx, yy = line.rstrip().split('\t')
#                 yy = '%s%s' % (BOS, yy)
#                 xx = np.array([CHARS_TO_IDX[c] for c in xx], dtype=np.int32)
#                 yy = np.array([CHARS_TO_IDX[c] for c in yy], dtype=np.int32)
#
#                 X.append(xx)
#                 y.append(yy)
#         self.X = X
#         self.y = y
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         xx = self.X[idx]
#         yy = self.y[idx]
#         sample = {'text': xx,
#                   'text_no_blank': np.arange(len(xx), dtype=np.int32),
#                   'labels': yy}
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample

# S is the BOS token
BOS = '$'
LABELS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'\
         'αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩάόέή'


class SynthMultiLabelDataset(torch.utils.data.Dataset):

    """Docstring for SynthMultiLabelDataset. """

    def __init__(self, filename, transform=None):
        self.filename = filename
        self.transform = transform
        self.load_data()

    @property
    def num_labels(self):
        filename = os.path.basename(self.filename)
        return int(filename.split('-')[1])

    def load_data(self):

        with open(self.filename, 'r') as f:
            X, y = [], []

            self.labels = '%s01%s ' % (LABELS[:self.num_labels], BOS)
            self.chars_to_idx = {c:i for c, i in zip(self.labels, range(len(self.labels)))}
            self.idx_to_chars = {i:c for c, i in zip(self.labels, range(len(self.labels)))}

            for line in f:
                xx, yy = line.rstrip().split('\t')
                # yy = '%s%s' % (BOS, yy)
                # yy = np.array([self.chars_to_idx[c] for c in yy], dtype=np.int32)
                xx = np.array([self.chars_to_idx[c] for c in xx], dtype=np.int32)
                yy = np.array([0 if y == '0' else 1 for y in yy], dtype=np.int32)

                X.append(xx)
                y.append(yy)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        xx = self.X[idx]
        yy = self.y[idx]
        sample = {'text': xx,
                  'text_no_blank': np.arange(len(xx), dtype=np.int32),
                  'labels': yy}
        if self.transform:
            sample = self.transform(sample)

        return sample


class MeSHMultiLabelDataset(torch.utils.data.IterableDataset):

    """Docstring for MeSHMultiLabelDataset. """

    def __init__(self, filename, transform=None):
        self.filename = filename
        self.transform = transform

    def load_data(self):
        with open(self.filename, 'r') as f:
            for line in f:
                items = json.loads(line.rstrip())
                yield items

    def __iter__(self):
        if self.transform:
            return self.transform(next(self.load_data()))
        else:
            return next(ShufflerIterDataPipe(self.load_data()))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        xx = self.X[idx]
        yy = self.y[idx]
        sample = {'text': xx,
                  'text_no_blank': np.arange(len(xx), dtype=np.int32),
                  'labels': yy}
        if self.transform:
            sample = self.transform(sample)

        return sample


class OpenImagesDataset(torch.utils.data.Dataset):

    def __init__(self, annotation_file, img_dir, num_labels, transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.num_labels = num_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        imagefile = self.img_labels.iloc[idx].imagefile

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx].imagefile + '.jpg')
        try:
            # image = read_image(img_path, ImageReadMode.RGB)
            image = Image.open(img_path).convert('RGB')
        except Exception:
            raise ValueError('Error on %s' % imagefile)
        if self.transform:
            image = self.transform(image)

        label_idxs = pd.eval(self.img_labels.iloc[idx].label_idxs)
        labels = np.zeros(self.num_labels, dtype=np.int32)
        labels[label_idxs] = 1

        return imagefile, image, labels
