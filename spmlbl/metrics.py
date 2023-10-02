import torch
import numpy as np
import sklearn.metrics as skmetrics

from collections import deque
from mlconf import Blueprint
from sklearn.metrics import f1_score
from spmlbl.utils import Timer


def to_flat_dict(d, delim='.', copy=True):
    """TLDR;
    While there are entries in the dictionary that have a dict as a value:
        pop them at the outer level and create a delimitted path as a key, eg:
            {'a': {'b': {'c': 0}}} -> {'a.b': {'c': 0}}
            # by same process
            {'a.b': {'c': 0}} -> {'a.b.c': 0}
    """
    flat = dict(d) if copy else d
    # we copy the keys since we are modifying the dict in place
    # we reverse to retain order
    incomplete = list(flat)[::-1]
    while(incomplete):
        k = incomplete.pop()
        if isinstance(flat[k], dict) or isinstance(flat[k], Blueprint):
            val = flat.pop(k)
            # Reverse to retain order since we are popping
            for subk, subv in tuple(val.items())[::-1]:
                new_key = delim.join((k, subk))
                flat[new_key] = subv
                incomplete.append(new_key)
        # elif isinstance(flat[k], (list, tuple)):
        #     val = flat.pop(k)
        #     for subk, subv in tuple(enumerate(val))[::-1]:
        #         new_key = delim.join((k, str(subk)))
        #         flat[new_key] = subv
        #         incomplete.append(new_key)
        else:
            # Re-insert entry to retain order in dict
            # Python guarantees dict ordered by insertion starting in 3.6.
            val = flat.pop(k)
            flat[k] = val
    return flat


class MetricMonitor(object):
    SEP = '\t'
    def __init__(self, metrics, filename):
        self.metrics = metrics
        self.filename = filename

    def write_header(self):
        s = self.SEP.join(m
                          for m in to_flat_dict(self.metrics))
        with open(self.filename, 'w') as f:
            f.write('%s\n' % s)

    def append(self):
        # assert len(row) == len(self.metrics)
        s = '\t'.join(v.str_value
                      for v in to_flat_dict(self.metrics).values())
        with open(self.filename, 'a') as f:
            f.write('%s\n' % s)


class Time(object):

    def __init__(self, label='Time taken'):
        self.label = label
        self.value = 0

    def __call__(self, value):
        self.value = value

    def reset(self):
        self.value = 0

    @property
    def legend(self):
        return '%s: %s' % (self.label, self.str_value)

    @property
    def str_value(self):
        return Timer.str_format(self.value)


class Average(object):

    def __init__(self, window_size=50, label='Avg'):
        self.window_size = window_size
        self.label = label
        self.buffer = deque()

    def __call__(self, value):
        if len(self.buffer) == self.window_size:
            self.buffer.pop()
        self.buffer.appendleft(value)

    def reset(self):
        self.buffer = deque()

    @property
    def value(self):
        length = len(self.buffer)
        if length:
            return sum(self.buffer) / len(self.buffer)
        else:
            return -1

    @property
    def legend(self):
        if self.value < .01:
            return '%s: %.2e' % (self.label, self.value)
        else:
            return '%s: %.4f' % (self.label, self.value)

    @property
    def str_value(self):
        return '%.6f' % self.value


class SKLearnMetricAvg(object):

    def __init__(self, label, sklearn_metric, window_size=50, **metric_params):
        self.window_size = window_size
        self.label = label
        self.sklearn_metric = getattr(skmetrics, sklearn_metric)
        self.metric_params = metric_params
        self.buffer = deque()

    def __call__(self, gold, pred):
        value = self.sklearn_metric(gold, pred, **self.metric_params)
        if len(self.buffer) == self.window_size:
            self.buffer.pop()
        self.buffer.appendleft(value)

    def reset(self):
        self.buffer = deque()

    @property
    def value(self):
        length = len(self.buffer)
        if length:
            return (sum(self.buffer) / len(self.buffer)) * 100.
        else:
            return -1

    @property
    def legend(self):
        if self.value < .01:
            return '%s: %.2e' % (self.label, self.value)
        else:
            return '%s: %.4f' % (self.label, self.value)

    @property
    def str_value(self):
        return '%.6f' % self.value


class ExactAccuracy(object):

    def __init__(self, window_size=50, label='Exact Acc'):
        self.window_size = window_size
        self.label = label
        self.matches = deque()
        self.total = deque()

    def __call__(self, gold, pred):
        batch_size, num_preds = pred.shape
        exact_match = gold.eq(pred).sum(axis=1) == num_preds

        if len(self.matches) == self.window_size:
            self.matches.pop()
            self.total.pop()

        self.matches.appendleft(exact_match.sum().item())
        self.total.appendleft(batch_size)

    def reset(self):
        self.matches = deque()
        self.total = deque()

    @property
    def value(self):
        length = len(self.matches)
        if length:
            return sum(self.matches) / sum(self.total)
        else:
            return 0.

    @property
    def legend(self):
        return '%s: %.2f%%' % (self.label, self.value * 100)

    @property
    def str_value(self):
        return '%.6f' % (self.value * 100)


class RecallAtK(object):

    def __init__(self, window_size=50, k=10):
        self.window_size = window_size
        self.k = k
        self.label = 'Rec@%d' % k
        self.buffer = deque()

    def __call__(self, gold, logits):
        assert gold.shape[0] == logits.shape[0]
        bs, n = gold.shape
        assert n <= logits.shape[1]

        tp = 0
        total = 0
        for g, l in zip(gold, logits):
            # Revert multi-hot to label indices
            g = torch.nonzero(g).view(-1)
            # Default argsort sorts smallest to largest
            # so we negate values to sort largest to smallest
            # top_k = np.argsort(-l)[:self.k]
            top_k = torch.topk(l, self.k).indices
            top_k = set(i for i in top_k.tolist())
            g_set = set(i for i in g.tolist())

            num = len(top_k.intersection(g_set))
            denom = len(g_set)
            recall = num / denom if denom != 0 else 0
            self.buffer.appendleft(recall)

        while len(self.buffer) > self.window_size:
            self.buffer.pop()

    def reset(self):
        self.buffer = deque()

    @property
    def value(self):
        length = len(self.buffer)
        if length > 0:
            sum_num = sum(self.buffer)
            prc = sum_num / length
            return prc
        else:
            return 0.

    @property
    def legend(self):
        return '%s: %.2f%%' % (self.label, self.value * 100)

    @property
    def str_value(self):
        return '%.4f' % (self.value * 100)


class PrecisionAtK(object):

    def __init__(self, window_size=50, k=10):
        self.window_size = window_size
        self.k = k
        self.label = 'Prec@%d' % k
        self.buffer = deque()

    def __call__(self, gold, logits):
        assert gold.shape[0] == logits.shape[0]
        bs, n = gold.shape
        assert n <= logits.shape[1]

        tp = 0
        total = 0
        for g, l in zip(gold, logits):
            # Revert multi-hot to label indices
            g = torch.nonzero(g).view(-1)
            # Default argsort sorts smallest to largest
            # so we negate values to sort largest to smallest
            # top_k = np.argsort(-l)[:self.k]
            top_k = torch.topk(l, self.k).indices
            top_k = set(i for i in top_k.tolist())
            g_set = set(i for i in g.tolist())

            num = len(top_k.intersection(g_set))
            denom = self.k
            prec = num / denom if denom != 0 else 0
            self.buffer.appendleft(prec)

        while len(self.buffer) > self.window_size:
            self.buffer.pop()

    def reset(self):
        self.buffer = deque()

    @property
    def value(self):
        length = len(self.buffer)
        if length > 0:
            sum_num = sum(self.buffer)
            prc = sum_num / length
            return prc
        else:
            return 0.

    @property
    def legend(self):
        return '%s: %.2f%%' % (self.label, self.value * 100)

    @property
    def str_value(self):
        return '%.4f' % (self.value * 100)


class MicroF1(object):

    def __init__(self, window_size=50, label='Micro F1'):
        self.window_size = window_size
        self.label = label
        self.tp_buffer = deque()
        self.fp_buffer = deque()
        self.fn_buffer = deque()

    def __call__(self, gold, pred):
        tp = ((gold == 1) & (pred == 1)).sum()
        fp = ((gold == 0) & (pred == 1)).sum()
        fn = ((gold == 1) & (pred == 0)).sum()
        if len(self.tp_buffer) == self.window_size:
            self.tp_buffer.pop()
            self.fp_buffer.pop()
            self.fn_buffer.pop()
        self.tp_buffer.appendleft(tp)
        self.fp_buffer.appendleft(fp)
        self.fn_buffer.appendleft(fn)

    def reset(self):
        self.tp_buffer = deque()
        self.fp_buffer = deque()
        self.fn_buffer = deque()

    @property
    def value(self):
        length = len(self.tp_buffer)
        if length:
            sum_tp = sum(self.tp_buffer)
            sum_fp = sum(self.fp_buffer)
            sum_fn = sum(self.fn_buffer)
            prc = sum_tp / max(sum_tp + sum_fp, 1)
            rec = sum_tp / max(sum_tp + sum_fn, 1)
            if (prc * rec) == 0:
                return 0.
            else:
                return 2 * prc * rec / (prc + rec)
        else:
            return 0.

    @property
    def legend(self):
        return '%s: %.2f%%' % (self.label, self.value * 100)

    @property
    def str_value(self):
        return '%.4f' % (self.value * 100)


def f1(pred, gold):
    tp = ((gold == 1) & (pred == 1)).sum()
    fp = ((gold == 0) & (pred == 1)).sum()
    fn = ((gold == 1) & (pred == 0)).sum()
    prc = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if (prc * rec) == 0:
        return 0.
    else:
        return 2 * prc * rec / (prc + rec)


def f1_counts(cc):
    prc = cc['tp'] / max(cc['tp'] + cc['fp'], 1)
    rec = cc['tp'] / max(cc['tp'] + cc['fn'], 1)
    if (prc * rec) == 0:
        return 0.
    else:
        return 2 * prc * rec / (prc + rec)


def counts(pred, gold):
    tp = ((gold == 1) & (pred == 1)).sum()
    fp = ((gold == 0) & (pred == 1)).sum()
    fn = ((gold == 1) & (pred == 0)).sum()
    return dict(tp=tp, fp=fp, fn=fn)


def argmaxf1(k, t, values, logits, gold):
    n, d = logits.shape

    tt = np.copy(t)
    tc = np.copy(t)
    preds = logits > tc

    cc = counts(preds, gold)
    # best_f = f1_score(preds, gold, average='micro')
    # best_f = f1(preds, gold)
    best_f = f1_counts(cc)

    # Remove the counts of the k column

    # STEP = 1
    # for i in range(0, n + 2 - STEP, STEP):
    for i in range(n + 1):

        target_cc = counts(preds[:, k], gold[:, k])
        for key in cc:
            cc[key] = cc[key] - target_cc[key]

        # tc[k] = (values[i, k] + values[i+STEP, k]) / 2.
        tc[k] = (values[i, k] + values[i+1, k]) / 2.
        # preds = logits > tc
        pred = logits[:, k] > tc[k]
        preds[:, k] = pred

        target_cc = counts(preds[:, k], gold[:, k])
        for key in cc:
            cc[key] = cc[key] + target_cc[key]

        # fs = f1_score(preds, gold, average='micro')
        fs = f1_counts(cc)
        if fs > best_f:
            best_f = fs
            tt[k] = tc[k]
    print(tt[k], best_f)
    return tt[k]


# Algorithm for choosing thresholds for multi-label classifiers
# the thresholds maximise micro-f1.
# http://pralab.diee.unica.it/sites/default/files/pillai_PR2013_Thresholding_0.pdf
def choose_thresholds(logits, gold):
    # n validation examples
    # d labels
    n, d = logits.shape
    NUM_SEEN_TO_TUNE = 10

    # print(f1(logits > 0., gold))
    # print(f1(logits > -.5, gold))
    # print(f1(logits > -1., gold))
    # print(f1(logits > -1.5, gold))

    assert np.all(logits.shape == gold.shape)

    values = np.sort(logits, axis=0)

    values = np.vstack([
        np.ones((1, d)) * -np.inf,
        values,
        np.ones((1, d)) * np.inf
    ])

    # t = values[0]
    t = np.zeros(d)
    stats = gold.sum(axis=0)

    # print((stats > NUM_SEEN_TO_TUNE).sum())
    # print((stats > 20).sum())
    # print((stats > 50).sum())

    patience = 1
    while True:

        updated = False
        for k in range(d):
            if stats[k] > NUM_SEEN_TO_TUNE:
                # Find argmax for threshold for this k
                tk = argmaxf1(k, t, values, logits, gold)
                if t[k] != tk and (not np.isinf(tk)):
                    t[k] = tk
                    updated = True
        patience -= 1
        if updated is False or patience == 0:
            break
    preds = logits > t
    print('Final F1: %.4f' % f1(preds, gold))
    return t
