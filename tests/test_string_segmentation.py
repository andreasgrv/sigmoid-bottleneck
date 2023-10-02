import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

from spmlbl.verifier import ArgmaxableSubsetVerifier
from spmlbl.verify import num_feasible_labels
from spmlbl.modules import vander
from spmlbl.math import orthogonal_complement


def num_bit_shifts(s):
    num = 0
    for i in range(len(s)-1):
        if s[i] != s[i+1]:
            num += 1
    return num


def test_string_segmentation(N=10, D=5):

    W = vander(N, D)

    ver = ArgmaxableSubsetVerifier(W)
    samples = [list(c) for cc in range(0, N+1) for c in combinations(range(N), cc)]
    print(len(samples))
    res = ver(samples)

    total_feas = 0
    labels = []
    for each in sorted(res, key=lambda x:x['pos_idxs']):
        if each['is_feasible']:
            s = [0,] * N
            for idx in each['pos_idxs']:
                s[idx] = 1
            labels.append(sum(s))
            s = ''.join(map(str, s))
            nshifts = num_bit_shifts(s)
            assert nshifts < D
            print('\t%s %d' % (s, nshifts))
            total_feas += 1
    print(total_feas, num_feasible_labels(N, D))
    assert total_feas == num_feasible_labels(N, D)
    print(Counter(labels))
    

    print('W orth results...')
    W_orth = orthogonal_complement(W)
    ver = ArgmaxableSubsetVerifier(W_orth)
    res = ver(samples)
    total_feas = 0
    for each in sorted(res, key=lambda x:x['pos_idxs']):
        if each['is_feasible']:
            s = [0,] * N
            for idx in each['pos_idxs']:
                s[idx] = 1
            s = ''.join(map(str, s))
            nshifts = num_bit_shifts(s)
            assert nshifts >= D
            print('\t%s %d' % (s, nshifts))
            total_feas += 1
    print(total_feas, num_feasible_labels(N, N-D))
    assert total_feas == num_feasible_labels(N, N-D)


if __name__ == "__main__":
    test_string_segmentation(8, 1)
    test_string_segmentation(8, 2)
    test_string_segmentation(8, 3)
    test_string_segmentation(8, 4)
    test_string_segmentation(8, 5)
    test_string_segmentation(8, 6)
    test_string_segmentation(8, 7)
    test_string_segmentation(10, 3)
    test_string_segmentation(10, 4)
    test_string_segmentation(10, 5)
    test_string_segmentation(10, 6)
