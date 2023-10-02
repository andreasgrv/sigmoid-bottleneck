import math
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


def test_exact_k_sparse(N=10, K=3):

    W = vander(N+1, K+1)

    # ax = plt.figure().add_subplot(projection='3d')
    # for w in W:
    #     ax.scatter(*w)
    # plt.show()

    ver = ArgmaxableSubsetVerifier(W)
    samples = [list(c) for cc in range(0, N+2) for c in combinations(range(N+1), cc)]
    print(len(samples))
    res = ver(samples)

    total_feas = 0
    labels = []
    for each in sorted(res, key=lambda x:x['pos_idxs']):
        if each['is_feasible']:
            s = [-1,] * (N+1)
            for idx in each['pos_idxs']:
                s[idx] = 1

            kprod = [int(-s[i]*s[i+1] > 0) for i in range(N)]
            ksum = sum(kprod)
            assert ksum <= K
            labels.append(ksum)

            s = ''.join(map(str, kprod))
            print('\t%s %d' % (s, ksum))
            total_feas += 1
    # print(total_feas, num_feasible_labels(N, D))
    # assert total_feas == num_feasible_labels(N, D)
    cc = Counter(labels)
    for k, v in cc.items():
        assert v == 2 * math.comb(N, k)
        print(k, v // 2)
    

    # print('W orth results...')
    # W_orth = orthogonal_complement(W)
    # ver = ArgmaxableSubsetVerifier(W_orth)
    # res = ver(samples)
    # total_feas = 0
    # for each in sorted(res, key=lambda x:x['pos_idxs']):
    #     if each['is_feasible']:
    #         s = [0,] * N
    #         for idx in each['pos_idxs']:
    #             s[idx] = 1
    #         s = ''.join(map(str, s))
    #         nshifts = num_bit_shifts(s)
    #         assert nshifts >= D
    #         print('\t%s %d' % (s, nshifts))
    #         total_feas += 1
    # print(total_feas, num_feasible_labels(N, N-D))
    # assert total_feas == num_feasible_labels(N, N-D)


if __name__ == "__main__":
    test_exact_k_sparse(8, 1)
    test_exact_k_sparse(8, 2)
    test_exact_k_sparse(8, 4)
    test_exact_k_sparse(8, 5)
    test_exact_k_sparse(8, 6)
    test_exact_k_sparse(8, 7)
    test_exact_k_sparse(10, 3)
    test_exact_k_sparse(10, 4)
