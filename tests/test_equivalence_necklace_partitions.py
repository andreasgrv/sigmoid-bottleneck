import torch
import numpy as np

from spmlbl.verify import num_feasible_labels, num_necklace_partitions_alt_colour


def test_equivalence(N=20, K=8):
    D = 2*K + 1
    num_feas = num_feasible_labels(N, D)
    num_neck = num_necklace_partitions_alt_colour(N, K)
    assert num_feas == num_neck


if __name__ == "__main__":

    n = 20
    for k in range(10):
        test_equivalence(n, k)
