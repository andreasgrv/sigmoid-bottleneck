import os
import tqdm
import time
import math
import torch
import random
import argparse
import numpy as np
import scipy as sp
import gurobipy as gp

from gurobipy import GRB
from functools import partial
from multiprocessing import Pool
from itertools import combinations

from spmlbl.verify import lp_chebyshev, search_sgd


W = None
b = None


class ArgmaxableSubsetVerifier(object):
    """Algorithm to detect what subsets of labels can be predicted
    Accepts the sigmoid layer parameters W and b as input."""
    def __init__(self, W, b=None, num_processes=None, check_rank=True):
        super().__init__()
        self.W = W
        self.n, self.d = self.W.shape
        self.num_processes = num_processes or int(os.environ.get('MLBL_NUM_PROC', 1))
        self.b = b
        self.check_rank = check_rank

    def __call__(self,
                 pos_idxs_list=None,
                 algorithm=None,
                 num_processes=None,
                 **kwargs):
        global W, b

        num_processes = num_processes or self.num_processes

        # List of classes to return
        results = []

        # We assume our vectors span the column space (d dimension)
        if self.check_rank:
            rank = np.linalg.matrix_rank(self.W)
            assert rank == min(self.d, self.n), 'Rank=%d, d=%d' % (rank, self.d)

        is_feasible = partial(subset_is_feasible,
                              shape=(self.n, self.d),
                              dtype=self.W.dtype,
                              algorithm=algorithm,
                              **kwargs)

        # Set global variables - they will be visible in threads
        if algorithm == search_sgd:
            W = torch.tensor(self.W, dtype=torch.float64, requires_grad=False)
            if b is not None:
                b = torch.tensor(self.b, dtype=torch.float64, requires_grad=False)
        else:
            W = self.W
            b = self.b

        # We freeze the pytorch variables by requiring grad=False
        tW = torch.tensor(W, requires_grad=False, dtype=torch.float32)
        if b is not None:
            tb = torch.tensor(b.ravel(), requires_grad=False, dtype=torch.float32)

        # Use multiprocessing to parallelise search across weight vectors
        # Each process checks if a particular class has stolen probability
        with Pool(processes=num_processes) as p:
            with tqdm.tqdm(total=len(pos_idxs_list), desc='Verifying subsets..') as pbar:
                for i, result in enumerate(p.imap_unordered(is_feasible, pos_idxs_list)):
                    results.append(result)
                    pbar.update()

        return results


def subset_is_feasible(pos_idxs,
                       shape,
                       dtype,
                       algorithm=None,
                       **kwargs):

    start_time = time.time()
    result = dict()

    n, d = shape

    algorithm = algorithm or lp_chebyshev

    result = algorithm(pos_idxs, W=W, b=b, **kwargs)

    result['pos_idxs'] = pos_idxs
    # There are many reasons the LP may fail, in which case
    # we want to avoid interpretting failure as infeasibility.
    result['run_successful'] = True
    # Check if we can validate the result
    # If not this points to numerical accuracy error / tolerance issues
    result['valid_successful'] = True

    # Verify Solution if we found one
    if result['is_feasible']:
        if b is not None:
            act = W.dot(result['point']).reshape(-1, 1) + b
        else:
            act = W.dot(result['point'])
        preds = act.ravel() > 0.
        on_idxs = preds.nonzero()[0].tolist()
        # If disagreement, we ran into numerical issues
        # e.g. number is within tolerance of threshold
        if set(on_idxs) != set(pos_idxs):
            result['is_feasible'] = False
            result['radius'] = None
            result['run_successful'] = False
            result['valid_successful'] = False
            # Add a code to identify this
            result['status'] = -1
            # print(result)
            # print(on_idxs, pos_idxs)
            # print(act[list(pos_idxs)])
            # raise AssertionError('Numpy != Gurobi result, numerical issues likely.')
    else:
        if result['status'] > 5:
            result['run_successful'] = False

    end_time = time.time()
    result['time_taken'] = end_time - start_time

    return result
