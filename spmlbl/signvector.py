import numpy as np
import itertools

from spmlbl.verifier import ArgmaxableSubsetVerifier
from spmlbl.math import orthogonal_complement, linear_dependence


class SignVector(object):
    """docstring for SignVector"""
    VOCAB = set(['+', '-', '0'])
    def __init__(self, sign_vector):
        super().__init__()
        self.sign_vector = tuple(sign_vector)
        assert len(set(sign_vector).union(SignVector.VOCAB)) == 3

    def __hash__(self):
        return hash(''.join(self.sign_vector))

    def __str__(self):
        return ''.join(self.sign_vector)

    def __repr__(self):
        return ''.join(self.sign_vector)

    def _idx_match(self, s, match):
        idxs = []
        for i, c in enumerate(self.sign_vector):
            if c == match:
                idxs.append(i)
        return idxs

    def __neg__(self):
        neg = []
        for c in self.sign_vector:
            if c == '-':
                neg.append('+')
            elif c == '+':
                neg.append('-')
            else:
                neg.append('0')
        return SignVector(neg)

    def __eq__(self, other):
        return self.sign_vector == other.sign_vector

    def __le__(self, other):
        assert len(self.sign_vector) == len(other.sign_vector)
        pos_idxs_match = set(self.pos_idxs).issubset(set(other.pos_idxs))
        neg_idxs_match = set(self.neg_idxs).issubset(set(other.neg_idxs))
        return pos_idxs_match and neg_idxs_match

    def __lt__(self, other):
        assert len(self.sign_vector) == len(other.sign_vector)
        pos_idxs_match = set(self.pos_idxs).issubset(set(other.pos_idxs))
        neg_idxs_match = set(self.neg_idxs).issubset(set(other.neg_idxs))
        return self != other and pos_idxs_match and neg_idxs_match

    # Composition
    def __call__(self, other):
        assert len(self.sign_vector) == len(other.sign_vector)
        assert self.is_compatible(other)
        composition = []
        for l, r in zip(self.sign_vector, other.sign_vector):
            if l != '0':
                composition.append(l)
            else:
                composition.append(r)
        return SignVector(composition)

    @property
    def pos_idxs(self):
        return self._idx_match(self.sign_vector, '+')

    @property
    def neg_idxs(self):
        return self._idx_match(self.sign_vector, '-')

    @property
    def zero_idxs(self):
        return self._idx_match(self.sign_vector, '0')

    def is_compatible(self, other):
        assert len(self.sign_vector) == len(other.sign_vector)
        compatible = True
        for l, r in zip(self.sign_vector, other.sign_vector):
            if ((l, r) == ('+', '-')) or ((l, r) == ('-', '+')):
                compatible = False
                break
        return compatible

    @classmethod
    def from_numpy(cl, v):
        assert len(v.shape) == 1
        signs = []
        for e in v:
            if np.abs(e) < 1e-12:
                signs.append('0')
            elif e > .0:
                signs.append('+')
            else:
                signs.append('-')
        return cl(signs)


class SignVectors(object):
    """docstring for SignVectors"""
    def __init__(self, vectors):
        super().__init__()
        self.vectors = list(vectors)

    def __len__(self):
        return len(self.vectors)

    def __iter__(self):
        for v in self.vectors:
            yield v

    def minimal_support(self):
        candidates = set(self.vectors)
        drop = []
        for candidate in candidates:
            for other in candidates:
                if (candidate < other):
                    drop.append(other)
        return SignVectors(candidates - set(drop))

    @classmethod
    def all_subsets(cl, n):
        subs = tuple(itertools.product(['+', '-'], repeat=n))
        return cl([SignVector(v) for v in subs])

    @classmethod
    def all_feasible(cl, W, b=None):

        feas_vectors, _ = compute_sign_vectors(W)

        return cl(feas_vectors)

    @classmethod
    def all_infeasible(cl, W, b=None):
        N, D = W.shape

        W_orth = orthogonal_complement(W)

        # Feasible vectors of the orth complement
        # are infeasible in W
        feas_vectors, _ = compute_sign_vectors(W)

        return cl(feas_vectors)


def compute_sign_vectors(W):
    N, D = W.shape
    ver = ArgmaxableSubsetVerifier(W, check_rank=False)

    all_subsets = [[i for i, b in enumerate(s) if b]
                   for s in itertools.product([0, 1], repeat=N)]
    res = ver(all_subsets)
    # print(res)

    feas, infs = [], []
    for r in res:
        s = ['-']*N
        for idx in r['pos_idxs']:
            s[idx] = '+'
        sign_vec = ''.join(s)
        if r['is_feasible']:
            feas.append(SignVector(sign_vec))
        else:
            infs.append(SignVector(sign_vec))
    return feas, infs


def compute_potential_circuits(W):
    # NOTE: This function is a work in progress
    # unsure if I can stop at D+1.
    # NOTE: Do not use this for large W - we cannot afford to
    # check all combinations.. this does not scale
    N, D = W.shape

    assert D < 10, 'This function is not meant to handle large W'

    candidates = []
    # Any D+1 number of vectors in general position
    for k in range(2, D+2):
        for idxs in itertools.combinations(range(N), k):
            idxs = list(idxs)
            W_sub = W[idxs]
            res = linear_dependence(W_sub)
            if res['is_dependence']:
                coeff = np.zeros(N)
                coeff[idxs] = res['solution']
                sv = SignVector.from_numpy(coeff)
                candidates.append(sv)
                candidates.append(-sv)
    return SignVectors(candidates)
