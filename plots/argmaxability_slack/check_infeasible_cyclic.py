import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from collections import defaultdict

from spmlbl.verifier import ArgmaxableSubsetVerifier
from spmlbl.verify import lp_chebyshev, search_sgd
from spmlbl.modules import gale, vander


def generate_card_samples(cardinality, num_labels, num_samples=10):
    samples = set()
    while len(samples) < num_samples:
        idxs = np.random.choice(num_labels, cardinality, replace=False)
        idxs = tuple(sorted(idxs.tolist()))
        samples.add(idxs)
    return samples


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--N', type=int, default=20000,
                        help='Number of vertices.')
    parser.add_argument('--D', type=int, default=20,
                        help='Number of dimensions.')
    parser.add_argument('--seed', type=int, default=12,
                        help='Random seed.')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to check.')
    parser.add_argument('--parametrisation', type=str, default='vander',
                        choices=['vander', 'gale'],
                        help='Which parametrisation to use.')
    parser.add_argument('--cardinality', type=int, default=None,
                        help='Cardinality of samples.')
    parser.add_argument('--slack-dims', type=int, default=0,
                        help='Number of slack dimensions.')

    args = parser.parse_args()

    # All labels of cardinality card should be feasible
    if not args.cardinality:
        card = args.D // 2
    else:
        card = args.cardinality

    if args.parametrisation == 'gale':
        # assert (args.D % 2) == 0, 'Gale parametrisation not guaranteed to give alt matroid when d is even'
        W = gale(args.N, args.D)
    else:
        W = vander(args.N, args.D)

    print('########################################################')
    print('###          Analysing Cyclic *C(%d, %d)*            ###' % (args.N, args.D))
    print('########################################################')

    if args.slack_dims > 0:

        np.random.seed(args.seed)
        W_full = np.random.uniform(-1., .1, (args.N, args.slack_dims)) / np.sqrt(args.N)

        W = np.hstack([W, W_full])

    np.random.seed(args.seed)

    v = ArgmaxableSubsetVerifier(W=W, num_processes=int(os.environ['MLBL_NUM_PROC']))

    num_labels = math.comb(args.N, card)

    print('### There are %.2e labels of cardinality %d' % (num_labels, card))

    # Make sure we aren't trying more samples than possible
    num_samples = min(args.num_samples, num_labels)
    print('### Sampling %d to compute feasible/infeasible regions...' % (num_samples))

    samples = generate_card_samples(card, args.N, num_samples=num_samples)

    res = v(samples, algorithm=lp_chebyshev, lb=-1e4, ub=1e4)

    num_feasible, count = 0, 0
    stati = defaultdict(int)
    rads = []
    for r in res:
        stati[r['status']] += 1
        if r['is_feasible']:
            num_feasible +=1
            rads.append(r['radius'])
        else:
            rads.append(1e-10)
        count += 1

    rads = np.array(rads)

    df = pd.DataFrame()

    df['N'] = [args.N]
    df['D'] = [args.D]
    df['slack_dims'] = [args.slack_dims]
    df['parametrisation'] = [args.parametrisation]
    df['num_samples'] = [num_samples]
    df['num_feasible'] = [num_feasible]
    df['num_infeasible'] = [num_samples - num_feasible]
    df['rad_min'] = [rads.min()]
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        df['rad_p_%d' % p] = [np.percentile(rads, p)]
    df['rad_max'] = [rads.max()]
    df['status_counter'] = [{k: v for k, v in sorted(stati.items())}]

    print()
    print('Feasible:', num_feasible, '/', count)
    print('Radius min: %.2E' % rads.min())
    print('Radius max: %.2E' % rads.max())
    print('Percentiles', np.percentile(rads, [1, 5, 10, 25, 50, 75, 90, 95, 99]))
    print(df.to_csv(index=False, float_format='%.2e'))

    # ax = sns.displot(rads, kind='kde')
    # ax.set(xscale='log')
    # plt.show()
