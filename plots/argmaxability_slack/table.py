import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from collections import defaultdict


if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument('--data', required=True, type=str,
                        help='The csv file to plot')
    parser.add_argument('--slack-dims', default=0, type=int,
                        help='How many slack dims.')
    parser.add_argument('--slack-source', default='None', type=str,
                        help='Source of slack.')

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    df.slack_source = df.slack_source.replace({np.nan: 'None'})

    df = df[(df.slack_dims == args.slack_dims) & (df.slack_source == args.slack_source)]

    table = defaultdict(dict)

    for N, D, ns, num_feas, num_inf in zip(df.N, df.D, df.num_samples, df.num_feasible, df.num_infeasible):
        assert (num_feas + num_inf) == ns
        table[N][D] = '%.3f%%' % (100 * (num_feas / ns))
        # table[N][D] = num_inf

    table = {k: [vv for kk, vv in sorted(v.items(), key=lambda x:x[0])]
             for k, v in table.items()}
    # table = pd.DataFrame(table, index=sorted(df.D.unique())).to_latex()
    table = pd.DataFrame(table, index=sorted(df.D.unique())).to_markdown()
    print('Slack: %d dims (%s) - number of feasible regions for C(N, D)' % (args.slack_dims, args.slack_source))
    print(table)
