import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from argparse import ArgumentParser


if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument('--data', required=True, type=str,
                        help='The csv file to plot')
    parser.add_argument('--prop', required=True, type=str,
                        help='Which property to plot')
    parser.add_argument('--slack-dims', default=0, type=int,
                        help='How many slack dims.')
    parser.add_argument('--slack-source', default='None', type=str,
                        help='Source of slack.')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save the image.')

    args = parser.parse_args()

    UNARGMAXABLE = 1e-10
    df = pd.read_csv(args.data)

    df.slack_source = df.slack_source.replace({np.nan: 'None'})

    df = df[(df.slack_dims == args.slack_dims) & (df.slack_source == args.slack_source)]
    print(df)

    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(12, 7))

    for d in sorted(df.D.unique()):
        d_slice = df[df.D == d]
        # ax.plot(np.arange(len(d_slice)), d_slice[args.prop], 'o-', label='k=%d' % (d//2))
        prop = d_slice[args.prop]
        xx = d_slice['N'][prop > UNARGMAXABLE].tolist()
        prop = prop[prop > UNARGMAXABLE]
        ax.plot(xx, prop, 'o-', label='k=%d' % (d//2))
        # perc_infeasible = (1. - d_slice['num_feasible'] / d_slice['num_samples']) * 100.
        # ax2.plot(xx, perc_infeasible, 'o-', label='k=%d' % (d//2))
        # ax.plot(d_slice.N, d_slice[args.prop], 'o-', label=d//2)
    ax.set_ylim([5e-9, 1e4])
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    # ax.set_xticklabels([' '] + d_slice.N.tolist())
    ax.set_xlabel('# of Labels (n)')
    ax.set_ylabel('1-Percentile - Radius of Chebyshev region $\epsilon$')
    if '_p_' in args.prop:
        p = int(args.prop.split('_')[-1])
        # plt.suptitle('Effect of $n$ on $\epsilon$-argmaxability %d-percentile' % (args.slack_dims, args.slack_source, p))
        if args.slack_dims:
            plt.suptitle('Effect of $n$ on $\epsilon$-argmaxability of $k$-sparse label assignments (%d slack variables)' % args.slack_dims, fontsize=20)
        else:
            plt.suptitle('Effect of $n$ on $\epsilon$-argmaxability of $k$-sparse label assignments (no slack variables)', fontsize=20)
    # ax.set_xlim([-1, len(d_slice.N)])
    plt.legend()
    plt.tight_layout()
    if args.save:
        filename = '%s-%s-dim-%d-%s.png' % (args.data.split('.')[0], args.slack_source, args.slack_dims, args.prop)
        plt.savefig('images/%s' % filename)
    else:
        plt.show()
