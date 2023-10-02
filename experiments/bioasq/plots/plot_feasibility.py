import re
import glob
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from itertools import product


def model_name(attrs):
    return '{clf} s={slack_dims}'.format(**attrs)


def populate_files(path, results, split):

    for f in sorted(glob.glob(path)):
        results['split'].append(split)

        exp = f.split('/')[-3]
        parser = re.compile(r'(?P<clf>[-\w+]+?)-L-P-1-D-(?P<slack_dims>\d+)-S-(?P<seed>\d+)')
        match = parser.match(exp)
        if match:
            results['model'].append(model_name(match.groupdict()))
            for k, v in match.groupdict().items():
                if k == 'slack_dims':
                    results[k].append(int(v))
                else:
                    results[k].append(v)

        with open(f, 'r') as ff:
            stats = json.load(ff)['examples']
            radii = [s['gold']['radius'] or 1e-8 for s in stats]
            eps_rad = [r > 1. for r in radii]
            feasible = [s['gold']['feasible'] for s in stats]
            # print('Num Feasible: %d/%d' % (sum(feasible), len(stats)))
            # print('Radius range: (%.2f, %.2f)' % (min(radii), max(radii)))

            num_argmaxable = sum(feasible)
            num_eps_argmaxable = sum(eps_rad)

            results['num_argmaxable'].append(num_argmaxable)
            results['num_eps_argmaxable'].append(num_eps_argmaxable)
            results['num_points'].append(len(radii))
            results['min_radius'].append(min(radii))
            results['max_radius'].append(max(radii))
            # results['perc_argmaxable'].append('%d/%d' % (num_argmaxable, len(feasible)))
            # results['perc_eps_argmaxable'].append('%d/%d' % (num_eps_argmaxable, len(feasible)))
            results['perc_argmaxable'].append(float(num_argmaxable) / len(feasible) * 100)
            results['perc_eps_argmaxable'].append(float(num_eps_argmaxable) / len(feasible) * 100)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, type=str, help='Path to analysis.json')

    args = parser.parse_args()

    results = defaultdict(list)
    # populate_files('%s/*/dev-analysis.json' % args.file, results, 'dev')
    populate_files('%s/*/analysis/test.json' % args.file, results, 'test')

    df = pd.DataFrame.from_dict(results)
    print(df)

    all_cols = ['num_argmaxable', 'num_eps_argmaxable']
    df2 = df.groupby(['slack_dims','clf', 'split', 'num_points']).agg({k: ['median'] if k in all_cols else [] for k in df.columns})
    df2 = df2.reset_index(level=['clf', 'slack_dims', 'num_points'])
    # Drop median
    df2 = df2.droplevel(1, axis=1)[['slack_dims', 'clf', 'num_argmaxable', 'num_eps_argmaxable', 'num_points']]

    # df2['num_argmaxable'] = df2['num_argmaxable'].astype(int).astype(str) + '/' +  df2['num_points'].astype(str)
    df2['num_argmaxable'] = df2['num_argmaxable'].astype(int).astype(str)
    # df2['num_eps_argmaxable'] = df2['num_eps_argmaxable'].astype(int).astype(str) + '/' +  df2['num_points'].astype(str)
    df2['num_eps_argmaxable'] = df2['num_eps_argmaxable'].astype(int).astype(str)
    # df2 = df2.sort_values(by=['slack_dims', 'clf'])
    df2 = df2.set_index('slack_dims', append=True)
    print(df2)
    d = {}
    for c in df2.clf.unique():
        d[c] = df2[df2.clf == c][['num_argmaxable', 'num_eps_argmaxable']]
    print(d)
    # for s, c in product(df2.slack_dims.unique(), df2.clf.unique()):
    #     d[(s, c)] = df2[(df2.slack_dims == s) & (df2.clf == c)][['num_argmaxable']]
    dd = pd.concat(d, axis=1)
    dd = dd.reorder_levels([1, 0], axis=1)
    dd = dd.sort_index(axis=1, level=0)
    dd = dd.sort_values(by='split')
    print(dd.to_latex())


    plot_df = df[['perc_argmaxable', 'clf', 'slack_dims', 'seed', 'split']]
    plot_df = plot_df.groupby(['clf', 'slack_dims', 'split']).agg({k: ['mean', 'std'] if k == 'perc_argmaxable' else ['first'] for k in plot_df.columns})
    plot_df = plot_df.drop(['slack_dims', 'seed', 'split', 'clf'], axis=1)
    plot_df = plot_df.reset_index()
    print(plot_df)
    plot_df = plot_df.sort_values(['slack_dims', 'split', 'clf'])

    split = 'test'
    fig, ax = plt.subplots(figsize=(1.5, 5))
    dev_df = plot_df[plot_df.split==split]
    sym = {'BSL': 'o', 'DFT': 's'}
    xx = [str(e) for e in dev_df['slack_dims'].unique()]
    for model in dev_df.clf.unique():
        mean = dev_df[dev_df.clf==model]['perc_argmaxable']['mean']
        std = dev_df[dev_df.clf==model]['perc_argmaxable']['std']
        ax.plot(xx, mean, '-' + sym[model], label=model)
        ax.fill_between(xx, mean-std, mean+std, alpha=.2)
    # ax.set_ylabel('%% of %s set Argmaxable' % split)
    ax.set_xlabel('$d$')
    ax.set_ylim([-2.5, 102.5])
    # ax.legend(loc=4)
    # ax.legend(bbox_to_anchor=(.0, 1.25), loc='upper left')
    ax.legend()
    plt.title('BioASQ')
    plt.tight_layout()
    plt.show()
