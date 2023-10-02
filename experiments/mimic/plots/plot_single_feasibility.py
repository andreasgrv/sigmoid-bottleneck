import re
import csv
import glob
import json
import argparse
import pandas as pd

from collections import defaultdict
from itertools import product
from datasets import load_full_codes


def load_train_codes(train_path):
    codes = set()
    with open(train_path, 'r') as f:
        lr = csv.reader(f)
        next(lr)
        for row in lr:
            for code in row[3].split(';'):
                codes.add(code)
    codes = set([c for c in codes if c != ''])
    return codes


def model_name(attrs):
    return '{clf} s={slack_dims}'.format(**attrs)


def populate_files(path, results, split, idx_to_code, train_codes):

    f = path
    results['split'].append(split)

    exp = f.split('/')[1].replace('cnn_vanilla_', '')
    exp = exp.replace('bottleneck_', 'CNN-CSL_')
    exp = exp.replace('fft_', 'CNN-DFT_')
    parser = re.compile(r'(?P<clf>[-\w]+?)_k(?P<cardinality>\d+)_s(?P<slack_dims>\d+)_seed(?P<seed>\d+)')
    match = parser.match(exp)
    if match:
        results['model'].append(model_name(match.groupdict()))
        for k, v in match.groupdict().items():
            if k == 'slack_dims':
                results[k].append(int(v))
            else:
                results[k].append(v)

    with open(f, 'r') as ff:
        stats = json.load(ff)
        radii = [s['radius'] or 1e-8 for s in stats]
        eps_rad = [r > 1. for r in radii]
        feasible = [s['is_feasible'] for s in stats]
        pos_idxs = [tuple(sorted(s['pos_idxs'])) for s in stats]
        codes = [[idx_to_code[idx] for idx in p] for p in pos_idxs] 
        unseen_codes = [[c for c in row_c if c not in train_codes] for row_c in codes]
        num_unseen_codes = [len(c) for c in unseen_codes]
        status = [s['status'] for s in stats]
        cardinalities = [len(s['pos_idxs']) for s in stats]
        # print('Num Feasible: %d/%d' % (sum(feasible), len(stats)))
        # print('Radius range: (%.2f, %.2f)' % (min(radii), max(radii)))

        num_argmaxable = sum(feasible)
        num_eps_argmaxable = sum(eps_rad)

        for k in results.keys():
            results[k] = results[k] * len(radii)
        results['feasible'] = feasible
        results['status'] = status
        results['radius'] = radii
        results['cardinality'] = cardinalities
        results['pos_idxs'] = pos_idxs
        results['icd_codes'] = codes
        results['unseen_codes'] = unseen_codes
        results['num_unseen_codes'] = num_unseen_codes
        # results['num_argmaxable'].append(num_argmaxable)
        # results['num_eps_argmaxable'].append(num_eps_argmaxable)
        # results['num_points'].append(len(radii))
        # results['min_radius'].append(min(radii))
        # results['max_radius'].append(max(radii))
        # results['perc_argmaxable'].append('%d/%d' % (num_argmaxable, len(radii)))
        # results['perc_eps_argmaxable'].append('%d/%d' % (num_eps_argmaxable, len(radii)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, type=str, help='Path to analysis.json')
    parser.add_argument('--data', required=True, type=str, help='Path to data file.')

    args = parser.parse_args()
     
    idx_to_code, desc_dict = load_full_codes(args.data)
    train_codes = load_train_codes(args.data)

    results = defaultdict(list)
    if 'dev-' in args.file:
        populate_files('%s' % args.file, results, 'dev', idx_to_code, train_codes)
    else:
        populate_files('%s' % args.file, results, 'test', idx_to_code, train_codes)

    df = pd.DataFrame.from_dict(results)

    filtered_df = df[(~df['feasible']) & (df['num_unseen_codes'] == 0)]
    print(filtered_df)
    print('Example of lowest cardinality:')
    min_card_pos_idxs = filtered_df.sort_values(by='cardinality').iloc[1].pos_idxs

    print(min_card_pos_idxs, len(min_card_pos_idxs))


    min_code = list(set(idx_to_code[idx] for idx in min_card_pos_idxs))
    print(min_code)
    print('All codes not in train: %r' % list(c for c in min_code if c not in train_codes))

    min_desc = [(c, desc_dict[c]) for c in min_code]
    for md in sorted(min_desc):
        print('%s: %s' % md)

    # print(df)
    #
    # all_cols = ['num_argmaxable', 'num_eps_argmaxable']
    # df2 = df.groupby(['slack_dims','clf', 'split', 'num_points']).agg({k: ['median'] if k in all_cols else [] for k in df.columns})
    # df2 = df2.reset_index(level=['clf', 'slack_dims', 'num_points'])
    # # Drop median
    # df2 = df2.droplevel(1, axis=1)[['slack_dims', 'clf', 'num_argmaxable', 'num_eps_argmaxable', 'num_points']]
    #
    # df2['num_argmaxable'] = df2['num_argmaxable'].astype(int).astype(str) + '/' +  df2['num_points'].astype(str)
    # df2['num_eps_argmaxable'] = df2['num_eps_argmaxable'].astype(int).astype(str) + '/' +  df2['num_points'].astype(str)
    # # df2 = df2.sort_values(by=['slack_dims', 'clf'])
    # df2 = df2.set_index('slack_dims', append=True)
    # print(df2)
    # d = {}
    # for c in df2.clf.unique():
    #     d[c] = df2[df2.clf == c][['num_argmaxable', 'num_eps_argmaxable']]
    # print(d)
    # # for s, c in product(df2.slack_dims.unique(), df2.clf.unique()):
    # #     d[(s, c)] = df2[(df2.slack_dims == s) & (df2.clf == c)][['num_argmaxable']]
    # dd = pd.concat(d, axis=1)
    # print(dd)
    # dd = dd.reorder_levels([1, 0], axis=1)
    # dd = dd.sort_index(axis=1, level=0)
    # dd = dd.sort_values(by='split')
    # print(dd.to_latex())
