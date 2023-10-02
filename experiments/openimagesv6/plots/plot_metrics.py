import re
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from collections import defaultdict


def model_name(attrs):
    return '{clf} s={slack_dims}'.format(**attrs)


metric_map = {
    'f1@10': 'F1@10 (%)',
    'prec@10': 'Prec@10 (%)',
    'rec@10': 'Rec@10 (%)',
    'macrof1': 'Macro F1 (%)',
    'f1': 'Micro F1 (%)',
    'ndcg': 'NDCG'
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path to folder with exps.')
    parser.add_argument('--metric', required=True, help='Metric to plot.')

    args = parser.parse_args()

    ENCODER_DIM = 2432
    VOCAB_SIZE = 8933

    results = defaultdict(list)
    for f in sorted(glob.glob('%s/*/analysis/test-metrics.json' % args.file)):
        exp = f.split('/')[-3]
        parser = re.compile(r'(?P<clf>[-\w+]+?)-D-(?P<slack_dims>\d+)-S-(?P<seed>\d+)')
        match = parser.match(exp)
        if match:
            params = match.groupdict()
            mname = model_name(params)
            results['model'].append(mname)
            for k, v in params.items():
                results[k].append(v)
            slack_dims = int(params['slack_dims'])
            cardinality = 50
            if mname.startswith('BSL '):
                results['num_params'].append(ENCODER_DIM * slack_dims + slack_dims * VOCAB_SIZE)
            elif mname.startswith('DFT'):
                results['num_params'].append(ENCODER_DIM * (2 * cardinality + 1 + slack_dims) + slack_dims * VOCAB_SIZE)
            with open(f) as fb:
                js = json.load(fb)
                for k, v in js.items():
                    if k != 'model':
                        results[k].append(v)
    df = pd.DataFrame.from_dict(results)
    df['f1@5'] = (2 * df['prec@5'] * df['rec@5']) / (df['prec@5'] + df['rec@5'])
    df['f1@10'] = (2 * df['prec@10'] * df['rec@10']) / (df['prec@10'] + df['rec@10'])
    # df = df.sort_values(by='model', key=lambda x: x.apply(lambda x: (x.split(' ')[0], int(x.split(' ')[-1][2:]))))
    #         # plt.plot(js['auc_macro'], label=exp)
    print(df)
    all_cols = [c for c in df.columns if c not in ['model', 'clf', 'slack_dims', 'seed']]
    df2 = df.groupby('model').agg({k: ['mean', 'std'] if k in all_cols else ['first'] for k in df.columns})
    df2 = df2.drop('model', axis=1)
    df2 = df2.sort_values(by='model', key=lambda x: x.apply(lambda x: (x.split(' ')[0], int(x.split(' ')[-1][2:]))))
    #
    #
    fig, ax1 = plt.subplots(figsize=(3.5, 5))
    ax2 = ax1.twinx()
    sym = {'BSL ': 'o', 'DFT': 's'}
    print(df2)
    for i, mlabel in enumerate(['BSL ', 'DFT']):
        df_m = df2[df2.index.str.startswith(mlabel)]
        xx = df_m['slack_dims']['first']
        # xx = df_m['slack_dims']
        # NOTE: Here this is not the mean - just actual value
        if args.metric not in ['ndcg', 'f1']:
            mean = df_m[args.metric]['mean'] * 100
            std = df_m[args.metric]['std'] * 100
        else:
            mean = df_m[args.metric]['mean']
            std = df_m[args.metric]['std']
        ax1.plot(xx, mean, '-' + sym[mlabel], label=mlabel, color=cm.tab10(i))
        ax2.plot(xx, df_m['num_params']['mean'], '--' + sym[mlabel], label=mlabel, color=cm.tab10(i))
        ax1.fill_between(xx, mean-std, mean+std, alpha=.2)
    ax1.set_xlabel('$d$')
    ax1.set_ylabel(metric_map.get(args.metric, args.metric))
    ax2.set_ylabel('# trainable parameters (dashed)')
    # ax1.legend(bbox_to_anchor=(.1, 1.125), loc='upper left')
    # ax1.legend(bbox_to_anchor=(.0, 1.25), loc='upper left')
    ax2.grid(None)
    # ax2.legend()
    plt.title('OpenImages v6')
    plt.tight_layout()
    plt.show()
    # print(df2)
    # print(df2['prec_macro'])

    # stats = pd.DataFrame()
    # for k in df2.columns.get_level_values(0):
    #     if k == 'model':
    #         stats[k] = df2[k]
    #     else:
    #         stats[k] = df2[k].apply(lambda x: '%.2f ± %.2f' % (x['mean']*100, x['std']*100), axis=1)
    # # print(stats.columns)
    # stats = stats.sort_values(by='model', key=lambda x: x.apply(lambda x: (x.split(' ')[0], int(x.split(' ')[-1][2:]))))

    # print(stats[at_k].to_latex())
    # print()
    # print(stats[macro_cols].to_latex())
    # print()
    # print(stats[micro_cols].to_latex())
    # print()
    # print(stats[['time_per_epoch_tr']].to_latex())
    # print()
    # print(stats[at_k_te].to_latex())
    # print()

    # print(stats[macro_te_cols].to_markdown())
    # print()
    # print(stats[micro_te_cols].to_markdown())
    # print()
