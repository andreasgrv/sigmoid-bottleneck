import re
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict


def model_name(attrs):
    return '{clf} s={slack_dims}'.format(**attrs)


metric_map = {
    'f1_at_8_te': 'F1@8 (%)',
    'prec_at_8_te': 'Prec@8 (%)',
    'rec_at_8_te': 'Rec@8 (%)',
    'f1_micro_te': 'Micro F1 (%)',
    'f1_macro_te': 'Macro F1 (%)'
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path to folder with exps.')
    parser.add_argument('--metric', required=True, help='Metric to plot.')

    args = parser.parse_args()

    ENCODER_DIM = 500
    VOCAB_SIZE = 8921

    results = defaultdict(list)
    for f in sorted(glob.glob('%s/*/metrics.json' % args.file)):
        exp = f.split('/')[-2].replace('cnn_vanilla_', '')
        exp = exp.replace('bottleneck_', 'BSL_')
        exp = exp.replace('fft_', 'DFT_')
        parser = re.compile(r'(?P<clf>[-\w+]+?)_k(?P<cardinality>\d+)_s(?P<slack_dims>\d+)_seed(?P<seed>\d+)')
        match = parser.match(exp)
        if match:
            params = match.groupdict()
            mname = model_name(params)
            results['model'].append(mname)
            for k, v in params.items():
                results[k].append(v)
            slack_dims = int(params['slack_dims'])
            cardinality = int(params['cardinality'])
            if mname.startswith('BSL '):
                results['num_params'].append(ENCODER_DIM * slack_dims + slack_dims * VOCAB_SIZE)
            elif mname.startswith('DFT'):
                results['num_params'].append(ENCODER_DIM * (2 * cardinality + 1 + slack_dims) + slack_dims * VOCAB_SIZE)
            with open(f) as fb:
                js = json.load(fb)
                for k, v in js.items():
                    if 'loss' in k:
                        results[k].append(min(v))
                    else:
                        results[k].append(max(v))
    df = pd.DataFrame.from_dict(results)
    # plt.plot(js['auc_macro'], label=exp)
    # plt.legend()
    # plt.show()
    te_cols = [c for c in df.columns if '_te' in c]
    at_8 = [c for c in df.columns if '_at_8' in c and c not in te_cols]
    macro_cols = [c for c in df.columns if 'macro' in c and c not in te_cols]
    micro_cols = [c for c in df.columns if 'micro' in c and c not in te_cols]

    at_8_te = [c for c in df.columns if '_at_8_te' in c]
    at_15_te = [c for c in df.columns if '_at_15_te' in c]
    macro_te_cols = [c for c in df.columns if 'macro' in c and c in te_cols]
    micro_te_cols = [c for c in df.columns if 'micro' in c and c in te_cols]

    all_cols = macro_cols + micro_cols + macro_te_cols + micro_te_cols + at_8 + at_8_te + at_15_te + ['time_per_epoch_tr'] + ['loss_test_te']

    df2 = df.groupby('model').agg({k: ['mean', 'std'] if k in all_cols else ['first'] for k in df.columns})
    df2 = df2.drop('model', axis=1)
    print(df2.columns)
    df2 = df2.sort_values(by='model', key=lambda x: x.apply(lambda x: (x.split(' ')[0], int(x.split(' ')[-1][2:]))))


    fig, ax1 = plt.subplots(figsize=(3.5, 5))
    ax2 = ax1.twinx()
    sym = {'BSL ': 'o', 'DFT': 's'}
    for m in ['BSL ', 'DFT']:
        df2_m = df2[df2.index.str.startswith(m)]
        # xx = df2_m['slack_dims']['first'].astype(int)
        xx = df2_m['slack_dims']['first']
        mean = df2_m[args.metric]['mean'] * 100
        std = df2_m[args.metric]['std'] * 100
        ax1.plot(xx, mean, '-' + sym[m], label=m)
        ax2.plot(xx, df2_m['num_params'], '--' + sym[m], label=m)
        ax1.fill_between(xx, mean-std, mean+std, alpha=.2)
    ax1.set_xlabel('$d$')
    ax1.set_ylabel(metric_map[args.metric])
    ax2.set_ylabel('# trainable parameters (dashed)')
    # ax1.legend(bbox_to_anchor=(.1, 1.125), loc='upper left')
    # ax1.legend(bbox_to_anchor=(.0, 1.25), loc='upper left')
    # ax1.legend()
    ax2.grid(None)
    # ax2.legend()
    plt.title('MIMIC III')
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
