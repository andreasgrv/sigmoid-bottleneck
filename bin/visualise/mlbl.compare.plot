import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from collections import defaultdict


def load_csv_attributes(filename, attributes):
    results = dict()
    with open(filename, 'r') as f:
        header = next(f).strip().split('\t')
        for col in header:
            results[col] = []
        nlines = 0
        for line in f:
            line = line.strip()
            for i, v in enumerate(line.split('\t')):
                try:
                    v = float(v)
                except Exception:
                    pass
                results[header[i]].append(v)
            if nlines > args.timesteps:
                break
            nlines += 1
    results = {attr: results[attr] for attr in attributes}

    return results


def load_json_attributes(filename, attributes):
    with open(filename, 'r') as f:
        results = json.load(f)
    results = {attr: results[attr] for attr in attributes}
    return results


def load_results(filenames, attributes):
    results = {}
    for filename in filenames:
        exp = os.path.basename(os.path.dirname(filename))
        exp = exp.replace('.tsv', '')
        if filename.endswith('.tsv'):
            results[exp] = load_csv_attributes(filename, attributes)
        elif filename.endswith('.json'):
            results[exp] = load_json_attributes(filename, attributes)
        else:
            raise ValueError('Unknown file ending, expected .json or .tsv')

    # Make the dict be attr headed
    attr_results = defaultdict(dict)
    for fname, attrs in results.items():
        for attr, vals in attrs.items():
            attr_results[attr][fname] = vals
    return attr_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, nargs='+', required=True,
                        help='Path to results tsv files.')
    parser.add_argument('--attributes', type=str, nargs='+', required=True,
                        help='What attributes to plot.')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Timesteps to plot.')
    parser.add_argument('--savefig', type=str, default=None,
                        help='Where to save file.')

    args = parser.parse_args()
    args.timesteps = args.timesteps or np.inf

    results = load_results(args.results, args.attributes)

    num_attrs = len(results)
    num_models = len(results[list(results.keys())[0]])

    fig, axs = plt.subplots(nrows=num_attrs, ncols=1, sharex=True)

    if num_attrs == 1:
        axs = [axs]

    for attr, ax in zip(results.keys(), axs):
        for model, vals in results[attr].items():
            p = ax.plot(vals, label=model, lw=1.2)
            if 'loss' in attr:
                ax.text(0, vals[0], '%.2E' % vals[0], color=p[0].get_color(),
                        ha='left', va='bottom')
                ax.set_yscale('log')
            else:
                ax.text(len(vals), vals[-1], '%.2f' % vals[-1],
                        color=p[0].get_color(),
                        ha='right',
                        va='bottom')
                # ax.set_ylim([-1, 101])
            ax.legend()
            ax.set_title(attr)
        ax.set_ylabel(attr)
    plt.xlabel('Number of steps (x100)')
    plt.tight_layout()
    if args.savefig:
        filepath = args.savefig
        print('Saving plot as %s...' % filepath)
        plt.savefig(filepath)
    else:
        plt.show()

    # ax1.plot(results['train.loss'], color='tab:red', label='Train Loss')
    # ax1.text(0, results['train.loss'][0], '%.2E' % results['train.loss'][0],
    #          color='tab:red', fontsize=14, ha='left', va='bottom')
    # ax1.text(len(results['train.loss']), results['train.loss'][-1], '%.2E' % results['train.loss'][-1],
    #          color='tab:red', fontsize=14, ha='right', va='top')
    # if 'valid.loss' in header:
    #     ax1.plot(results['valid.loss'], color='tab:orange', label='Valid Loss')
    #     ax1.text(len(results['valid.loss']), results['valid.loss'][-1], '%.2E' % results['valid.loss'][-1],
    #              color='tab:orange', fontsize=14, ha='right', va='bottom')
    # ax1.set_yscale('log')
    # ax1.legend(loc='upper right')
    # if 'train.f1' in header:
    #     ax2.plot(results['train.f1'], color='tab:blue', label='Train F1')
    #     ax2.text(len(results['train.f1']), results['train.f1'][-1], '%.2f%%' % (results['train.f1'][-1]),
    #              color='tab:blue', fontsize=14, ha='right', va='bottom')
    # if 'train.accuracy' in header:
    #     ax2.plot(results['train.accuracy'], color='tab:blue', label='Train Acc')
    #     ax2.text(len(results['train.accuracy']), results['train.accuracy'][-1], '%.2f%%' % (results['train.accuracy'][-1]),
    #              color='tab:blue', fontsize=14, ha='right', va='bottom')
    # if 'train.exact_match' in header:
    #     ax2.plot(results['train.exact_match'], color='tab:purple', label='Train Exact Match Acc')
    #     ax2.text(len(results['train.exact_match']), results['train.exact_match'][-1], '%.2f%%' % (results['train.exact_match'][-1]),
    #              color='tab:purple', fontsize=14, ha='right', va='bottom')
    # if 'valid.f1' in header:
    #     ax2.plot(results['valid.f1'], color='tab:green', label='Valid F1')
    #     ax2.text(len(results['valid.f1']), results['valid.f1'][-1] + 2., '%.2f%%' % (results['valid.f1'][-1]),
    #              color='tab:green', fontsize=14, ha='right', va='bottom')
    # if 'valid.exact_match' in header:
    #     ax2.plot(results['valid.exact_match'], color='tab:pink', label='Valid Exact Match Acc')
    #     ax2.text(len(results['valid.exact_match']), results['valid.exact_match'][-1] - 2., '%.2f%%' % (results['valid.exact_match'][-1]),
    #              color='tab:pink', fontsize=14, ha='right', va='top')
    # # ax2.plot(results['Exact Acc'], color='tab:purple', label='Structured Accuracy')
    # ax2.legend(loc='upper center')
    # ax1.set_ylabel('Log Crossentropy Loss')
    # ax2.set_ylabel('Micro F1 %')
    # plt.xlabel('Batch Updates x 100')
    # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # # model, _, labeldim,  _, dataset = args.results.split('/')[-1][:-3].split('-')
    # ax1.set_title('Model: %s' % (args.results.split('.')[0]))
    # ax1.grid(linewidth=.8)
    # ax2.grid(linewidth=.8)
    # # ax1.set_ylim([1e-2, 9e-1])
    # # ax1.set_ylim([5e-4, 2e-2])
    # # ax1.set_ylim([5e-4, 1e-1])
    # ax2.set_ylim([-1., 105])
    # plt.tight_layout()
    # if args.save:
    #     filepath = os.path.join('plots', 'images', '%s.png' % filename)
    #     print('Saving plot as %s...' % filepath)
    #     plt.savefig(filepath)
    # else:
    #     plt.show()
