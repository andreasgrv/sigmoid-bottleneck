import os
import json
import torch
import argparse
import numpy as np
import datasets

from mlbl.verifier import ArgmaxableSubsetVerifier
from mlbl.components import KSparseFFTClassifier
from mlbl.plot import plot_cardinalities
from constants import MIMIC_3_DIR


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str,
                        required=True,
                        help='Path to data to verify csv')
    parser.add_argument('--model', type=str,
                        required=True,
                        help='Path to model file')

    args = parser.parse_args()


    print('Loading model...')
    model = torch.load(args.model, map_location=torch.device('cpu'))
    print(model)
    try:
        W = model['clf.mlp.1.weight'].detach().numpy()
    except Exception as e:
        if 'clf.Ws.weight' in model:
            slack = model['clf.Ws.weight']
            proj_dim, in_dim = model['clf.proj.weight'].shape
            out_dim = slack.shape[0]
            k = ((proj_dim - slack.shape[1]) // 2)
            print(in_dim, out_dim, k)
            mod = KSparseFFTClassifier(in_dim, out_dim, k=k, slack_dims=slack.shape[1])
            mod.Ws.weight = torch.nn.parameter.Parameter(torch.tensor(slack))
        W = mod.compute_W().detach().numpy()
    b = None

    N, D = W.shape
    print('Loaded model with %d labels and %d label dim' % (N, D))


    print('Loading vocabulary...')
    train_file = os.path.join(MIMIC_3_DIR, 'train_full.csv')
    idx_to_icd, icd_to_desc = datasets.load_full_codes(train_file)
    icd_to_idx = {v:k for k, v in idx_to_icd.items()}

    print('Loading examples...')
    examples = []
    test_file = os.path.join(MIMIC_3_DIR, args.test_file)
    with open(test_file) as f:
        # Skip header
        next(f)
        for line in f:
            line = line.rstrip()
            line = line.split(',')[-2]
            str_labels = tuple(s for s in line.split(';') if s)
            labels = tuple([icd_to_idx[icd] for icd in str_labels])
            examples.append(labels)

    print('Loaded %d examples from %s' % (len(examples), test_file))
    # plot_cardinalities(examples, 'MIMIC-III test set')

    examples = list(set(examples))
    print('%d are unique - checking those...' % (len(examples)))


    vv = ArgmaxableSubsetVerifier(W=W, b=b, num_processes=15, check_rank=False)
    res = vv(examples)
    num_feasible = sum([r['is_feasible'] for r in res])
    print('%d/%d feasible' % (num_feasible, len(res)))

    test_filename = os.path.basename(test_file)
    out_folder = os.path.dirname(args.model)
    out_file = os.path.join(out_folder, '%s-analysis.json' % test_filename.split('_')[0])
    # Do not save the point since this takes up a lot of space and obscures json
    for r in res:
        r['point'] = []
    print('Writing analysis results to: %s' % out_file)
    with open(out_file, 'w') as f:
        json.dump(res, f, indent=2, cls=NumpyEncoder)
