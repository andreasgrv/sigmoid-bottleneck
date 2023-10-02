import os
import json
import torch
import random
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from transformers import AutoModel

from mlconf import YAMLLoaderAction, ArgumentParser
from spmlbl.components import SigmoidBottleneckLayer
from spmlbl.verifier import ArgmaxableSubsetVerifier
from spmlbl.data import create_dataloader, encode_batch_labels, decode_batch_labels
from spmlbl.data import one_hot_decode, encode_label_list, bioasq_submission
from torch.utils.data import DataLoader
from spmlbl.data import OpenImagesDataset
from train import run_eval


# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_model_W(model):
    if isinstance(model.head, SigmoidBottleneckLayer):
        W = model.head.mlp[-1].weight
    else:
        W = model.head.compute_W()
    return W.cpu().detach().numpy()


if __name__ == "__main__":

    parser = ArgumentParser()
    # parser.add_argument('--model', type=str, default=None, help='Path to model to eval.')
    parser.add_argument('--eval-type', type=str, default=None, help='Path to dataset to eval on')
    parser.add_argument('--blueprint', action=YAMLLoaderAction)

    config = parser.parse_args()
    config.paths.root_folder = os.environ['MLBL_OPENIMAGES_ROOT']
    config.device = 'cuda:0'


    conf = config.build()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)


    device = torch.device(config.device)
    # if config.model:
    #     model = torch.load(conf.model, map_location=device)
    # else:
    model = torch.load(conf.paths.model, map_location=device)
    print('Loading model from %s...' % conf.paths.model)
    model = model.eval()

    ts = transforms.Compose([
             # transforms.ToPILImage(),
             transforms.Resize((config.image_size, config.image_size)),
             transforms.ToTensor(),
        ])

    print('Loading dataset...')
    if config.eval_type == 'valid':
        ds = OpenImagesDataset(config.data.valid_ann_path,
                               config.data.valid_images_path,
                               config.output_layer.out_dim,
                               transform=ts)
        data = DataLoader(ds, batch_size=conf.batch_size,
                          shuffle=False, num_workers=config.workers)
    elif config.eval_type == 'test':
        ds = OpenImagesDataset(config.data.test_ann_path.replace('test.csv', 'test-10k.csv'),
                               config.data.test_images_path,
                               config.output_layer.out_dim,
                               transform=ts)
        data = DataLoader(ds, batch_size=conf.batch_size,
                          shuffle=False, num_workers=config.workers)
    else:
        raise ValueError('Unknown eval: %s' % config.eval_type)

    print('Running evaluation...')
    preds = run_eval(model, data, conf)
    results = dict()
    results['model'] = conf.paths.experiment_name
    for k, v in conf.metrics.valid.items():
        results[v.label.lower()] = v.value

    all_labels = set()
    for out in preds:
        all_labels.add(tuple(out['pred']))
        all_labels.add(tuple(out['target']))

    W = load_model_W(model)

    print('Checking argmaxability...')
    v = ArgmaxableSubsetVerifier(W=W, check_rank=False)
    res = v(list(all_labels), lb=-1e4, ub=1e4)

    lookup = dict()
    for r in res:
        lookup[tuple(r['pos_idxs'])] = r

    entries = dict()
    entries['examples'] = []
    for out in preds:
        entry = dict()
        entry['pred'] = dict()
        entry['pred']['labels'] = out['pred']
        pred_feas = lookup[tuple(out['pred'])]
        assert pred_feas['is_feasible']
        entry['pred']['radius'] = pred_feas['radius']

        entry['gold'] = dict()
        entry['gold']['labels'] = out['target']

        gold_feas = lookup[tuple(out['target'])]
        entry['gold']['feasible'] = gold_feas['is_feasible']
        entry['gold']['status'] = gold_feas['status']
        entry['gold']['radius'] = gold_feas['radius']

        entries['examples'].append(entry)

    EPSILON_THRESH = 1.
    num_argmaxable = 0
    num_epsilon_argmaxable = 0
    total = 0
    for entry in entries['examples']:
        if entry['gold']['feasible']:
            num_argmaxable += 1
            if entry['gold']['radius'] > EPSILON_THRESH:
                num_epsilon_argmaxable += 1
        total += 1
    argmax_perc = (num_argmaxable / total) * 100
    e_argmax_perc = (num_epsilon_argmaxable / total) * 100

    results['argmax_p'] = argmax_perc
    results['e-argmax_p'] = e_argmax_perc
    # df = pd.DataFrame.from_dict(results)
    # print(df)
    # print('Argmaxable perc: %.2f' % argmax_perc)
    # print('E-Argmaxable perc: %.2f' % e_argmax_perc)

    out_file = conf.paths.analysis_for(config.eval_type)
    # Save analysis as json to file
    print('### Writing entries to %s...' % out_file)
    with open(out_file, 'w') as f:
        json.dump(entries, f, indent=2, sort_keys=True, cls=NumpyEncoder)

    m_out_file = conf.paths.analysis_for('%s-metrics' % config.eval_type)
    # Save analysis as json to file
    print('### Writing metrics to %s...' % m_out_file)
    with open(m_out_file, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)
