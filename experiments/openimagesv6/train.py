import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from mlconf import YAMLLoaderAction, ArgumentParser
from spmlbl.metrics import MetricMonitor
from torch.utils.data import DataLoader
from spmlbl.data import OpenImagesDataset
from spmlbl.components import SigmoidBottleneckLayer, KSparseFFTClassifier

from asl.models import create_model


def run_eval(model, data, conf):

    conf.metrics.valid.loss.reset()
    conf.metrics.valid.exact_acc.reset()
    conf.metrics.valid.f1.reset()
    conf.metrics.valid.p5.reset()
    conf.metrics.valid.p10.reset()
    conf.metrics.valid.r5.reset()
    conf.metrics.valid.r10.reset()
    conf.metrics.valid.macrof1.reset()
    conf.metrics.valid.ndcg.reset()
    conf.metrics.valid.ndcg5.reset()
    conf.metrics.valid.ndcg10.reset()

    all_outputs = []

    bce = torch.nn.BCEWithLogitsLoss()

    for _, images, labels in data:

        with torch.no_grad():
            logits = model(images.to(conf.device))
            loss = bce(logits, labels.float().to(conf.device))
            L = loss.item()

        logits = logits.cpu().detach()
        preds = (logits > 0.).int()

        conf.metrics.valid.loss(L)
        conf.metrics.valid.exact_acc(labels, preds)
        conf.metrics.valid.f1(labels, preds)
        conf.metrics.valid.p5(labels, logits)
        conf.metrics.valid.p10(labels, logits)
        conf.metrics.valid.r5(labels, logits)
        conf.metrics.valid.r10(labels, logits)
        conf.metrics.valid.macrof1(labels, preds)
        conf.metrics.valid.ndcg(labels, logits)
        conf.metrics.valid.ndcg5(labels, logits)
        conf.metrics.valid.ndcg10(labels, logits)

        for pred, target in zip(preds.numpy(), labels.numpy()):
            out = dict(pred=np.nonzero(pred)[0].tolist(),
                       target=np.nonzero(target)[0].tolist())
            all_outputs.append(out)
    return all_outputs



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--blueprint', action=YAMLLoaderAction)
    parser.add_argument('--device', type=str, default=os.environ['MLBL_DEVICE'])
    config = parser.parse_args()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Build blueprint
    conf = config.build()
    # Save current setup in experiment folder
    config.to_file(conf.paths.blueprint)

    monitor = MetricMonitor(conf.metrics, conf.paths.stats)
    # If we are continuing training an existing model
    # do not overwrite previous stats - just append
    if not conf.continue_training:
        monitor.write_header()

    state = torch.load(config.model, map_location='cpu')
    config.num_classes = state['num_classes']
    model = create_model(config)
    model.load_state_dict(state['model'], strict=True)

    # Replace head with our output layer
    model.head = conf.output_layer
    # We actually have less classes than the loaded model
    config.num_classes = config.output_layer.out_dim

    # Place model on device
    print('Running on %s' % config.device)
    model = model.to(config.device)
    print(model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    optimizer = torch.optim.Adam([{'params': model.body.parameters()},
                                  {'params': model.head.parameters(), 'lr': 1e-3},
                                  ], lr=conf.lr)


    # TODO: Do we need this?
    # normalize = transforms.Normalize(mean=[0, 0, 0],
    #                                  std=[1, 1, 1])
    # ==================== DATA LOADING =======================================
    ts = transforms.Compose([
             transforms.Resize((config.image_size, config.image_size)),
             transforms.ToTensor(),
        ])

    train_ds = OpenImagesDataset(config.data.train_ann_path,
                                 config.data.train_images_path,
                                 config.num_classes,
                                 transform=ts)
    train_data = DataLoader(train_ds, batch_size=conf.batch_size,
                            shuffle=True, num_workers=config.workers, pin_memory=True)

    valid_ds = OpenImagesDataset(config.data.valid_ann_path,
                                 config.data.valid_images_path,
                                 config.num_classes,
                                 transform=ts)
    valid_data = DataLoader(valid_ds, batch_size=conf.batch_size,
                            shuffle=False, num_workers=config.workers)

    # ==================== TRAIN LOOP =========================================
    conf.timer.tap()
    # Keep track of best metric
    prev_best_eval = None
    iter_idx, epoch = 0, 1
    patience = conf.patience

    bce = torch.nn.BCEWithLogitsLoss()

    while iter_idx < config.max_iters:

        for _, images, labels in train_data:

            # Flags for whether we should print or eval during this iteration
            should_eval = (iter_idx % conf.eval_every) == 0
            should_print = (iter_idx % conf.print_every) == 0

            optimizer.zero_grad()

            logits = model(images.to(config.device))
            loss = bce(logits, labels.float().to(config.device))
            loss.backward()
            optimizer.step()
            L = loss.item()

            logits = logits.cpu().detach()
            preds = (logits > 0.).int()

            conf.metrics.train.loss(L)
            conf.metrics.train.p5(labels, logits)
            conf.metrics.train.p10(labels, logits)
            conf.metrics.train.r5(labels, logits)
            conf.metrics.train.r10(labels, logits)

            if should_print:
                conf.metrics.train.time(conf.timer.delta)
                conf.timer.tap()
                print('Epoch %d, %.1fk steps\n'
                      'Training %s, %s, %s'
                      % (
                         epoch,
                         (iter_idx - 1) / 1000.,
                         conf.metrics.train.loss.legend,
                         conf.metrics.train.p5.legend,
                         conf.metrics.train.r5.legend))

                gold_example = labels[0, :].nonzero(as_tuple=True)[0].tolist()
                pred_example = preds[0, :].nonzero(as_tuple=True)[0].tolist()
                pred_example = pred_example[:20]

                print('Gold Active: %r' % gold_example)
                print('Pred Active: %r' % pred_example)
            # Eval loop every "eval_every" iterations
            if should_eval:
                model.eval()
                run_eval(model, valid_data, conf)
                model.train()
                print('Epoch %d, %.1fk steps\n'
                      'Valid %s, %s, %s, %s'
                      % (
                         epoch,
                         (iter_idx - 1) / 1000.,
                         conf.metrics.valid.loss.legend,
                         conf.metrics.valid.f1.legend,
                         conf.metrics.valid.exact_acc.legend,
                         conf.metrics.valid.p5.legend))

                # Checkpointing based on valid loss
                current_eval = conf.metrics.valid.loss.value
                if conf.save_checkpoints:
                    torch.save(model, conf.paths.model_at_step(iter_idx))
                if prev_best_eval is None:
                    prev_best_eval = current_eval
                if prev_best_eval > current_eval:
                    print('### New best valid loss %.5f < %.5f' %
                          (current_eval, prev_best_eval))
                    print('Saving model to "%s"' % conf.paths.model)
                    torch.save(model, conf.paths.model)
                    prev_best_eval = current_eval
                    patience = conf.patience
                else:
                    patience -= 1
                print('### (Patience: %d)' % patience)
            # Log metrics and decide whether to continue
            if should_print or should_eval:
                # Write current stats window to file
                monitor.append()
                if (patience <= 0) or (iter_idx > config.max_iters):
                    print('<TRAINING COMPLETE> after %d steps' % iter_idx)
                    print('Time taken: <%s>' % str(conf.timer))
                    iter_idx = config.max_iters
                    break

            iter_idx += 1
        epoch += 1
