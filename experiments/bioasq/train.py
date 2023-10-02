#!/env/bin/python
import os
import torch
import random
import numpy as np

from transformers import AutoModel, get_cosine_schedule_with_warmup

from mlconf import YAMLLoaderAction, ArgumentParser
from spmlbl.metrics import MetricMonitor
from spmlbl.model import FrozenEncoderMultiLabelModel, MultiLabelModel
from spmlbl.data import create_dataloader, encode_batch_labels
from spmlbl.data import one_hot_decode, decode_batch_labels, encode_label_list


def encode_batch(batch, conf):
    pmids = batch.pop('pmid', None)

    batch = {k: v.to(torch.device(conf.device))
             for k, v in batch.items()}
    # Prepare labels depending on the method
    # TODO: May want to run below on cpu
    batch['labels'] = encode_batch_labels(batch['labels'],
                                          conf.label_encoder,
                                          method='sigmoid')


    if conf.freeze_encoder:
        inputs = batch['doc_embedding']
    else:
        doc_embeds = batch.pop('doc_embedding', None)
        inputs = dict()
        for k in ['input_ids', 'token_type_ids', 'attention_mask']:
            if k in batch:
                inputs[k] = batch[k]
    target = batch['labels']
    return inputs, target


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
    for batch in data:

        pmids = batch.pop('pmid', None)
        inputs, targets = encode_batch(batch, conf)

        with torch.no_grad():
            logits = model(inputs, labels=targets)
            L = model.loss.item()
        targets = targets.cpu().detach()

        model.loss = None

        logits = logits.cpu().detach()
        preds = (logits > 0.).int()

        conf.metrics.valid.loss(L)
        conf.metrics.valid.exact_acc(targets, preds)
        conf.metrics.valid.f1(targets, preds)
        conf.metrics.valid.p5(targets, logits)
        conf.metrics.valid.p10(targets, logits)
        conf.metrics.valid.r5(targets, logits)
        conf.metrics.valid.r10(targets, logits)
        conf.metrics.valid.macrof1(targets, preds)
        conf.metrics.valid.ndcg(targets, logits)
        conf.metrics.valid.ndcg5(targets, logits)
        conf.metrics.valid.ndcg10(targets, logits)

        assert len(pmids) == len(preds) == len(targets)
        for pmid, pred, target in zip(pmids, preds.numpy(), targets.numpy()):
            out = dict(pmid=pmid,
                       pred=np.nonzero(pred)[0].tolist(),
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

    MAX_ITERS = conf.max_iters
    BATCH_SIZE = conf.batch_size


    # ================== PREPARE DATASET =====================================
    dataloader = create_dataloader(conf, split='train')
    eval_dataloader = create_dataloader(conf, split='valid')

    # ================== PREPARE MODEL ========================================
    # print('Creating <%s> model...' % conf.model)

    if conf.continue_training:
        print('Continuing training... Loading <%s>...' % conf.paths.model)
        model = torch.load(conf.paths.model)
    else:
        if conf.freeze_encoder:
            model = FrozenEncoderMultiLabelModel(output_layer=conf.output_layer)
        else:
            encoder = AutoModel.from_pretrained(conf.encoder_model)
            model = MultiLabelModel(encoder=encoder, output_layer=conf.output_layer)
        print('Created model %r...' % (model.__class__))
        print(model)

    device = torch.device(conf.device)
    print('Running on %s' % device)
    model = model.to(device)
    model.train()

    if conf.freeze_encoder:
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    else:
        optimizer = torch.optim.Adam([{'params': model.encoder.parameters()},
                                      # {'params': [p for name, p in model.output_layer.named_parameters() if name != 'mlp.Ws.weight']},
                                      # {'params': model.output_layer.mlp.Ws.parameters(), 'lr': 1e-3, 'weight_decay': 1e-3},
                                      # {'params': model.output_layer.mlp.Ws.parameters(), 'lr': 1e-3},
                                      {'params': model.output_layer.parameters(), 'lr': 1e-3},
                                      ], lr=conf.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)


    # ==================== TRAIN LOOP =========================================
    conf.timer.tap()
    # Keep track of best metric
    prev_best_eval = None
    iter_idx, epoch = 0, 1
    patience = conf.patience

    while iter_idx < MAX_ITERS:

        for batch in dataloader:

            # Flags for whether we should print or eval during this iteration
            should_eval = (iter_idx % conf.eval_every) == 0
            should_print = (iter_idx % conf.print_every) == 0

            optimizer.zero_grad()

            inputs, target = encode_batch(batch, conf)
            logits = model(inputs, labels=target)

            target = target.cpu().detach()

            model.loss.backward()
            optimizer.step()
            L = model.loss.item()
            model.loss = None

            logits = logits.cpu().detach()
            preds = (logits > 0.).int()

            conf.metrics.train.loss(L)
            conf.metrics.train.exact_acc(target, preds)
            # conf.metrics.train.f1(target, preds)
            # conf.metrics.train.macrof1(target, preds)
            conf.metrics.train.p5(target, logits)
            conf.metrics.train.p10(target, logits)
            conf.metrics.train.r5(target, logits)
            conf.metrics.train.r10(target, logits)
            # NOTE: These are expensive to compute
            # conf.metrics.train.ndcg(target, logits)
            # conf.metrics.train.ndcg5(target, logits)
            # conf.metrics.train.ndcg10(target, logits)

            # Train print loop every "print_every" iterations
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

                gold_example = target[0, :].nonzero(as_tuple=True)[0].tolist()
                pred_example = preds[0, :].nonzero(as_tuple=True)[0].tolist()
                pred_example = pred_example[:conf.data.max_cardinality]

                print('Gold Active: %r' % gold_example)
                print('Pred Active: %r' % pred_example)
            # Eval loop every "eval_every" iterations
            if should_eval:
                model.eval()
                run_eval(model, eval_dataloader, conf)
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
                    # If we are not freezing the encoder, model too big to always save
                    if conf.freeze_encoder:
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
                if (patience <= 0) or (iter_idx > MAX_ITERS):
                    print('<TRAINING COMPLETE> after %d steps' % iter_idx)
                    print('Time taken: <%s>' % str(conf.timer))
                    iter_idx = MAX_ITERS
                    break
            iter_idx += 1
        epoch += 1
