import math
import torch
import numpy as np


class TransformerEncoder(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, nhead, dropout):
        super(TransformerEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.dropout = dropout

        self.embed = torch.nn.Embedding(self.vocab_size, self.hidden_dim)
        transformer_layer = torch.nn.TransformerEncoderLayer(self.hidden_dim,
                                                             self.nhead,
                                                             dropout=self.dropout,
                                                             batch_first=True,
                                                             norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(transformer_layer, num_layers=2)

    def forward(self, input_ids, **kwds):
        src_embed = self.embed(input_ids) * math.sqrt(self.hidden_dim)

        # TODO: Improve this - do not want to just take first
        outputs = self.encoder(src_embed)[:, 0, :]
        return outputs


class TransformerSigmoid(torch.nn.Module):
    def __init__(self, encoder, label_dim, num_labels, pos_weight=None):
        super(TransformerSigmoid, self).__init__()
        self.encoder = encoder
        self.label_dim = label_dim
        self.num_labels = num_labels
        # --- Uncomment for BERT encoder
        # self.hidden_dim = encoder.pooler.dense.weight.shape[-1]
        # --- Uncomment for transformer encoder
        self.hidden_dim = 512
        self.pos_weight = pos_weight
        # self.hidden_dim = 1024

        # assert 512 <= self.hidden_dim <= 1024

        self.logit_mlp = torch.nn.Sequential(
            # torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            # torch.nn.ReLU(),
            # torch.nn.LayerNorm(self.hidden_dim),
            # torch.nn.Linear(self.hidden_dim, self.label_dim, bias=False),
            # torch.nn.Linear(self.hidden_dim, self.num_labels, bias=False)

            # This is equivalent to the 2 cardinality case of Cyclic
            # torch.nn.Linear(self.hidden_dim, 61, bias=False),
            # torch.nn.Linear(61, self.num_labels, bias=False)

            # torch.nn.Linear(self.hidden_dim, self.num_labels, bias=False)

            CyclicLinear(self.hidden_dim, num_labels, 15, freeze=False)
            # CyclicLinear(self.hidden_dim, num_labels, 30, freeze=True)
        )
        # torch.nn.init.uniform_(self.logit_mlp[-1].weight)
        # torch.nn.init.uniform_(self.logit_mlp[-1].weight, 0, 1./np.sqrt(self.hidden_dim))

        if self.pos_weight is not None:
            self.lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones([self.num_labels]) * self.pos_weight)
        else:
            self.lossfn = torch.nn.BCEWithLogitsLoss()
        self.loss = None

    def forward(self, batch_dict):
        # labels: batch_dim x self.num_labels  (int vector of 0s and 1s)
        # the tensor contains the indices of active labels
        # tensor second dim is always max_active_labels size
        # <PAD> token used to pad to max size
        labels = batch_dict.pop('labels')

        # Take BERT pooler output
        # --- Uncomment for BERT encoder
        # outputs = self.encoder(**batch_dict).pooler_output
        # --- Uncomment for transformer encoder
        outputs = self.encoder(**batch_dict)
        logits = self.logit_mlp(outputs)

        batch_dim, num_labels = logits.shape

        targets = labels.float()
        # assert ((labels == 0).sum() + (labels == 1).sum()) == (labels.shape[0] * labels.shape[1])
        self.loss = self.lossfn(logits, targets)
        return logits


class TransformerSigmoidBPE(torch.nn.Module):
    def __init__(self, encoder, num_labels, vocab, pos_weight=None):
        super(TransformerSigmoidBPE, self).__init__()
        self.encoder = encoder
        self.num_labels = num_labels
        self.hidden_dim = encoder.pooler.dense.weight.shape[-1]
        vocab = torch.tensor(vocab)
        self.register_buffer('vocab', vocab)
        self.pos_weight = pos_weight
        # self.W = torch.nn.Linear(self.hidden_dim, self.num_labels)

        if self.pos_weight is not None:
            self.lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones([self.num_labels]) * self.pos_weight)
        else:
            self.lossfn = torch.nn.BCEWithLogitsLoss()
        self.loss = None

    def forward(self, batch_dict):
        # labels: batch_dim x self.num_labels  (int vector of 0s and 1s)
        # the tensor contains the indices of active labels
        # tensor second dim is always max_active_labels size
        # <PAD> token used to pad to max size
        batch_dict.pop('labels')
        labels = batch_dict.pop('bpe_labels')

        # Take BERT pooler output
        outputs = self.encoder(**batch_dict).pooler_output
        # outputs = self.encoder(**batch_dict)
        # outputs = self.encoder(**batch_dict).last_hidden_state[:, 0, :]

        W = self.encoder.embeddings.word_embeddings(self.vocab)
        logits = outputs.matmul(W.T)
        # logits = self.W(outputs)

        batch_dim, num_labels = logits.shape

        targets = labels.float()
        # assert ((labels == 0).sum() + (labels == 1).sum()) == (labels.shape[0] * labels.shape[1])
        self.loss = self.lossfn(logits, targets)
        return logits


class TransformerSigmoidThresh(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, label_dim, num_labels, nhead, dropout):
        super(TransformerSigmoidThresh, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.label_dim = label_dim
        self.num_labels = num_labels
        self.nhead = nhead
        self.dropout = dropout


        self.embed = torch.nn.Embedding(self.vocab_size, self.hidden_dim)
        transformer_layer = torch.nn.TransformerEncoderLayer(self.hidden_dim,
                                                             self.nhead,
                                                             dropout=self.dropout,
                                                             batch_first=True,
                                                             norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(transformer_layer, num_layers=4)
        # Logits params
        self.logit_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.label_dim),
            torch.nn.Linear(self.label_dim, self.num_labels, bias=False)
        )

        # Threshold params
        self.thresh_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, 1),
        )

        self.lossfn = torch.nn.BCEWithLogitsLoss()
        self.loss = None

    def forward(self, src, labels):
        src_embed = self.embed(src) * math.sqrt(self.hidden_dim)

        # TODO: Improve this - do not want to just take first
        outputs_all = self.encoder(src_embed)
        # outputs = outputs_all[:, 0, :]

        logits = self.logit_mlp(outputs_all[:, 0, :])
        thresh = self.thresh_mlp(outputs_all[:, 0, :])

        logits = logits - thresh
        # outputs_l = self.project_l(outputs)
        # logits = self.classifier_l(outputs_l)
        # logits = torch.tanh(logits)

        # outputs_t = self.project_t(outputs_all[:, -1, :])
        # outputs_t = torch.relu(outputs_t)
        # outputs_t = outputs_t + outputs
        # outputs_t = self.ln(outputs_t)
        # If we do not constrain the threshold - it minimises the loss by
        # making predictions it is super confident about very certain
        # thresh = self.classifier_t(outputs_all[:, -1, :])

        # logits = logits - thresh

        labels = labels.float()
        assert ((labels == 0).sum() + (labels == 1).sum()) == (labels.shape[0] * labels.shape[1])
        self.loss = self.lossfn(logits, labels)
        return logits


class TransformerBlockSigmoid(torch.nn.Module):
    def __init__(self, encoder, label_dim, num_labels, pos_weight=None):
        super(TransformerBlockSigmoid, self).__init__()
        self.encoder = encoder
        self.label_dim = label_dim
        self.num_labels = num_labels
        self.pos_weight = pos_weight

        self.hidden_dim = encoder.pooler.dense.weight.shape[-1]
        # self.hidden_dim = 1024

        assert 512 <= self.hidden_dim <= 1024

        self.num_blocks = self.num_labels // self.label_dim
        self.block_sizes = [self.label_dim for i in range(self.num_blocks)]

        remainder = self.num_labels % self.label_dim
        if remainder:
            self.block_sizes.append(remainder)

        self.block_names = []
        for i, block_size in enumerate(self.block_sizes):
            name = 'encoder_block_%d' % i
            self.block_names.append(name)
            setattr(self, name, torch.nn.Sequential(
                # NOTE: Interestingly, it even works with a linear layer?
                #-------- below block can potentially be removed --------------
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(self.hidden_dim),
                #---------------------------------------------------------
                torch.nn.Linear(self.hidden_dim, self.label_dim),
                # Both adding ReLU and LayerNorm make results
                # significantly worse here
                # ReLU totally makes sense, because negative activations
                # become impossible.
                # However, I have no idea why LayerNorm does not work
                torch.nn.Linear(self.label_dim, block_size, bias=False),
            ))
            assert block_size <= self.label_dim

        if self.pos_weight is not None:
            self.lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones([self.num_labels]) * self.pos_weight)
        else:
            self.lossfn = torch.nn.BCEWithLogitsLoss()
        self.loss = None

    def forward(self, batch_dict):
        # labels: batch_dim x self.num_labels  (int vector of 0s and 1s)
        # the tensor contains the indices of active labels
        # tensor second dim is always max_active_labels size
        # <PAD> token used to pad to max size
        labels = batch_dict.pop('labels', None)

        # Take BERT pooler output
        outputs = self.encoder(**batch_dict).pooler_output
        # outputs = self.encoder(**batch_dict)
        # outputs = self.encoder(**batch_dict).last_hidden_state[:, 0, :]

        block_acts = []
        for i, block in enumerate(self.block_names):
            mlp_block = getattr(self, block)
            act = mlp_block(outputs)
            block_acts.append(act)

        logits = torch.cat(block_acts, dim=1)

        if labels is not None:
            labels = labels.float()
            # assert ((labels == 0).sum() + (labels == 1).sum()) == (labels.shape[0] * labels.shape[1])
            self.loss = self.lossfn(logits, labels)
        return logits


class TransformerSeq2Seq(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, nhead, dropout):
        super(TransformerSeq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.dropout = dropout

        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.transformer = torch.nn.Transformer(d_model=self.hidden_dim,
                                                num_encoder_layers=4,
                                                num_decoder_layers=1,
                                                nhead=self.nhead,
                                                dropout=self.dropout,
                                                batch_first=True,
                                                norm_first=True)
        self.project = torch.nn.Linear(hidden_dim, 2)
        self.decoder = torch.nn.Linear(2, 1)

        self.lossfn = torch.nn.BCEWithLogitsLoss()
        self.loss = None

    def forward(self, src, tgt, labels):
        src_embed = self.embed(src) * math.sqrt(self.hidden_dim)
        tgt_embed = self.embed(tgt) * math.sqrt(self.hidden_dim)

        tgt_len = tgt.shape[1]

        tgt_mask = torch.triu(torch.full((tgt_len, tgt_len), float('-inf'), device=tgt.device),
                              diagonal=1)
        outputs = self.transformer(src_embed, tgt_embed, tgt_mask=tgt_mask)
        outputs = self.project(outputs)
        logits = self.decoder(outputs).squeeze(2)

        labels = labels.float()
        assert ((labels == 0).sum() + (labels == 1).sum()) == (labels.shape[0] * labels.shape[1])
        self.loss = self.lossfn(logits, labels)
        return logits


class MultiLabelModel(torch.nn.Module):
    def __init__(self, encoder, output_layer):
        super().__init__()
        self.encoder = encoder
        self.output_layer = output_layer

        # self.lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=10*torch.ones([self.output_layer.output_dim]))
        self.lossfn = torch.nn.BCEWithLogitsLoss()
        self.loss = None

    def forward(self, inputs, labels):
        outputs = self.encoder(**inputs)
        # If we use a BERT model we need to unpack
        # the dictionary returned
        if isinstance(outputs, dict) and 'pooler_output' in outputs:
            outputs = outputs['pooler_output']
        logits = self.output_layer(outputs)

        labels = labels.float()
        assert ((labels == 0).sum() + (labels == 1).sum()) == (labels.shape[0] * labels.shape[1])
        self.loss = self.lossfn(logits, labels)
        return logits


class FrozenEncoderMultiLabelModel(torch.nn.Module):
    def __init__(self, output_layer):
        super().__init__()
        self.output_layer = output_layer

        # self.lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=10*torch.ones([self.output_layer.output_dim]))
        self.lossfn = torch.nn.BCEWithLogitsLoss()
        self.loss = None
        # self.perm = torch.randperm(output_layer.out_dim)
        # self.iperm = self.perm.argsort()

    def forward(self, inputs, labels):
        # The ordering of the labels usually captures semantic information
        # e.g. how frequent a class is. Since frequent classes fire together
        # it sounds like a good idea to shuffle. We undo the permutation later.
        # NOTE: The Ordered Vandermonde parametrisation fit changes a lot
        # depending on the order of the labels - the frequency order seems
        # like a good inductive bias, since it prefers "contiguous" blocks of ones.

        # labels = labels[:, self.perm]

        logits = self.output_layer(inputs)

        labels = labels.float()
        assert ((labels == 0).sum() + (labels == 1).sum()) == (labels.shape[0] * labels.shape[1])
        labels = self.output_layer.encode(labels)
        self.loss = self.lossfn(logits, labels)

        logits = self.output_layer.decode_logits(logits)

        # logits = logits[:, self.iperm]
        return logits
