import math
import torch
import numpy as np


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
