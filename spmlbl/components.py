import torch
import torch.nn.functional as F
import numpy as np

from spmlbl.modules import torch_interleave_columns, gale
from scipy.special import logit


class TransformerEncoder(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, nhead=8, nlayers=2, dropout=0.):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.nlayers = nlayers
        self.dropout = dropout

        self.embed = torch.nn.Embedding(self.vocab_size, self.hidden_dim)
        transformer_layer = torch.nn.TransformerEncoderLayer(self.hidden_dim,
                                                             self.nhead,
                                                             dropout=self.dropout,
                                                             batch_first=True,
                                                             norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(transformer_layer, num_layers=self.nlayers)

    def forward(self, src, src_key_padding_mask=None):
        src_embed = self.embed(src)

        # TODO: Improve this - do not want to just take first
        outputs = self.encoder(src_embed, src_key_padding_mask=src_key_padding_mask)[:, 0, :]
        return outputs


class SigmoidBottleneckLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, feature_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feature_dim = feature_dim

        self.mlp = torch.nn.Sequential(
            # Project from input dim to label dim
            torch.nn.Linear(self.in_dim, self.feature_dim, bias=True),
            torch.nn.Linear(self.feature_dim, self.out_dim, bias=False)
        )

    def forward(self, src):
        logits = self.mlp(src)
        return logits

    def encode(self, labels):
        return labels

    def decode(self, labels):
        return labels

    def decode_logits(self, logits):
        return logits


class KSparseFFTClassifier(torch.nn.Module):
    """
    A Linear layer that guarantees that k-sparse labels are argmaxable.
    For multi-label classification this means that outputs with k labels "on" are argmaxable.
    """
    def __init__(self, in_dim, out_dim, k, slack_dims=0, use_init=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.slack_dims = slack_dims
        self.use_init = use_init

        self.D = 2 * self.k + 1
        self.N = out_dim

        self.total_D = self.D + self.slack_dims

        self.proj = torch.nn.Linear(self.in_dim, self.total_D, bias=True)

        if self.use_init:
            # The first column of W is the all 1/np.sqrt(N) vector
            # let's initialise the bias of the linear projection
            # such that the output is the MLE.
            # This makes sense since we know that our labels are sparse
            mle = self.k / self.N
            self.proj.bias.data[0] = logit(mle) * np.sqrt(self.N)

        # Slack dimensions
        if self.slack_dims:
            self.Ws = torch.nn.Linear(self.slack_dims, self.N, bias=False)

    def compute_W(self):
        Wc = torch.tensor(gale(self.N, self.D),
                          dtype=torch.float32,
                          device=self.proj.weight.device)
        # Wc[:, 1:] = Wc[:, 1:] / np.sqrt(2) 
        # print(Wc.T @ Wc)
        # _, s, _ = np.linalg.svd(Wc.detach().cpu().numpy())
        # print(s)
        # print(np.linalg.cond(Wc.detach().cpu().numpy()))

        if self.slack_dims:
            Ws = torch.hstack([Wc, self.Ws.weight])
        else:
            Ws = Wc
        return Ws

    def forward(self, xx):

        bs, dim = xx.shape

        xx = self.proj(xx)
        # FFT does not make cols orthonormal - multiply here to make
        # forward and mat mul with computed W equivalent
        xx[:, 1:self.D] = xx[:, 1:self.D] * np.sqrt(2)

        # Interpret activation as complex number for dft
        dc = torch.complex(xx[:, 0], torch.zeros(bs, device=xx.device))
        # Use conjugate as cyclic polytope is framed with + sign for sin
        xxc = xx[:, 1:self.D].view(bs, -1, 2)
        cx = torch.complex(xxc[:, :, 0], xxc[:, :, 1])
        cx = cx.conj()

        # Concatenate dc term and cos, sin terms
        cx = torch.cat([dc.view(-1, 1), cx], axis=1)

        yy_dft = torch.fft.ifft(cx, n=self.N, norm='ortho', dim=-1).real

        if self.slack_dims > 0:
            yy_slack = self.Ws(xx[:, self.D:])

            yy = yy_dft + yy_slack
        else:
            yy = yy_dft
        return yy

    def encode(self, labels):
        return labels

    def decode(self, labels):
        return labels

    def decode_logits(self, logits):
        return logits
