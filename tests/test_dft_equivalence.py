import torch
import numpy as np
from spmlbl.modules import gale
from spmlbl.components import KSparseFFTClassifier


def test_dft_equivalence():
    # This test checks equivalence of fixed cyclic polytope layer
    # and DFT layer that pads higher frequencies with zeroes
    BS, N, CARD = 10, 20, 8
    IN_DIM = 2 * CARD + 1

    xx = torch.randn(BS, IN_DIM)

    xx_np = xx.cpu().detach().numpy()
    # Pytorch's FFT with 'ortho' normalisation divides all by sqrt(1/N).
    # when actually for this case we need to multiply DC term by sqrt(1/N).
    # and all cos & sin terms by sqrt(2/N).
    W = gale(N, 2*CARD+1)
    W[:, 1:] = W[:, 1:] / np.sqrt(2)
    gale_res = [W.dot(x) for x in xx_np]
    gale_res = np.vstack(gale_res)

    torch.manual_seed(0)
    dc = torch.complex(xx[:, 0], torch.zeros(BS, device=xx.device))
    # Use conjugate as cyclic polytope is framed with + sign for sin
    cx = torch.view_as_complex(xx[:, 1:].view(BS, -1, 2).contiguous()).conj()

    # Concatenate dc term and cos, sin terms
    cx = torch.cat([dc.view(-1, 1), cx], axis=1)

    y = torch.fft.ifft(cx, n=N, norm='ortho', dim=-1).real
    fft_res = y.cpu().detach().numpy()

    assert np.allclose(gale_res, fft_res)

    BS, N, IN_DIM = 10, 13, 20

    xx = torch.randn(BS, IN_DIM)

    clf = KSparseFFTClassifier(IN_DIM, N, k=5)

    fft = clf(xx)

    xxp = clf.proj(xx)
    W = clf.compute_W()
    manual_gale = (W @ xxp.T).T

    assert torch.allclose(fft, manual_gale, atol=1e-4)


def test_dft_and_manual_matmul_equivalence():
    BS, N, IN_DIM = 10, 8000, 200

    xx = torch.randn(BS, IN_DIM)

    clf = KSparseFFTClassifier(IN_DIM, N, k=5, slack_dims=32)
    fft = clf(xx)

    xxp = clf.proj(xx)
    W = clf.compute_W()
    manual_gale = (W @ xxp.T).T

    assert torch.allclose(fft, manual_gale)


if __name__ == "__main__":

    # test_dft_equivalence()
    test_dft_and_manual_matmul_equivalence()
