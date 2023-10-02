import numpy as np

import itertools
from spmlbl.verifier import ArgmaxableSubsetVerifier


matrix = r"""
\begin{tikzpicture}[mycolour,scale=1]
    \matrix [matrix of math nodes, left delimiter=[, right delimiter=],ampersand replacement=\&](A){ 
    %s
    };
    \node[left of=A, xshift=-20pt](L) {$\vv{W}=$};
\end{tikzpicture}
"""

arrangement = r"""
\begin{tikzpicture}[mycolour,scale=2]
    \def \scale {10};
    \begin{scope}
        \clip(0, 0) circle (1.35);
    %s
    \end{scope}
\end{tikzpicture}
"""

def generate_tikz_table(W):
    rs = ''
    for i, row in enumerate(W, 1):
        cc = []
        for c in row:
            cs = '\t'
            cs += r'\textcolor{clr%d}{%.1f}' % (i, c)
            cc.append(cs)
        rs += r' \& '.join(cc) + r' \\' + '\n'
    rs = matrix % rs
    return rs

def generate_tikz_arrangement(W,
                              idxs=None,
                              draw_arrows=False,
                              draw_hyperplanes=False,
                              draw_sign_vecs=False,
                              draw_circles=False):
    N, D = W.shape
    idxs = idxs or list(range(N))
    rs = ''
    for idx in idxs:
        row = W[idx]
        i = idx + 1
        rs += '\t'
        rs += r'\node[](w%d) at (%.2f, %.2f){};' % (i, row[0], row[1])
        rs += '\n'
        if draw_hyperplanes:
            rs += '\t'
            rs += r'\draw[clr%d, thick] let \p{w%d}=(w%d) in (\scale * \y{w%d}, -\scale * \x{w%d}) -- (-\scale * \y{w%d}, \scale * \x{w%d});' % (i, i, i, i, i, i, i)
            rs += '\n'
    for idx in idxs:
        row = W[idx]
        i = idx + 1
        if draw_arrows:
            rs += '\t'
            rs += '\draw[-{Latex[length=3.5mm]}, thick, clr%d] (0, 0) -- (w%d);' % (i, i)
            rs += '\n'
        if draw_arrows:
            rs += '\t'
            rs += r'\node[] at (%.2f, %.2f){\textcolor{clr%d}{$\mathbf{w}_%d$}};' % (row[0], row[1], i, i)
            rs += '\n'
    rs += '\n'
    ver = ArgmaxableSubsetVerifier(W[idxs])
    all_subsets = [[i for i, b in enumerate(s) if b]
                   for s in itertools.product([0, 1], repeat=len(idxs))]
    res = ver(all_subsets, lb=-1, ub=1)
    for r in res:
        if r['is_feasible']:
            sv = ['-'] * len(idxs)
            for idx in r['pos_idxs']:
                sv[idx] = '+'
            sv = ''.join(sv)
            xx, yy = r['point']
            if draw_sign_vecs:
                rs += '\t'
                rs += r'\sv{%s}{%.2f}{%.2f}{%s};' % (sv, xx, yy, sv)
                rs += '\n'
            if draw_circles:
                rs += '\t'
                rs += r'\draw[]({%.2f},{%.2f}) circle (%.5f);' % (xx, yy, r['radius'])
                rs += '\n'
    rs = arrangement % rs
    return rs


if __name__ == "__main__":

    W = np.array([[1, 0],
                  [.5, .7],
                  [0, 1],
                  [-.5, .5]])

    for i, j in itertools.combinations(range(W.shape[0]), 2):
        print(np.linalg.det(W[[i, j]]))
    # Plot 1:
    print(generate_tikz_arrangement(W, idxs=[0,1,3], draw_sign_vecs=True, draw_hyperplanes=True))
    print()
    print(generate_tikz_arrangement(W, idxs=[0,1,3], draw_sign_vecs=True, draw_hyperplanes=True, draw_circles=True))
    print()
    print(generate_tikz_arrangement(W, draw_hyperplanes=True, draw_sign_vecs=True))
    print(generate_tikz_arrangement(W, draw_arrows=True, draw_hyperplanes=True))
    print(generate_tikz_table(W))
