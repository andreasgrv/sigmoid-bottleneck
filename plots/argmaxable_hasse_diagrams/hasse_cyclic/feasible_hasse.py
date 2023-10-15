from itertools import combinations
from collections import defaultdict
from math import comb as binomial_coefficient
from argparse import ArgumentParser

from spmlbl.verify import pattern_is_feasible_cyclic


def hamming_distance(a, b):
    assert len(a) == len(b)
    diff = 0
    for i in range(len(a)):
        diff += a[i] != b[i]
    return diff


def generate_hasse_diagram_tikz(N, C):
    tikz_code = "\\documentclass{standalone}\n"
    tikz_code += "\\usepackage[dvipsnames]{xcolor}\n"
    tikz_code += "\\usepackage{tikz}\n"
    tikz_code += "\\usepackage{bm}\n"
    tikz_code += "\\begin{document}\n"
    tikz_code += "\\begin{tikzpicture}[scale=1.5]\n"

    max_nodes_per_level = binomial_coefficient(N, N//2)
    WIDTH =  max_nodes_per_level / 1.5
    SCALE_HEIGHT = 1.2
    SCALE_WIDTH = .7
    NODE_SCALE = .7
    if N > 8:
        SCALE_WIDTH = .2
        NODE_SCALE = .7

    node_dict = defaultdict(dict)

    top = .96 * SCALE_HEIGHT * N
    tikz_code += r'\node at ({}, {}) {{\LARGE Enforcing argmaxable ${}$-active label assignments}};'.format(WIDTH * SCALE_WIDTH / 2, top, C)
    for k in range(N + 1):
        nodes_in_level = binomial_coefficient(N, k)
        nodes_in_next_level = max(binomial_coefficient(N, k+1), 1)
        LEVEL_SCALE = .6 * (nodes_in_level / max_nodes_per_level)
        # NEXT_LEVEL_SCALE = .5 * (nodes_in_next_level / max_nodes_per_level)
        NEXT_LEVEL_SCALE = 1.
        combs = list(combinations(range(N), k))
        # Instead of indices represent as bitstring
        nodes = []
        for c in combs:
            node = ['0',] * N
            for i in c:
                node[i] = '1'
            nodes.append(''.join(node))

        width = WIDTH * SCALE_WIDTH
        height = SCALE_HEIGHT * (k - NEXT_LEVEL_SCALE)

        tikz_code += r'\node[draw, color={}] at ({}, {}) {{k={}}};'.format('ForestGreen' if k <= C or k >= N-C else 'Black', -.9, height, k)
        for i, node in enumerate(nodes):
            node_dict[k][i] = node
            color = 'ForestGreen' if pattern_is_feasible_cyclic(C, node) else 'Red'
            node_str = ''.join('$\\bm{+}$' if x == '1' else '$\\bm{-}$' for x in node)
            tikz_code += "\\node[draw, circle, scale={}, inner color={}, outer color={}!80!black] (L{}N{}) at ({}*{}/{}, {}) {{{}}};\n".format(
                NODE_SCALE * (1 - LEVEL_SCALE), color, color, k, i, i+1, width, nodes_in_level+1, height, node_str
            )

    # Only plot edges if n is small enough (too much clutter otherwise)
    if N <= 5:
        for level_from in range(N):
            for node_from in range(binomial_coefficient(N, level_from)):
                s_node_from = node_dict[level_from][node_from]
                for node_to in range(binomial_coefficient(N, level_from + 1)):
                    s_node_to = node_dict[level_from + 1][node_to]
                    if hamming_distance(s_node_from, s_node_to) == 1:
                        tikz_code += "\\draw (L{}N{}) -- (L{}N{});\n".format(level_from, node_from, level_from + 1, node_to)

    tikz_code += "\\end{tikzpicture}\n"
    tikz_code += "\\end{document}\n"

    return tikz_code


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--N', default=8, type=int)
    parser.add_argument('--C', default=1, type=int)
    args = parser.parse_args()

    tikz_code = generate_hasse_diagram_tikz(args.N, args.C)
    print(tikz_code)
