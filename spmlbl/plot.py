import matplotlib.pyplot as plt

from collections import defaultdict


def plot_cardinalities(examples, title):
    counts = defaultdict(int)
    for row in examples:
        card = len(row)
        counts[card] += 1
    fig, ax = plt.subplots()
    # for k, v in counts.items():
    ax.bar(tuple(counts.keys()), tuple(counts.values()))
    ax.set_xlabel('Number of Active Labels')
    ax.set_ylabel('Number of Examples')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
