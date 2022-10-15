from typing import List, Tuple

import matplotlib.pyplot as plt


def plot_list(elements: List[float], label=""):
    plt.plot(elements, label=label)
    plt.legend(loc="upper right")
    plt.show()

def plot_lists(elements: List[Tuple[List[int], str]]):
    for l, label in elements:
        plt.plot(l, label=label)
        plt.legend(loc="upper right")
    plt.show()
