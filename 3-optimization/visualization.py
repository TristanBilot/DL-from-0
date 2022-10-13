from typing import List

import matplotlib.pyplot as plt


def plot_list(elements: List[float], label=""):
    plt.plot(elements, label=label)
    plt.legend(loc="upper left")
    plt.show()
