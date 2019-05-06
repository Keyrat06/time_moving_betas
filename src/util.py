import numpy as np
import matplotlib.pyplot as plt
import time
from functools import wraps
plt.ion()


def time_it(func):
    @wraps(func)
    def timed(*args, **kwargs):
        b4 = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print('{} took  {:3.3f} ms'.format(func.__name__, (te - b4) * 1000))
        return result
    return timed


def make_data(n=100, k=10, s=0.01):
    x = np.random.normal(size=(n, k))
    b = np.zeros((n, k))
    b[0, :] = np.random.binomial(1, p=0.5, size=k) * np.random.normal(0.1, 0.4, size=k)
    for i in range(1, n):
        b[i, :] = b[i-1, :]
        if np.random.random() < 0.2:
            for j in range(k):
                p = np.random.random()
                if b[i, j] == 0:
                    if p < 0.05:
                        b[i, j] = np.random.normal(0.1, 0.3)
                else:
                    if p < 0.05:
                        b[i, j] = 0
                    else:
                        b[i, j] += np.random.normal(0, 0.03)
    y = (x*b).sum(axis=1) + np.random.normal(0, s, n)
    return x, y, b

def plot_tends(betas):
    fig, axs = plt.subplots(len(betas), 1, figsize=(14, 7))
    fig.tight_layout()

    for i, (beta, title) in enumerate(betas):
        im = axs[i].imshow(beta.T)
        axs[i].set_title(title)

    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.1, 0.04, 0.8, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

    plt.show("hold")


if __name__ == "__main__":
    x, y, b = make_data()
    plot_tends(((b, "lalala"), (b, "lalala 2")))