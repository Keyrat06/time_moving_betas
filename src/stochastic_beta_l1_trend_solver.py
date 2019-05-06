import numpy as np
from tqdm import trange
import argparse
from src.util import make_data, plot_tends, time_it


@time_it
def beta_l1_trend_solver(y, x, l1=0.01, l2=0.001, a=0.1, b=0.1, iterations=10000, con=100):
    """
    Solves dynamic beta motion using stochastic gradient decent.
    Using momentum gradient decent for additional stability.

    let N = number of samples
    let K = number of features
    let B = [N, K] numpy array

    Formulation:
    L(B) = 1/2(y-Bx)**2 + l1 * abs(dB/dt) + l2 abs(B)
    dL(B)/dB = -x(y-Bx) + l1 * sign(dB/dt) + l2 * sign(B)
    DB_0 = 0
    B_0 = 0
    DB_i+1 = a * (dL(B)/dB_i) + (1-a) DB_i
    B_i+1 = B_i - b * DB_i

    :param y: observed outputs | [N] numpy array
    :param x: observed inputs | [N, K] numpy array
    :param l1: motion loss scale | scalar
    :param l2: shrinkage param | scalar
    :param a: gradient decent momentum param | scalar
    :param b: gradient step | scalar
    :param iterations: max number of iterations | scalar
    :param con: number of turns that current loss can be worse then best loss
    :return: B_-1 [N, K] numpy array, losses, betas
    """
    n, k = x.shape

    def get_loss(beta):
        y_hat = (beta * x).sum(axis=1)
        d_beta_d_t = np.diff(beta, axis=0)
        return 1/2 * ((y-y_hat)**2).sum() + l1 * abs(d_beta_d_t).sum() + l2 * abs(beta).sum()

    def get_d_loss(beta):
        y_hat = (beta * x).sum(axis=1)
        d_beta_d_t = np.diff(beta, axis=0)
        track_loss = -x * (y-y_hat)[:, np.newaxis]
        motion_loss = -1 * np.vstack((np.sign(d_beta_d_t), np.zeros(k)))
        shrink_loss = np.sign(beta)
        return track_loss + l1 * motion_loss + l2 * shrink_loss


    beta = np.zeros((n, k))
    dbeta = np.zeros((n, k))
    losses = []
    dloss = float("inf")
    betas = []
    tbar = trange(iterations)
    for i in tbar:
        # Calculate Gradient and momentum gradient
        dloss_db = get_d_loss(beta)
        dbeta = dloss_db if (dbeta == 0).all() else a * dloss_db + (1-a) * dbeta

        # Record current stats
        loss = get_loss(beta)
        losses.append(loss)
        betas.append(beta)

        # Gradient Descend
        beta -= b * dbeta

        best_beta_index = np.argmin(losses)
        if (i-best_beta_index) > con:
            break

        tbar.set_description("Loss {:.5f} dLoss {:.6f}".format(loss, dloss))
        tbar.refresh()


    loss = get_loss(beta)
    betas.append(beta)
    losses.append(loss)

    beta = betas[np.argmin(losses)]
    return beta, losses, betas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, default=100)
    parser.add_argument("-k", "--features", type=int, default=13)
    parser.add_argument("-l1", "--lambda1", type=float, default=0.01)
    parser.add_argument("-l2", "--lambda2", type=float, default=0.001)
    parser.add_argument("-a", "--alpha", type=float, default=0.5)
    parser.add_argument("-b", "--beta", type=float, default=0.1)
    parser.add_argument("-i", "--iterations", type=int, default=1000)
    parser.add_argument("-s", "--sigma", type=float, default=1e-3)
    parser.add_argument("-c", "--con", type=int, default=100)

    args = parser.parse_args()
    n = args.number
    k = args.features
    l1 = args.lambda1
    l2 = args.lambda2
    a = args.alpha
    b = args.beta
    s = args.sigma
    iters = args.iterations
    con = args.con

    x, y, true_beta = make_data(n, k, s)
    beta_hat, losses, betas = beta_l1_trend_solver(y, x, l1=l1, l2=l2, a=a, b=b, iterations=iters, con=con)

    plot_tends(((true_beta, "true_beta"), (beta_hat, "beta_hat")))
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show("hold")
