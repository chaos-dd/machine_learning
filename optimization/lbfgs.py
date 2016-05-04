# coding=utf-8

import numpy as np


def optimize(func, grad, x, eps=1e-4, m=20, max_iter=1000, debug=False):
    """
    find minium value of input function f by lbfgs
    :param func:   object function, accept numpy 1d array as input, output scalar value
    :param grad:   gradient function, acccept numpy 1d arrya as input, output 1d array of gradient values
    :param x: 1d array
    :param eps: epsilon error threshold
    :param m step number used to calculate Hession matrix
    :param max_iter: max iteration times
    :param debug: whethe to print debug info
    :return:
    """
    x = np.mat(x).T
    k = 0
    diff_y = 1

    dim = x.shape[0]
    s = np.mat(np.zeros((dim, m), np.float64))

    t = s.copy()
    rho = np.mat(np.zeros((m, 1), np.float64))
    g = np.mat(grad(x.A1)).T

    y = func(x.A1)

    it = 0
    while True:
        it += 1
        k += 1

        current = k % m
        d = - get_h_dot_g(g, s, t, rho, k, m)
        step, new_x, new_y = back_track_line_search(func, grad, x, d, eps)
        s[:, current] = step * d
        new_g = np.mat(grad(new_x.A1)).T

        new_diff_y = y - new_y
        # 更新
        t[:, current] = new_g - g

        cur_rho = t[:, current].T * s[:, current]

        rho[current, 0] = 1 / cur_rho if cur_rho != 0 else 0
        x, y, g = new_x, new_y, new_g
        if debug:
            print('lbfgs', 'iter:', k, 'step:', step, 'diff_y:', diff_y, 'y:', y)
        diff_y = new_diff_y

        if new_diff_y < eps:
            print('lbfgs finished: diff_y =(%s) is small than eps = %s !!' % (new_diff_y, eps))
            break
        if np.sum(g.A ** 2) < eps:
            print('lbfgs finished: norm of g is small than eps = %s !!' % eps)
            break
        if it > max_iter:
            print('lbfgs finished: iteration number exceed max limit = %s !!' % max_iter)
            break

    return new_x.A1


def get_h_dot_g(g, s, t, rho, k, m):
    delta = 0 if k <= m else k - m
    L = k if k <= m else m
    alpha = np.mat(np.zeros((L, 1), np.float64))

    beta = alpha.copy()
    q = np.mat(np.zeros((g.shape[0], L + 1), np.float64))
    q[:, L] = g

    for i in range(L - 1, -1, -1):
        j = (i + delta) % m
        alpha[i, 0] = s[:, j].T * q[:, i + 1] * rho[j, 0]

    # z = q.copy()
    z = q[:, L]
    for i in range(0, L, 1):
        j = (i + delta) % m
        beta[j, 0] = (t[:, j].T * z * rho[j, 0])[0]
        z = z + s[:, j] * (alpha[i] - beta[i])
    return z


def back_track_line_search(f, grad, x, d, eps=1e-4):
    """
    回溯直线法
    :param f:目标函数
    :param grad: 梯度函数
    :param x:
    :param d:
    :return:
    """
    d = d.reshape((d.size, 1))
    step = 1
    alpha = 0.001
    beta = 0.8
    while step > eps:
        new_x = x + step * d
        new_y = f(new_x.A1)
        tmp = f(x.A1) + alpha * step * (np.mat(grad(x.A1)) * d)[0, 0]
        if new_y < tmp:
            break
        step *= beta
    return step, new_x, new_y
