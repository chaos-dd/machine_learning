# coding=utf-8

import numpy as np


def optimize(func, grad, x, eps=1e-4, max_iter=300, debug=False):
    """
    find minium value of input function f by gradient descent
    :param func:   object function, accept numpy 1d array as input, output scalar value
    :param grad:   gradient function, acccept numpy 1d arrya as input, output 1d array of gradient values
    :param x: 1d array
    :param max_iter: max iteration times
    :param debug: whethe to print debug info
    :param eps: epsilon error threshold
    :return:
    """
    it = 0
    x = np.mat(x.reshape(len(x), 1))
    y = func(x.A1)
    g = np.mat(grad(x.A1)).T
    while True:
        it += 1
        step, new_x, new_y = back_track_line_search(func, grad, x, -g)
        new_g = np.mat(grad(new_x.A1)).T
        # 更新
        new_diff_y = y - new_y
        x, y, g = new_x, new_y, new_g
        if debug:
            print('gd', 'iter:', it, 'step:', step, 'diff of y:', new_diff_y, 'y:', y)

        if new_diff_y < eps:
            print('gd finished: diff_y is small than eps = %s !!' % eps)
            break
        if np.sum(g.A ** 2) < eps:
            print('gd finished: norm of g is small than eps = %s !!' % eps)
            break
        if it > max_iter:
            print('gd finished: iteration number exceed max limit = %s !!' % max_iter)
            break
    return new_x.A1


def back_track_line_search(f, grad, x, d, eps=1e-4):
    """
    回溯直线法
    :param f:目标函数
    :param grad: 梯度函数
    :param x: n*1 ndarray
    :param d:
    :param eps:
    :return:
    """
    step = 1
    alpha = 0.001
    beta = 0.8
    # while step > eps and f((x + step * d).A1) >= f(x.A1):
    while step > eps:
        new_x = x + step * d
        new_y = f(new_x.A1)
        tmp = f(x.A1) + alpha * step * (np.mat(grad(x.A1)) * d)[0, 0]

        if new_y < tmp:
            break
        step *= beta
    return step, new_x, new_y


def g(x):
    return 2 * x


def f(x):
    return x ** 2


if __name__ == '__main__':
    min_x = optimize(f, g, np.array([100]), debug=True)
    print(min_x)
