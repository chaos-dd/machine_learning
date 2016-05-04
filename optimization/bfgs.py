# coding=utf-8
import numpy as np


def optimize(func, grad, x, eps=1e-4, max_iter=1000, debug=False):
    """
    find minium value of input function f by bfgs
    :param func:
    :param grad:
    :param x:
    :param eps:
    :param max_iter:
    :param debug:
    :return:
    """
    it = 0
    diff_y = 1
    x = np.asmatrix(x).T
    y = func(x.A1)
    # hession矩阵的近似
    b = np.mat(np.eye(x.shape[0]))
    # 单位阵
    I = np.mat(np.eye(x.shape[0]))
    # 梯度
    g = np.mat(grad(x.A1)).T
    # 搜索方向
    while True:
        it += 1
        d = -b * g
        step,new_x,new_y = back_track_line_search(func, grad, x, d, eps)
        ds = step * d
        new_g = np.mat(grad(new_x.A1)).T
        dy = new_g - g
        # new_b = (I - ds * dy.T / dy.T * ds) * b * (I - dy * ds.T / dy.T * ds) + ds * ds.T / dy.T * ds
        new_b = b + (ds.T * dy + dy.T * b * dy)[0, 0] * (ds * ds.T) / ((ds.T * dy)[0, 0] ** 2) \
                - (b * dy * ds.T + ds * dy.T * b) / (ds.T * dy)[0, 0]

        new_diff_y = y - new_y
        # 更新
        x, y, g, b = new_x, new_y, new_g, new_b
        if debug:
            print('bfgs', 'iter:', it, 'step:', step, 'diff_y:', diff_y, 'y:', y)
        diff_y = new_diff_y
        if new_diff_y < eps:
            print('bfgs finished: diff_y =(%s) is small than eps = %s !!' % (new_diff_y, eps))
            break
        if np.sum(g.A ** 2) < eps:
            print('bfgs finished: norm of g is small than eps = %s !!' % eps)
            break
        if it > max_iter:
            print('bfgs finished: iteration number exceed max limit = %s !!' % max_iter)
            break
    return new_x.A1


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


def g(x):
    return 2 * x


def f(x):
    return x ** 2


if __name__ == '__main__':
    min_x = optimize(f, g, np.array([100000]), debug=True)
    print(min_x)
