# coding=utf-8
from functools import partial
import numpy as np
from optimization import lbfgs, bfgs
from optimization import gd


class SoftMax(object):
    """
    softmax classifier, input class label must start from 0, ie. [0,1,2....N]
    """

    def __init__(self, lam=1, normalize=True, opt='lbfgs', debug=False):
        """
        初始化
        :param lam: lambda，系数规则化因子
        :param normalize: 是否对输入数据归一化
        :param opt: 优化方法
        :param debug: 是否打印debug信息
        :return:
        """
        self.mean = None
        self.std = None
        self.m = 0
        self.n = 0
        self.k = 0
        self.label_set = set()
        self.theta = None
        self.lam = lam
        self.normalize = normalize
        self.debug = debug
        self.opt = opt
        self.cls_label = None
        if self.opt == 'lbfgs':
            self.opt_method = lbfgs.optimize
        elif self.opt == 'gd':
            self.opt_method = gd.optimize
        elif self.opt == 'bfgs':
            self.opt_method = bfgs.optimize
        else:
            raise Exception("optimize method does not supported !")

    def norm(self, x):
        if self.mean is None:
            self.mean = np.mean(x, 1)
            self.std = np.std(x, 1)
        x = (x - self.mean.reshape((self.mean.size, 1))) / self.std.reshape((self.std.size, 1))
        return x

    def train(self, x, y):
        """
        训练分类器
        :param x: 输入数据，n*m的二维ndarray，n是特征个数，m是样本个数
        :param y: 输入类别，ndarray，从0开始
        :return:
        """
        self.label_set = self.label_set.union(y)
        self.m = x.shape[1]
        self.n = x.shape[0]
        self.k = len(self.label_set)

        theta0 = np.ones((self.k, self.n + 1)) * 0.00001

        if self.normalize:
            x = self.norm(x)
        theta = self.opt_method(
            partial(self.softmax_loss, x=np.vstack((np.ones(self.m), x)), y=y),
            partial(self.softmax_loss_gradient, x=np.vstack((np.ones(self.m), x)), y=y),
            theta0.ravel(), debug=self.debug)
        self.theta = theta.reshape((self.k, self.n + 1))

    def predict(self, x):
        """
        predict labels of test samples
        :param x: 输入数据，1d array，或者 n*m的二维ndarray，n是特征个数，m是样本个数
        :return: 样本标签
        """
        if self.mean is not None:
            x = self.norm(x)

        resp = np.exp(self.theta.dot(np.vstack((np.ones(x.shape[1]), x))))
        prob = resp / np.sum(resp, 0)
        label = np.argmax(prob, 0)

        return np.asarray(label).reshape(label.size)

    def softmax_loss(self, theta, x, y):
        """
        softmax loss function
        :param theta: parameters, theta=[ theta1.T ; theta2.T; ...;thetak.T]
        :param x:
        :param y:
        :return:
        """
        m = self.m
        n = self.n
        k = self.k

        theta = np.mat(theta)
        x = np.mat(x)
        y = np.mat(y).reshape(y.size, 1)
        theta = theta.reshape((k, n + 1))

        resp = np.exp(theta * x)

        sample_sum = np.sum(resp, 0)
        cost = 0
        for i in range(m):
            cost += np.log(resp[y[i, 0], i] / sample_sum[0, i])

        cost += np.sum(np.asarray(theta[0:k, 1:n + 1]) ** 2 * self.lam) / 2

        cost /= (-m)

        return cost

    def softmax_loss_gradient(self, theta, x, y):
        """
        softmax 损失函数的梯度
        :param theta: k*（n+1)维矩阵
        :param x: n*m的二维ndarray，n是特征个数，m是样本个数
        :param y:
        :return:
        """
        m = self.m
        n = self.n
        k = self.k
        theta = np.mat(theta)
        x = np.mat(x)
        y = np.mat(y).reshape(y.size, 1)
        theta = theta.reshape((k, n + 1))

        prob = np.exp(theta * x)
        prob_sum = np.sum(prob, 0)
        prob /= prob_sum
        prob = -prob

        g = np.mat(np.zeros(theta.shape, np.float64))
        for i in range(k):
            for j in range(m):
                if y[j, 0] == i:
                    prob[i, j] += 1
                g[i, :] += x[:, j].T * prob[i, j]

            g[i, 1:n + 1] += self.lam * theta[i, 1:n + 1]
            g[i, :] /= (-m)
        return g.ravel()
