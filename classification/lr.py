# coding=utf-8
from functools import partial
import numpy as np
from optimization import lbfgs
from optimization import gd
from optimization import bfgs


class LR(object):
    """
    logistic regression
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
        self.theta = None
        self.lam = lam
        self.debug = debug
        self.normalize = normalize
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
        self.m = x.shape[1]
        self.n = x.shape[0]

        if self.normalize:
            x = self.norm(x)

        self.cls_label = np.unique(np.sort(y))
        if len(self.cls_label) == 2:
            new_y = (y == self.cls_label[1]).astype(np.int)
            self.theta = self.train_2_cls(x, new_y, debug=self.debug)
        else:
            self.theta = np.empty((len(self.cls_label), self.n + 1))
            for i in range(len(self.cls_label)):
                new_y = (y == self.cls_label[i]).astype(np.int)
                self.theta[i, :] = self.train_2_cls(x, new_y, debug=self.debug)

    def train_2_cls(self, x, y, debug=False):
        """
        训练二分类模型
        :param x:
        :param y:
        :param debug:
        :return:
        """
        theta0 = np.ones(self.n + 1) * 0.001
        theta = self.opt_method(
            partial(self.lr_loss, x=np.vstack((np.ones(self.m), x)), y=y),
            partial(self.lr_loss_gradient, x=np.vstack((np.ones(self.m), x)), y=y),
            theta0, debug=debug)
        return theta

    def predict(self, x):
        """
        predict labels of test samples
        :param x: 输入数据，1d array，或者 n*m的二维ndarray，n是特征个数，m是样本个数
        :return: 样本标签
        """
        if self.mean is not None:
            x = self.norm(x)
        scores = self.theta.dot(np.vstack((np.ones(x.shape[1]), x)))
        scores = np.exp(scores)
        if len(self.cls_label) == 2:
            scores = np.vstack((np.ones(x.shape[1]), scores))
        prob = scores / np.sum(scores, 0)
        indices = np.argmax(prob, 0)
        return self.cls_label[indices]

    def lr_loss(self, theta, x, y):
        """
        lr loss function
        :param theta:
        :param x: 数据数据，n*m的二维ndarray，n是特征个数，m是样本个数
        :param y:
        :return:
        """
        cost = 0
        for i in range(x.shape[1]):
            resp = np.exp(theta.dot(x[:, i]))
            prob1 = resp / (1 + resp)
            if prob1 == float('inf'):
                prob1 = 1
            if y[i] == 1:
                prob1 += 1e-10
                cost += np.log(prob1)
            else:
                prob1 -= 1e-10
                cost += np.log(1 - prob1)

        cost /= - x.shape[1]
        # 不规则化theta0
        cost += np.sum(theta[1:x.shape[0]] ** 2 * self.lam / 2)
        # cost += np.sum(theta[0:x.shape[0]] ** 2 * self.lam / 2)

        return cost

    def lr_loss_gradient(self, theta, x, y):
        """
        gradient lr loss function
        :param theta:
        :param x: n*m的二维ndarray，n是特征个数，m是样本个数
        :param y:
        :return:
        """
        prob1 = np.exp(theta.dot(x))
        prob1 /= prob1 + 1
        g = np.sum((prob1 - y) * x, 1)

        g /= x.shape[1]

        g[1:x.shape[0]] += theta[1:x.shape[0]] * self.lam
        # g[0:x.shape[0]] += theta[0:x.shape[0]] * self.lam

        return g
