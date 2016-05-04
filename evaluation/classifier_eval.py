# coding=utf-8
import numpy as np
import random
import copy


def eval(classifier, x, y, fold=5, debug=False):
    """
    cross validation
    :param classifier:
    :param x:
    :param y:
    :param fold:
    :param debug:
    :return:
    """
    n, m = x.shape

    index = np.ones(m, np.int32)

    if m < fold:
        fold = m

    sz = int(m / fold)

    labels = [i for i in range(fold)] * sz
    for i in range(len(labels)):
        index[i] = labels[i]

    random.shuffle(index)

    for i in range(fold):
        train_set = x[:, index != i]
        train_label = y[index != i]
        test_set = x[:, index == i]
        test_label = y[index == i]

        test_classifier = copy.deepcopy(classifier)
        test_classifier.train(train_set, train_label)
        predict_y = test_classifier.predict(test_set)
        err_list = predict_y == test_label

        corr = 0
        for err in err_list:
            if err:
                corr += 1
        print("fold:", i, "precision:", corr / len(err_list))
