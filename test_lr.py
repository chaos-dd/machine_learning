# coding=utf-8

import numpy as np
from classification import *
import evaluation.classifier_eval as ce


def svm_read_problem_with_rc(file_name):
    data = []
    y = []
    with open(file_name) as f:

        max_dim = 0
        for line in f:
            if line is None or line.strip() == '':
                continue

            line = line.split(None, 1)

            if len(line) == 1:
                line += " "
            label, feat = line
            y.append(int(label) - 1)

            sample = {}
            for p in feat.split():
                ind, val = p.split(":")
                ind = int(ind) - 1
                sample[ind] = float(val)
                if ind > max_dim:
                    max_dim = ind
            data.append(sample)

    x = np.zeros((max_dim + 1, len(data)), np.float64)

    for r in range(len(data)):
        for k, v in data[r].items():
            x[k, r] = v
    y = np.array(y)

    return x, y


x, y = svm_read_problem_with_rc("D:\\train_data\\data.libsvm")

print(x.shape, y.shape)

# classifier = LR(lam=0.0, normalize=True, debug=True, opt='gd')
classifier = LR(lam=0.01, normalize=True, debug=False, opt='lbfgs')
# classifier = LR(lam=0, normalize=True, debug=True, opt='bfgs')
# classifier = LR(lam=0.1, normalize=True, debug=True, opt='lbfgs')

classifier.train(x, y)
predict_y = classifier.predict(x)
err_list = predict_y == y

corr = 0
for err in err_list:
    if err:
        corr += 1
print("precision:", corr / len(err_list))

ce.eval(classifier, x, y)
