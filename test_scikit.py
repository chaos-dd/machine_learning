import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.svm as svm


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


def test_lr(x, y, C):
    lr = LogisticRegression(C=C, solver='lbfgs')
    lr.fit(x, y)

    predict_y = lr.predict(x)

    err_list = predict_y == y

    corr = 0
    for err in err_list:
        if err:
            corr += 1
    print("lr precision:", corr / len(err_list))


def test_svm(x, y, C):
    svmcc = svm.LinearSVC(C=C)
    svmcc.fit(x, y)

    predict_y = svmcc.predict(x)

    err_list = predict_y == y

    corr = 0
    for err in err_list:
        if err:
            corr += 1
    print("svmcc precision:", corr / len(err_list))


def test_svm2(x, y, C):
    svmcc = svm.SVC(kernel='linear', C=C)
    svmcc.fit(x, y)

    predict_y = svmcc.predict(x)

    err_list = predict_y == y

    corr = 0
    for err in err_list:
        if err:
            corr += 1
    print("svm svc precision:", corr / len(err_list))


def normalize(x):
    m = np.mean(x, 1)
    std = np.std(x, 1)

    m.shape = (m.size, 1)
    std.shape = (std.size, 1)
    x = (x - m) / std
    return x, m, std


x, y = svm_read_problem_with_rc("D:\\train_data\\data.libsvm")

# min_x = np.min(x, 1)
# max_x = np.max(x, 1)
# range_x = max_x - min_x
# range_x[range_x == 0] = 1
# x = (x - min_x.reshape((len(min_x), 1))) / range_x.reshape((len(range_x), 1))

x = x.T

x, m, std = normalize(x)
c = 100000
test_lr(x, y, 1)
test_svm(x, y, 1)
test_svm2(x, y, 1)
print()
