import numpy as np


def svm_read_problem(file_name):
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
