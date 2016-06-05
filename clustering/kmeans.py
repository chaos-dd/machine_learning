import numpy as np
import random


def randomized_center(data, k):
    ind = random.sample(range(0, np.shape(data)[0]), k)
    return data[ind]


def dist(arr1, arr2):
    return np.power(np.sum(np.power(arr1 - arr2, 2)), 0.5)


def calc_centers(data, indice, k):
    centers = np.empty([k, np.shape(data)[1]])

    for i in range(0, k):
        centers[i] = np.average(data[indice == i], 0)
    return centers


def assign_sample(data, center):
    rows, cols = data.shape
    K = center.shape[0]

    dist_arr = np.empty([rows, 1])
    indice = np.array(range(rows))

    for r in range(rows):

        min_d = np.inf
        ind = 0
        for k in range(K):
            d = dist(data[r], center[k])
            if d < min_d:
                min_d = d
                ind = k
        dist_arr[r] = min_d
        indice[r] = ind

    return indice, sum(sum(dist_arr))


def kmeans(data, K, times=10):
    min_cost = np.inf
    for i in range(times):
        centers = randomized_center(data, K)
        cnt = 0
        last_cost = np.inf
        while True:
            indice, cost = assign_sample(data, centers)
            print(cost)

            if (last_cost - cost) / cost < 0.0001 or cnt > 100:
                break
            centers = calc_centers(data, indice, K)
            last_cost = cost
            cnt += 1

        if cost < min_cost:
            min_cost = cost
            rv_centers = centers
            rv_indice = indice
    return rv_centers, rv_indice, cost


if __name__ == '__main__':
    a1 = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])

    print(randomized_center(a1, 2))
