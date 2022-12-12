import numpy as np
import math
import matplotlib.pyplot as plt

dataset1 = np.load('hac/dataset1.npy')
dataset2 = np.load('hac/dataset2.npy')
dataset3 = np.load('hac/dataset3.npy')
dataset4 = np.load('hac/dataset4.npy')


def calculate_distance(p1, p2, d):
    total = 0
    for i in range(d):
        total += pow(p1[i] - p2[i], 2)
    distance = math.sqrt(total)
    return distance


def single_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the single linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """

    link = None
    for i in c1:
        for j in c2:
            distance = calculate_distance(i, j, len(i))
            if link is None:
                link = distance
            elif distance < link:
                link = distance
    return link


def complete_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the complete linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    link = None
    for i in c1:
        for j in c2:
            distance = calculate_distance(i, j, len(i))
            if link is None:
                link = distance
            elif distance > link:
                link = distance
    return link


def average_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the average linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    distance = 0
    noOfAllPairs: int = len(c1) * len(c2)
    for i in c1:
        for j in c2:
            distance += calculate_distance(i, j, len(i))
    return distance / noOfAllPairs


def centroid_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the centroid linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """

    points1, points2 = np.zeros((len(c1[0]))), np.zeros((len(c2[0])))

    for i in c1:
        for j in range(len(i)):
            points1[j] += i[j]
    for k in c2:
        for m in range(len(k)):
            points2[m] += k[m]
    for a in range(len(points1)):
        points1[a] /= len(c1)
    for b in range(len(points2)):
        points2[b] /= len(c2)

    link = calculate_distance(points1, points2, len(points1))
    return link


def hac(data, criterion, stop_length):
    """
    Applies hierarchical agglomerative clustering algorithm with the given criterion on the data
    until the number of clusters reaches the stop_length.
    :param data: An (N, D) shaped numpy array containing all of the data points.
    :param criterion: A function. It can be single_linkage, complete_linkage, average_linkage, or
    centroid_linkage
    :param stop_length: An integer. The length at which the algorithm stops.
    :return: A list of numpy arrays with length stop_length. Each item in the list is a cluster
    and a (Ni, D) sized numpy array.
    """
    oldCluster1, oldCluster2 = None, None
    dataList = [[x] for x in data.tolist()]
    clusterNumber = len(dataList)
    while clusterNumber > stop_length:
        distance = None
        for i in range(clusterNumber):
            for j in range(i + 1, clusterNumber):
                newDistance = criterion(dataList[i], dataList[j])
                if distance is None or newDistance < distance:
                    distance = newDistance
                    oldCluster1 = dataList[i]
                    oldCluster2 = dataList[j]

        newCluster = oldCluster1 + oldCluster2
        dataList.remove(oldCluster1)
        dataList.remove(oldCluster2)
        dataList.append(newCluster)
        clusterNumber = len(dataList)
    return [np.asarray(x) for x in dataList]


"""
color = ['red', 'blue', 'green', 'orange']
i = -1
clusters = hac(dataset4, centroid_linkage, 4)
for c in clusters:
    i += 1
    x = []
    y = []
    for index in range(len(c)):
        x.append(c[index][0])
        y.append(c[index][1])
    plt.scatter(x, y, color=color[i])
plt.show()"""
