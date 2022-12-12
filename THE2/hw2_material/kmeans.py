import math
import numpy as np
import matplotlib.pyplot as plt

dataset1 = np.load('kmeans/dataset1.npy')
dataset2 = np.load('kmeans/dataset2.npy')
dataset3 = np.load('kmeans/dataset3.npy')
dataset4 = np.load('kmeans/dataset4.npy')


def calculate_distance(p1, p2, d):
    total = 0
    for i in range(d):
        total += pow(p1[i] - p2[i], 2)
    distance = math.sqrt(total)
    return distance


def initialize_clusters(data, k):
    initialClusters = []
    x_min, x_max, y_min, y_max = data[0][0], data[0][0], data[0][1], data[0][1]

    for i in range(data.shape[0]):
        x, y = data[i]
        if x < x_min:
            x_min = x
        elif x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        elif y > y_max:
            y_max = y

    for i in range(k):
        x_initial = np.random.uniform(x_min, x_max)
        y_initial = np.random.uniform(y_min, y_max)
        initialClusters.append((x_initial, y_initial))

    return np.asarray(initialClusters)


def assign_clusters(data, cluster_centers):
    """
    Assigns every data point to its closest (in terms of Euclidean distance) cluster center.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: An (N, ) shaped numpy array. At its index i, the index of the closest center
    resides to the ith data point.
    """
    clusterIndex = None
    indexes = []
    for i in range(data.shape[0]):
        distance = None
        x, y = data[i]
        for j in range(cluster_centers.shape[0]):
            new_distance = math.sqrt(pow(x - cluster_centers[j][0], 2) + pow(y - cluster_centers[j][1], 2))
            if distance is None or new_distance < distance:
                distance = new_distance
                clusterIndex = j
        indexes.append(clusterIndex)

    return np.asarray(indexes)


def calculate_cluster_centers(data, assignments, cluster_centers, k):
    """
    Calculates cluster_centers such that their squared Euclidean distance to the data assigned to
    them will be lowest.
    If none of the data points belongs to some cluster center, then assign it to its previous value.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param assignments: An (N, ) shaped numpy array with integers inside. They represent the cluster index
    every data assigned to.
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :param k: Number of clusters
    :return: A (K, D) shaped numpy array that contains the newly calculated cluster centers.
    """
    members = np.zeros(k)
    newCenters = np.zeros((k, cluster_centers.ndim))
    for i in range(data.shape[0]):
        for j in range(data.ndim):
            newCenters[assignments[i]][j] += data[i][j]
        members[assignments[i]] += 1
    for m in range(k):
        for n in range(data.ndim):
            if members[m] != 0:
                newCenters[m][n] /= members[m]
            else:
                newCenters[m][n] = cluster_centers[m][n]
    return np.asarray(newCenters)


def kmeans(data, initial_cluster_centers):
    """
    Applies k-means algorithm.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param initial_cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: cluster_centers, objective_function
    cluster_center.shape is (K, D).
    objective function is a float. It is calculated by summing the squared euclidean distance between
    data points and their cluster centers.
    """
    objective = 0
    k = initial_cluster_centers.shape[0]
    assignments = assign_clusters(data, initial_cluster_centers)
    while True:
        for i in range(len(assignments)):
            objective += pow(calculate_distance(data[i], initial_cluster_centers[assignments[i]], data.shape[1]), 2)
        newCenters = calculate_cluster_centers(data, assignments, initial_cluster_centers, k)
        size = len(newCenters)
        flag = 0
        for j in range(k):
            if (initial_cluster_centers[j] == newCenters[j]).all():
                flag += 1
        if flag == size:
            break
        initial_cluster_centers = newCenters
        assignments = assign_clusters(data, initial_cluster_centers)
        objective = 0

    return newCenters, objective/2

"""
kValues = np.arange(1, 11)
objectives = []
for i in range(1, 11):
    minObj = None
    finalClusters = []
    for j in range(1, 11):
        initialClusters = initialize_clusters(dataset1, i)
        result = kmeans(dataset1, initialClusters)
        if minObj is None:
            minObj = result[1]
            finalClusters = initialClusters
        elif result[1] < minObj:
            minObj = result[1]
            finalClusters = initialClusters
    result = kmeans(dataset1, finalClusters)
    objectives.append(result[1])

color = ['red', 'blue', 'green', 'orange']
i = -1
inClusters = initialize_clusters(dataset3, 4)
result = kmeans(dataset3, inClusters)
clusters = assign_clusters(dataset3, result[0]).tolist()

for c in range(4):
    i += 1
    x = []
    y = []
    for j in range(len(clusters)):
        if clusters[j] == c:
            x.append(dataset3[j][0])
            y.append(dataset3[j][1])
    plt.scatter(x, y, color=color[i])
plt.scatter(result[0][0][0], result[0][0][1], color='black', marker='D')
plt.scatter(result[0][1][0], result[0][1][1], color='black', marker='D')
plt.scatter(result[0][2][0], result[0][2][1], color='black', marker='D')
plt.scatter(result[0][3][0], result[0][3][1], color='black', marker='D')
plt.show()
"""
