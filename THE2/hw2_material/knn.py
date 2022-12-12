import numpy as np
import math
import matplotlib.pyplot as plt

train_set1 = np.load('knn/train_set.npy')
train_labels1 = np.load('knn/train_labels.npy')
test_set1 = np.load('knn/test_set.npy')
test_labels1 = np.load('knn/test_labels.npy')


def calculate_distances(train_data, test_instance, distance_metric):
    """
    Calculates Manhattan (L1) / Euclidean (L2) distances between test_instance and every train instance.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data.
    :param test_instance: A (D, ) shaped numpy array.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: An (N, ) shaped numpy array that contains distances.
    """

    dim = train_data.shape[1]
    distances = []

    for i in train_data:
        total = 0
        for j in range(dim):
            if distance_metric == 'L1':
                total += abs(i[j] - test_instance[j])
            else:
                total += pow(i[j] - test_instance[j], 2)
        if distance_metric == 'L1':
            distances.append(total)
        else:
            distances.append(math.sqrt(total))
    return np.asarray(distances)


def find_majority(labels, k):
    maxCount = 0
    index = -1
    for i in range(k):
        count = 0
        for j in range(k):
            if labels[i] == labels[j]:
                count += 1
        if count > maxCount:
            maxCount = count
            index = i
    if maxCount > k // 2:
        return labels[index]
    else:
        return min(labels)


def majority_voting(distances, labels, k):
    """
    Applies majority voting. If there are more then one major class, returns the smallest label.
    :param distances: An (N, ) shaped numpy array that contains distances
    :param labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: An integer. The label of the majority class.
    """

    labelCounts = dict()
    for i in set(labels):
        labelCounts[i] = 0
    voting = []
    for i in range(k):
        voting.append((labels[i], distances[i]))
    voting.sort(key=lambda x: x[1])

    for j in range(k, len(distances)):
        if distances[j] < voting[-1][1]:
            voting[-1] = (labels[j], distances[j])
            voting.sort(key=lambda x: x[1])

        elif distances[j] == voting[-1][1]:
            if labels[j] < voting[-1][0]:
                voting[-1] = (labels[j], distances[j])
                voting.sort(key=lambda x: x[1])

    labelArr = []
    for i in range(k):
        labelArr.append(voting[i][0])
    for l in labelArr:
        labelCounts[l] += 1
    sortedLabels = [(key, value) for key, value in labelCounts.items()]
    sortedLabels.sort(key=lambda x: x[0])
    sortedLabels.sort(key=lambda x: x[1], reverse=True)
    return sortedLabels[0][0]


def knn(train_data, train_labels, test_data, test_labels, k, distance_metric):
    """
    Calculates accuracy of knn on test data using train_data.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param train_labels: An (N, ) shaped numpy array that contains labels
    :param test_data: An (M, D) shaped numpy array where M is the number of examples
    and D is the dimension of the data
    :param test_labels: An (M, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. The calculated accuracy.
    """

    predictions = []
    count = 0
    for i in range(len(test_data)):
        distances = calculate_distances(train_data, test_data[i], distance_metric)
        majority = majority_voting(distances, train_labels, k)
        predictions.append(majority)
    for j in range(len(test_labels)):
        if predictions[j] == test_labels[j]:
            count += 1
    return float(count) / len(test_labels)


def split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold):
    """
    Splits training dataset into k and returns the validation_indexth one as the
    validation set and others as the training set. You can assume k_fold divides N.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param validation_index: An integer. 0 <= validation_index < k_fold. Specifies which fold
    will be assigned as validation set.
    :param k_fold: The number of groups that the whole_train_data will be divided into.
    :return: train_data, train_labels, validation_data, validation_labels
    train_data.shape is (N-N/k_fold, D).
    train_labels.shape is (N-N/k_fold, ).
    validation_data.shape is (N/k_fold, D).
    validation_labels.shape is (N/k_fold, ).
    """

    folds = np.array_split(whole_train_data, k_fold)
    foldLabels = np.array_split(whole_train_labels, k_fold)
    train_data, train_labels, validation_data, validation_labels = [], [], [], []

    for i in range(k_fold):
        if i != validation_index:
            train_data.append(folds[i])
            train_labels.append(foldLabels[i])
        else:
            validation_data.append(folds[i])
            validation_labels.append(foldLabels[i])
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    validation_data = np.concatenate(validation_data)
    validation_labels = np.concatenate(validation_labels)

    return train_data, train_labels, validation_data, validation_labels


def cross_validation(whole_train_data, whole_train_labels, k_fold, k, distance_metric):
    """
    Applies k_fold cross-validation and averages the calculated accuracies.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param k_fold: An integer.
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. Average accuracy calculated.
    """

    predictions = []
    for i in range(k_fold):
        train_data, train_labels, validation_data, validation_labels = split_train_and_validation(whole_train_data,
                                                                                                  whole_train_labels, i,
                                                                                                  k_fold)
        currentAccuracy = knn(train_data, train_labels, validation_data, validation_labels, k, distance_metric)
        predictions.append(currentAccuracy)
    return sum(predictions) / len(predictions)


"""kValues = np.arange(1, 181)
acc = []
for i in range(1, 181):
    acc.append(cross_validation(train_set1, train_labels1, 10, i, 'L2'))

print(max(acc), acc.index(max(acc)) + 1)
plt.plot(kValues, acc)
plt.show()"""

#print(knn(train_set1, train_labels1, test_set1, test_labels1, 18, 'L1'))
