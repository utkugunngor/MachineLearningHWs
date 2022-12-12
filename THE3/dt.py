import math
import numpy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def entropy(bucket):
    """
    Calculates the entropy.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated entropy.
    """
    result_entropy = 0
    num_classes = len(bucket)
    total_examples = 0
    for i in bucket:
        total_examples += i
    if total_examples == 0:
        return 0

    for j in range(num_classes):
        p = bucket[j] / total_examples
        if p == 0:
            current_entropy = 0
        else:
            current_entropy = -p * math.log2(p)
        result_entropy += current_entropy

    return result_entropy


def info_gain(parent_bucket, left_bucket, right_bucket):
    """
    Calculates the information gain. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param parent_bucket: Bucket belonging to the parent node. It contains the
    number of examples that belong to each class before the split.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated information gain.
    """
    parent_entropy = entropy(parent_bucket)
    left_entropy = entropy(left_bucket)
    right_entropy = entropy(right_bucket)
    total_examples = 0
    for i in parent_bucket:
        total_examples += i
    if total_examples == 0:
        return 0
    left_weight = sum(left_bucket) / total_examples
    right_weight = sum(right_bucket) / total_examples
    i_gain = parent_entropy - left_weight * left_entropy - right_weight * right_entropy
    return i_gain


def gini(bucket):
    """
    Calculates the gini index.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated gini index.
    """
    total_p = 0
    total_examples = 0
    for i in bucket:
        total_examples += i
    if total_examples == 0:
        return 0
    num_classes = len(bucket)
    for i in range(num_classes):
        p = bucket[i] / total_examples
        total_p += math.pow(p, 2)
    return 1 - total_p


def avg_gini_index(left_bucket, right_bucket):
    """
    Calculates the average gini index. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated average gini index.
    """
    left_gini = gini(left_bucket)
    right_gini = gini(right_bucket)
    total_examples = sum(left_bucket) + sum(right_bucket)
    left_examples = sum(left_bucket)
    right_examples = sum(right_bucket)
    left_weight = left_examples / total_examples
    right_weight = right_examples / total_examples
    return left_weight * left_gini + right_weight * right_gini


def calculate_split_values(data, labels, num_classes, attr_index, heuristic_name):
    """
    For every possible values to split the data for the attribute indexed by
    attribute_index, it divides the data into buckets and calculates the values
    returned by the heuristic function named heuristic_name. The split values
    should be the average of the closest 2 values. For example, if the data has
    2.1 and 2.2 in it consecutively for the values of attribute index by attr_index,
    then one of the split values should be 2.15.
    :param data: An (N, M) shaped numpy array. N is the number of examples in the
    current node. M is the dimensionality of the data. It contains the values for
    every attribute for every example.
    :param labels: An (N, ) shaped numpy array. It contains the class values in
    it. For every value, 0 <= value < num_classes.
    :param num_classes: An integer. The number of classes in the dataset.
    :param attr_index: An integer. The index of the attribute that is going to
    be used for the splitting operation. This integer indexs the second dimension
    of the data numpy array.
    :param heuristic_name: The name of the heuristic function. It should either be
    'info_gain' of 'avg_gini_index' for this homework.
    :return: An (L, 2) shaped numpy array. L is the number of split values. The
    first column is the split values and the second column contains the calculated
    heuristic values for their splits.
    """
    attributes = []
    heuristics = []
    for i in range(data.shape[0]):
        attributes.append(data[i][attr_index])
    sorted_attributes = sorted(attributes)
    split_values = []
    for i in range(len(sorted_attributes) - 1):
        split_values.append((sorted_attributes[i] + sorted_attributes[i + 1]) / 2)

    for i in range(len(sorted_attributes) - 1):
        left_bucket = []
        right_bucket = []
        for j in range(data.shape[0]):
            if data[j][attr_index] < split_values[i]:
                left_bucket.append(j)
            elif data[j][attr_index] >= split_values[i]:
                right_bucket.append(j)

        left_labels = [0] * num_classes
        right_labels = [0] * num_classes
        parent_labels = [0] * num_classes

        for index in left_bucket:
            left_labels[labels[index]] += 1
        for index in right_bucket:
            right_labels[labels[index]] += 1
        for k in range(num_classes):
            parent_labels[k] = left_labels[k] + right_labels[k]

        if heuristic_name == "info_gain":
            heuristic = info_gain(parent_labels, left_labels, right_labels)
        else:
            heuristic = avg_gini_index(left_labels, right_labels)
        heuristics.append(heuristic)

    return np.asarray(list(zip(split_values, heuristics)))


def chi_squared_test(left_bucket, right_bucket):
    """
    Calculates chi squared value and degree of freedom between the selected attribute
    and the class attribute. A bucket is a list of size num_classes. bucket[i] is the
    number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float and and integer. Chi squared value and degree of freedom.
    """
    total_examples = sum(left_bucket) + sum(right_bucket)
    num_classes = len(left_bucket)
    chi_square = 0

    crosstab = [left_bucket[:], right_bucket[:], [0] * (num_classes + 1)]

    crosstab[0] = numpy.append(crosstab[0], [sum(left_bucket)])
    crosstab[1] = numpy.append(crosstab[1], [sum(right_bucket)])
    crosstab[2] = numpy.append(crosstab[2], [0])

    """crosstab[0].append(sum(left_bucket))
    crosstab[1].append(sum(right_bucket))
    crosstab[2].append(0)"""

    for i in range(2):
        for j in range(num_classes + 1):
            crosstab[2][j] += crosstab[i][j]

    for i in range(2):
        for j in range(num_classes):
            observed = crosstab[i][j]
            calculated = (crosstab[2][j] * crosstab[i][num_classes]) / total_examples
            if calculated != 0:
                chi_square += math.pow(observed - calculated, 2) / calculated

    table = np.zeros((len(crosstab[0]), len(crosstab[0])))
    rows = 0
    for i in range(2):
        table[i][-1] = crosstab[i][-1]
        if table[i][-1] != 0:
            rows += 1

    columns = 0
    for i in range(len(crosstab[0]) - 1):
        table[-1][i] = crosstab[2][i]
        if table[-1][i] != 0:
            columns += 1

    return chi_square, (rows-1)*(columns-1)


if __name__ == "__main__":

    train_data = np.load('dt/train_set.npy')
    train_labels = np.load('dt/train_labels.npy')
    test_data = np.load('dt/test_set.npy')
    test_labels = np.load('dt/test_labels.npy')

    clf = DecisionTreeClassifier(random_state=1234)
    model = clf.fit(train_data, train_labels)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    tree.plot_tree(clf)
    fig.savefig('decision_tree.png')