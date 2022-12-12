import copy
import math

import numpy as np


def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """
    voc = set()
    for i in data:
        for j in i:
            voc.add(j)
    return voc


def estimate_pi(train_labels):
    """
    Estimates the probability of every class label that occurs in train_labels.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :return: pi. pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    pi = {}
    size = len(train_labels)
    for i in range(size):
        if train_labels[i] in pi:
            pi[train_labels[i]] += 1
        else:
            pi[train_labels[i]] = 1
    for key, value in pi.items():
        pi[key] = value / size
    return pi


def estimate_theta(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all the words in vocab and the values are their estimated probabilities given
             the first level class name.
    """
    theta = {}
    labels = list(set(train_labels))
    length = len(vocab)
    counts = {}
    for i in vocab:
        if i in counts:
            counts[i] += 1
        else:
            counts[i] = 1

    for label in labels:
        temp = counts.copy()
        tmp_length = length
        for i in range(len(train_data)):
            if train_labels[i] == label:
                tmp_length += len(train_data[i])
                for data in train_data[i]:
                    temp[data] += 1

        for key, value in temp.items():
            temp[key] /= tmp_length

        theta[label] = copy.deepcopy(temp)

    return theta


def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """

    scores = []
    for data in test_data:
        tmp = []
        for t in theta:
            t_value = 0
            for word in data:
                if word not in vocab:
                    continue
                t_value += math.log(theta[t][word])
            tmp.append((math.log(pi[t]) + t_value, t))
        scores.append(tmp)
    return scores


def compute_accuracy(results, labels):
    count = 0
    for i, res in enumerate(results):
        data = sorted(res, reverse=True)
        if data[0][1] == labels[i]:
            count += 1

    return float(count) / len(labels)


# I did not handle the spacial characters. This function reads line by line, and vocabulary is created in
# vocabulary() function.
def read_file(file_name):
    with open(file_name, encoding='utf8') as f:
        return map(lambda x: x.replace('\n', ''), f.readlines())


if __name__ == '__main__':

    train_data = list(read_file('nb_data/train_set.txt'))
    train_labels = list(read_file('nb_data/train_labels.txt'))
    test_data = list(read_file('nb_data/test_set.txt'))
    test_labels = list(read_file('nb_data/test_labels.txt'))

    print("Vocabulary creation...")
    vocab = vocabulary(train_data)

    print("Pi and theta estimation...")
    pi = estimate_pi(train_labels)
    theta = estimate_theta(train_data, train_labels, vocab)
    print("Getting test results and computing accuracy...")
    test_results = test(theta, pi, vocab, test_data)

    acc = compute_accuracy(test_results, test_labels)
    print(f"Accuracy: %.3f" % acc)

