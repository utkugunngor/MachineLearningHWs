from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix

train_set = np.load('svm/task4/train_set.npy')
train_lbs = np.load('svm/task4/train_labels.npy')
test_set = np.load('svm/task4/test_set.npy')
test_lbs = np.load('svm/task4/test_labels.npy')

kernel = 'rbf'
c_value = 1


def oversample(train_s, train_l):
    oversampled_set = train_s
    oversampled_labels = train_l
    minor, major = 0, 0

    for i in range(len(train_l)):
        if train_l[i] == 0:
            minor += 1
        else:
            major += 1
    imbalance_ratio = major // minor

    for i in range(imbalance_ratio - 1):
        oversampled_set = np.append(oversampled_set, train_s[train_l == 0], axis=0)
        oversampled_labels = np.append(oversampled_labels, train_l[train_l == 0], axis=0)

    # print(len(oversampled_labels))
    # print(np.count_nonzero(oversampled_labels))
    return oversampled_set, oversampled_labels


def undersample(train_s, train_l):
    minor = 0
    for i in range(len(train_l)):
        if train_l[i] == 0:
            minor += 1

    undersampled_set_1 = train_s[train_l == 0]
    undersampled_set_2 = train_s[train_l == 1][:minor]
    undersampled_set = np.append(undersampled_set_1, undersampled_set_2, axis=0)

    undersampled_labels_1 = train_l[train_l == 0]
    undersampled_labels_2 = train_l[train_l == 1][:minor]
    undersampled_labels = np.append(undersampled_labels_1, undersampled_labels_2, axis=0)

    # print(len(undersampled_labels))
    # print(np.count_nonzero(undersampled_labels))
    return undersampled_set, undersampled_labels


def svm_4(train_s, train_l, test_s, test_l, balanced):
    n, x, y = train_s.shape
    train_data = train_s.reshape((n, x * y))
    n2, x2, y2 = test_s.shape
    test_data = test_s.reshape((n2, x2 * y2))
    if balanced:
        classifier = SVC(C=c_value, kernel=kernel, class_weight='balanced')
    else:
        classifier = SVC(C=c_value, kernel=kernel)
    classifier.fit(train_data, train_l)
    accuracy = classifier.score(test_data, test_l)
    print("Accuracy:", accuracy)
    prediction_list = classifier.predict(test_data)
    matrix = confusion_matrix(test_l, prediction_list)
    print("Confusion matrix:", matrix)


if __name__ == "__main__":
    print("Initial train data without balanced")
    svm_4(train_set, train_lbs, test_set, test_lbs, False)
    print("---------------------")
    print("Oversampled train data without balanced")
    oversampled_s, oversampled_l = oversample(train_set, train_lbs)
    svm_4(oversampled_s, oversampled_l, test_set, test_lbs, False)
    print("---------------------")
    print("Undersampled train data without balanced")
    undersampled_s, undersampled_l = undersample(train_set, train_lbs)
    svm_4(undersampled_s, undersampled_l, test_set, test_lbs, False)
    print("---------------------")
    print("Balanced initial data")
    svm_4(train_set, train_lbs, test_set, test_lbs, True)
    print("---------------------")
