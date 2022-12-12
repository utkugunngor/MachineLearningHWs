from sklearn.svm import SVC
import numpy as np
from draw import draw_svm

train_set = np.load('svm/task1/train_set.npy')
train_lbs = np.load('svm/task1/train_labels.npy')
c_values = [0.01, 0.1, 1, 10, 100]


def svm_1(index):
    x_min = np.min(train_set[:, 0])
    x_max = np.max(train_set[:, 0])
    y_min = np.min(train_set[:, 1])
    y_max = np.max(train_set[:, 1])

    classifier = SVC(C=c_values[index], kernel='linear').fit(train_set, train_lbs)
    draw_svm(classifier, train_set, train_lbs, x_min, x_max, y_min, y_max, f"plots/svm_q1-{index+1}.png")


if __name__ == "__main__":
    for i in range(len(c_values)):
        svm_1(i)