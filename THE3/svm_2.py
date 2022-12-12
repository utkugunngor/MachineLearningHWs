from sklearn.svm import SVC
import numpy as np
from draw import draw_svm

train_set = np.load('svm/task2/train_set.npy')
train_lbs = np.load('svm/task2/train_labels.npy')
c_value = 1
kernels = ["linear", "rbf", "poly", "sigmoid"]


def svm_2(index):
    x_min = np.min(train_set[:, 0])
    x_max = np.max(train_set[:, 0])
    y_min = np.min(train_set[:, 1])
    y_max = np.max(train_set[:, 1])

    classifier = SVC(C=c_value, kernel=kernels[index]).fit(train_set, train_lbs)
    draw_svm(classifier, train_set, train_lbs, x_min, x_max, y_min, y_max, f"plots/svm_q2-{kernels[index]}.png")


if __name__ == "__main__":
    for i in range(4):
        svm_2(i)