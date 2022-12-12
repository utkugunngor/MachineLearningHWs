from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
import numpy as np

train_set = np.load('svm/task3/train_set.npy')
train_lbs = np.load('svm/task3/train_labels.npy')
test_set = np.load('svm/task3/test_set.npy')
test_lbs = np.load('svm/task3/test_labels.npy')

kernels = ["linear", "rbf", "poly", "sigmoid"]
c_values = [0.1, 1, 10]
gamma_values = [0.001, 0.01, 0.1]


def train_hyperparams(train_s, train_l, kernel, c_value, gamma_value):
    classifier = SVC(C=c_value, kernel=kernel, gamma=gamma_value)
    results = cross_validate(classifier, train_s, train_l, cv=5)
    print("Kernel:", kernel, "C:", c_value, " gamma:", gamma_value, " acc:", np.average(results["test_score"]))


def train_best(train_s, train_l, test_s, test_l, kernel, c_value, gamma_value):
    classifier = SVC(C=c_value, kernel=kernel, gamma=gamma_value).fit(train_s, train_l)
    accuracy = classifier.score(test_s, test_l)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    n, x, y = train_set.shape
    train_data = train_set.reshape((n, x * y))
    n2, x2, y2 = test_set.shape
    test_data = test_set.reshape((n2, x2 * y2))
    for i in range(len(kernels)):
        for j in range(len(c_values)):
            for k in range(len(gamma_values)):
                train_hyperparams(train_data, train_lbs, kernels[i], c_values[j], gamma_values[k])

    # train the whole set with the best hyperparameter configuration
    train_best(train_data, train_lbs, test_data, test_lbs, 'rbf', 10, 0.01)
