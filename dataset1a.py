import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


train_data = pd.read_csv('data/Dataset 1A/train.csv')
train_data = train_data.to_numpy()

dev_data = pd.read_csv('data/Dataset 1A/dev.csv')

dev_data = dev_data.to_numpy()


def get_class_wise_data(data):
    obj = {}
    d = len(data[0]) - 1
    for point in data:
        curr_class = point[d]
        if obj.get(curr_class) is None:
            obj[curr_class] = [point[:d]]
        else:
            obj.get(curr_class).append(point[:d])
    for class_ in obj:
        obj[class_] = np.array(obj.get(class_))
    return obj


def get_knns(k, x, data):
    distances = []
    x_coords = x[:len(x)-1]
    for point in data:
        if (point == x).all():  # don't check with same point
            continue
        point_coords = point[:len(point)-1]  # last element is the class
        dist = np.linalg.norm(x_coords - point_coords)  # l2 norm
        distances.append((dist, point_coords, point[len(point)-1]))
    distances.sort(key=lambda a: a[0])
    top_k_points = distances[:k]
    class_counts = {}  # key will be class, and value will be number of points belonging to the class
    for point in top_k_points:
        if class_counts.get(point[2]) is None:
            class_counts[point[2]] = 1
        else:
            class_counts[point[2]] = class_counts.get(point[2]) + 1
    pred = 0
    max_count = 0
    for class_ in class_counts:
        if class_counts.get(class_) > max_count:
            pred = class_
            max_count = class_counts.get(class_)
    return pred


def get_knn_accuracy(train_, val_, k_vals=[1]):
    accuracies = {}
    for k in k_vals:
        print(k)
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        for point in val_:
            pred = get_knns(k, point, train_)
            y_pred.append(pred)
            y_true.append(point[len(point)-1])
            if pred == point[len(point)-1]:
                correct += 1
            total += 1
        acc = float(correct / total)
        accuracies[k] = acc
        print(confusion_matrix(y_true, y_pred))
    return accuracies


def naive_bayes_classifier(data, case=1):
    assert 1 <= case <= 3
    class_wise_data = {}
    d = len(data[0]) - 1
    for point in data:
        class_ = point[len(point)-1]
        if class_wise_data.get(class_) is None:
            class_wise_data[class_] = [point[:d]]
        else:
            class_wise_data.get(class_).append(point[:d])
    num_classes = len(class_wise_data)
    class_wise_mean_var = {}
    for class_ in class_wise_data:
        class_points = np.array(class_wise_data.get(class_), dtype=object)
        curr_mean = class_points.mean(axis=0)
        curr_variance = class_points.var(axis=0)
        class_wise_mean_var[class_] = (curr_mean, curr_variance)
    res = {}
    priors = {}
    if case == 1:  # cov matrix = sigma^2 * I
        sigma = 0
        for class_ in class_wise_mean_var:
            sigma += class_wise_mean_var.get(class_)[1].sum()
        sigma = sigma / (num_classes * d)
        cov_matrix = sigma * np.identity(d)
        for class_ in class_wise_mean_var:
            res[class_] = (class_wise_mean_var.get(class_)[0], cov_matrix)
            priors[class_] = len(class_wise_data.get(class_)) / len(data)
    elif case == 2:  # cov_matrix = C
        cov_matrix = np.zeros((d, d))
        for class_ in class_wise_mean_var:
            class_points = np.array(class_wise_data.get(class_), dtype=object)
            class_mean = class_wise_mean_var.get(class_)[0]
            mean_subtracted = class_points - class_mean
            class_cov_matrix = (np.matmul(mean_subtracted.T, mean_subtracted)) / len(class_points)
            cov_matrix = np.add(cov_matrix, class_cov_matrix)
        cov_matrix = cov_matrix / num_classes
        for class_ in class_wise_mean_var:
            res[class_] = (class_wise_mean_var.get(class_)[0], cov_matrix)
            priors[class_] = len(class_wise_data.get(class_)) / len(data)
    elif case == 3:
        for class_ in class_wise_mean_var:
            class_points = np.array(class_wise_data.get(class_), dtype=object)
            class_mean = class_wise_mean_var.get(class_)[0]
            mean_subtracted = class_points - class_mean
            class_cov_matrix = (np.matmul(mean_subtracted.T, mean_subtracted)) / len(class_points)
            res[class_] = (class_wise_mean_var.get(class_)[0], class_cov_matrix)
            priors[class_] = len(class_wise_data.get(class_)) / len(data)
    return res, priors


def predict_naive_bayes(mean_var, priors, x):
    pred = 0
    max_prob = 0.0
    d = len(x)
    for class_ in mean_var:
        class_mean = mean_var.get(class_)[0]
        class_cov = mean_var.get(class_)[1]
        class_cov = class_cov.astype(np.float64)
        x_mean_sub = x - class_mean
        exp_pow = np.dot(x_mean_sub, np.dot(np.linalg.inv(class_cov), x_mean_sub.T))
        exp_pow = (-1/2) * exp_pow
        cov_det = np.linalg.det(class_cov)
        p = np.exp(exp_pow)
        p /= (2 * np.pi) ** (d / 2)
        p /= np.sqrt(cov_det)
        if p > max_prob:
            max_prob = p
            pred = class_
    return pred


def get_naive_bayes_accuracy(mean_var, priors, data):
    correct = 0
    y_true = []
    y_pred = []
    for point in data:
        pred = predict_naive_bayes(mean_var, priors, point)
        y_pred.append(pred)
        y_true.append(point[len(point)-1])
        if pred == point[len(point)-1]:
            correct += 1
    print(confusion_matrix(y_true, y_pred))
    acc = correct / len(data)
    return acc


# accs = get_knn_accuracy(train_data, train_data, k_vals=[1, 7, 15])
# for key in accs:
#     print('K:', key, 'Accuracy:', accs.get(key))


# cases = [1, 2, 3]
# for case in cases:
#     mean_vars, priors = naive_bayes_classifier(train_data, case=case)
#     print('Case:', case, 'Accuracy:', get_naive_bayes_accuracy(mean_vars, priors, dev_data))

class_wise_data = get_class_wise_data(train_data)
fig, ax = plt.subplots()
colors = ['r', 'g', 'b', 'y']
legend_arr = []
i = 0
min_x = 999
max_x = -999
min_y = 999
max_y = -999
for class_ in class_wise_data:
    class_data = class_wise_data.get(class_)
    for point in class_data:
        if point[0] < min_x:
            min_x = point[0]
        if point[0] > max_x:
            max_x = point[0]
        if point[1] < min_y:
            min_y = point[1]
        if point[1] > max_y:
            max_y = point[1]
    # ax.scatter(x, y, color=colors[i])
    i += 1
# plt.legend(legend_arr)
# plt.show()

# mean_vars, priors = naive_bayes_classifier(train_data, case=2)
x_list = np.linspace(min_x, max_x, 100)
y_list = np.linspace(min_y, max_y, 100)

z = np.zeros((len(x_list), len(x_list)))
X = []
Y = []

for i in range(len(x_list)):
    temp_x = []
    temp_y = []
    for j in range(len(y_list)):
        temp_x.append(x_list[i])
        temp_y.append(y_list[j])
        point = np.array([x_list[i], y_list[j], 0])
        z[i][j] = get_knns(7, point, train_data)
    X.append(temp_x)
    Y.append(temp_y)

cp = ax.contourf(X, Y, z)

for class_ in class_wise_data:
    x = []
    y = []
    legend_arr.append(class_)
    class_data = class_wise_data.get(class_)
    for point in class_data:
        x.append(point[0])
        y.append(point[1])
    ax.scatter(x, y, color='k')
    i += 1

plt.show()
