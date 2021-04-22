import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import multivariate_normal


train_data = pd.read_csv('data/Dataset 1B/train.csv')
train_data = train_data.to_numpy()

val_data = pd.read_csv('data/Dataset 1B/dev.csv')
val_data = val_data.to_numpy()


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


def get_kmeans(data, k, num_iters=10, diagonal=False):
    d = data.shape[1]
    init_means = np.random.randint(0, data.shape[0], size=k)
    kmeans = np.zeros((k, d))
    for i in range(k):
        kmeans[i] = data[init_means[i]]
    point_means = np.zeros(data.shape)
    tol = 0.001
    for i in range(num_iters):
        for j in range(data.shape[0]):
            point = data[j]
            min_dist = 999
            curr_mean = kmeans[0]
            for k in range(kmeans.shape[0]):
                curr_dist = np.linalg.norm(point - kmeans[k])
                if curr_dist < min_dist:
                    curr_mean = kmeans[k]
                    min_dist = curr_dist
            point_means[j] = curr_mean
        err = 0
        for j in range(kmeans.shape[0]):
            curr_sum = np.zeros(data.shape[1])
            curr_num = 0
            for k in range(data.shape[0]):
                if (point_means[k] == kmeans[j]).all():
                    curr_sum += data[k]
                    curr_num += 1
            new_mean = curr_sum / curr_num
            err += np.linalg.norm(kmeans[j] - new_mean)
            kmeans[j] = new_mean
        if err < tol:
            break

    for j in range(data.shape[0]):
        point = data[j]
        min_dist = 999
        curr_mean = kmeans[0]
        for k in range(kmeans.shape[0]):
            curr_dist = np.linalg.norm(point - kmeans[k])
            if curr_dist < min_dist:
                curr_mean = kmeans[k]
                min_dist = curr_dist
        point_means[j] = curr_mean

    covmatrices = []
    wqs = []
    for j in range(kmeans.shape[0]):
        data_points = []
        curr_count = 0
        for k in range(data.shape[0]):
            if (point_means[k] == kmeans[j]).all():
                data_points.append(data[k])
                curr_count += 1
        data_points = np.array(data_points)
        data_points = data_points - kmeans[j]
        cov_matrix = np.matmul(data_points.T, data_points)
        if diagonal:
            for x in range(cov_matrix.shape[0]):
                for y in range(cov_matrix.shape[1]):
                    if x != y:
                        cov_matrix[x][y] = 0.0
        covmatrices.append(cov_matrix)
        wqs.append(curr_count / data.shape[0])
    covmatrices = np.array(covmatrices)
    wqs = np.array(wqs)
    return kmeans, covmatrices, wqs


def calculate_responsibilty_terms(data, kmeans, covmatrices, wqs):
    Q = kmeans.shape[0]
    gammas = np.zeros((data.shape[0], Q))
    d = len(data[0]) - 1
    for i in range(data.shape[0]):
        den = 0
        for j in range(Q):
            mean_subtracted = data[i] - kmeans[j]
            cov_inv = np.linalg.inv(covmatrices[j])
            cov_det = np.linalg.det(covmatrices[j])
            prod = np.dot(mean_subtracted, np.dot(cov_inv, mean_subtracted.T))
            prod = -0.5 * prod
            p = np.exp(prod)
            p /= (2 * np.pi) ** (d / 2)
            p /= np.sqrt(cov_det)
            p = wqs[j] * p
            den += p
            gammas[i][j] = p
        gammas[i] /= den
    gammas = np.array(gammas)
    return gammas


def maximization_step(data, gammas, diagonal=False):
    d = data.shape[1]
    q = gammas.shape[1]
    nq = np.sum(gammas, axis=0)
    wqs = nq / data.shape[0]
    mu_q = np.zeros((q, d))
    for i in range(q):
        curr = np.zeros(d)
        for j in range(data.shape[0]):
            curr += gammas[j][i] * data[j]
        mu_q[i] = curr / nq[i]
    c_q = []
    for i in range(q):
        curr = np.zeros((d, d))
        for j in range(data.shape[0]):
            mean_subtracted = data[j] - mu_q[i]
            mean_subtracted = np.array([mean_subtracted])
            curr += gammas[j][i] * np.multiply(mean_subtracted.T, mean_subtracted)
        curr /= nq[i]
        if diagonal:
            for x in range(curr.shape[0]):
                for y in range(curr.shape[1]):
                    if x != y:
                        curr[x][y] = 0.0
        c_q.append(curr)
    c_q = np.array(c_q)
    return mu_q, c_q, wqs


def calculate_log_likelihood(data, mus, cqs, wqs):
    likelihood = 0
    q = wqs.shape[0]
    d = data.shape[1]
    for i in range(data.shape[0]):
        curr = 0
        for j in range(q):
            mean_subtracted = data[i] - mus[j]
            cov_inv = np.linalg.inv(cqs[j])
            cov_det = np.linalg.det(cqs[j])
            prod = np.dot(mean_subtracted, np.dot(cov_inv, mean_subtracted.T))
            prod = -0.5 * prod
            p = np.exp(prod)
            p /= (2 * np.pi) ** (d / 2)
            p /= np.sqrt(cov_det)
            curr += wqs[j] * p
        likelihood += np.log(curr)
    return likelihood


def gmm(data, q=4, diagonal=False):
    print('Making model...')
    class_wise_data = get_class_wise_data(data)
    tol = 0.001
    res = {}
    for class_ in class_wise_data:
        k_means, cov_matrices, w_qs = get_kmeans(class_wise_data.get(class_), q, diagonal=diagonal)
        curr_likelihood = calculate_log_likelihood(class_wise_data.get(class_), k_means, cov_matrices, w_qs)
        err = 999
        while err > tol:
            gammas_ = calculate_responsibilty_terms(class_wise_data.get(class_), k_means, cov_matrices, w_qs)
            k_means, cov_matrices, w_qs = maximization_step(class_wise_data.get(class_), gammas_, diagonal=diagonal)
            new_likelihood = calculate_log_likelihood(class_wise_data.get(class_), k_means, cov_matrices, w_qs)
            err = abs(new_likelihood - curr_likelihood)
            curr_likelihood = new_likelihood
        res[class_] = {
            'mu_q': k_means,
            'c_q': cov_matrices,
            'w_q': w_qs
        }
    return res


def calculate_gaussian_probabilty(x, mu, cov_matrix):
    d = len(x)
    mean_subtracted = x - mu
    cov_inv = np.linalg.inv(cov_matrix)
    cov_det = np.linalg.det(cov_matrix)
    prod = np.dot(mean_subtracted, np.dot(cov_inv, mean_subtracted.T))
    prod = -0.5 * prod
    p = np.exp(prod)
    p /= (2 * np.pi) ** (d / 2)
    p /= np.sqrt(cov_det)
    return p


def get_probabilty(x, mus, cqs, wqs):
    prob = 0
    q = len(mus)
    for i in range(q):
        p = calculate_gaussian_probabilty(x, mus[i], cqs[i])
        prob += wqs[i] * p
    return prob


def predict(x, model):
    pred = 0
    max_prob = 0
    for class_ in model:
        class_params = model.get(class_)
        muq = class_params.get('mu_q')
        cq = class_params.get('c_q')
        wqs = class_params.get('w_q')
        curr_prob = get_probabilty(x, muq, cq, wqs)
        if curr_prob > max_prob:
            pred = class_
            max_prob = curr_prob
    return pred


def get_accuracy(data, model):
    print('Getting accuracy...')
    correct = 0
    total = data.shape[0]
    d = data.shape[1] - 1
    y_true = []
    y_pred = []
    for point in data:
        correct_class = point[d]
        pred_class = predict(point[:d], model)
        y_pred.append(pred_class)
        y_true.append(correct_class)
        if pred_class == correct_class:
            correct += 1
    acc = correct / total
    return acc, y_true, y_pred


def knn_classifier(x, data, k):
    pred = 0
    min_radius = 999
    for class_ in data:
        class_data = data.get(class_)
        distances = []
        for point in class_data:
            if (point == x).all():
                continue
            dist = np.linalg.norm(x - point)
            distances.append(dist)
        distances.sort()
        r_i = distances[k-1]
        if r_i < min_radius:
            pred = class_
            min_radius = r_i
    return pred


def get_knn_accuracy(train_, val_, k):
    class_wise_acc = {}
    y_pred = []
    y_true = []
    for class_ in val_:
        class_points = val_.get(class_)
        correct = 0
        total = 0
        for point in class_points:
            pred = knn_classifier(point, train_, k)
            y_pred.append(pred)
            y_true.append(class_)
            total += 1
            if pred == class_:
                correct += 1
        acc = correct / total
        class_wise_acc[class_] = {
            'correct': correct,
            'total': total,
            'accuracy': acc
        }
    return class_wise_acc, y_true, y_pred


# GMM model
# Qs = [4, 5, 6, 7]
# for x in Qs:
#     model = gmm(train_data, q=x, diagonal=False)  # diagonal = True for diagonal covariance matrix
#     train_acc, train_true, train_pred = get_accuracy(train_data, model)
#     val_acc, val_true, val_pred = get_accuracy(val_data, model)
#     print('Q:', x, 'Train Accuracy:', train_acc)
#     print('Q:', x, 'Val Accuracy:', val_acc)
#     print('Train confusion matrix:')
#     print(confusion_matrix(train_true, train_pred))
#     print('Validation Confusion Matrix:')
#     print(confusion_matrix(val_true, val_pred))

# KNN model
# train_class_wise = get_class_wise_data(train_data)
# val_class_wise = get_class_wise_data(val_data)
# Ks = [10, 20]
# for k in Ks:
#     train_res, train_true, train_pred = get_knn_accuracy(train_class_wise, train_class_wise, k)
#     val_res, val_true, val_pred = get_knn_accuracy(train_class_wise, val_class_wise, k)
#     print('K:', k)
#     total = 0.0
#     val_total = 0.0
#     for x in train_res:
#         print('Class:', x, 'Train Accuracy:', train_res.get(x).get('accuracy'))
#         print('Class:', x, 'Val Accuracy:', val_res.get(x).get('accuracy'))
#         total += train_res.get(x).get('accuracy')
#         val_total += val_res.get(x).get('accuracy')
#     total /= len(train_res)
#     val_total /= len(val_res)
#     print('Train Accuracy:', total)
#     print('Val Accuracy:', val_total)
#     print('Train Confusion Matrix:')
#     print(confusion_matrix(train_true, train_pred))
#     print('Val Confusion Matrix:')
#     print(confusion_matrix(val_true, val_pred))


fig, ax = plt.subplots()
colors = ['r', 'g', 'b', 'y']
# legend_arr = []
# i = 0
min_x = 999
max_x = -999
min_y = 999
max_y = -999
class_wise_data = get_class_wise_data(train_data)
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
# # plt.legend(legend_arr)
# # plt.show()
#
# x_list = np.linspace(min_x, max_x, 100)
# y_list = np.linspace(min_y, max_y, 100)
#
# z = np.zeros((len(x_list), len(x_list)))
# X = []
# Y = []
#
# for i in range(len(x_list)):
#     temp_x = []
#     temp_y = []
#     for j in range(len(y_list)):
#         temp_x.append(x_list[i])
#         temp_y.append(y_list[j])
#         point = np.array([x_list[i], y_list[j]])
#         z[i][j] = knn_classifier(point, class_wise_data, 10)
#     X.append(temp_x)
#     Y.append(temp_y)
#
# ax.contourf(X, Y, z)
#
# for class_ in class_wise_data:
#     x = []
#     y = []
#     legend_arr.append(class_)
#     class_data = class_wise_data.get(class_)
#     for point in class_data:
#         x.append(point[0])
#         y.append(point[1])
#     ax.scatter(x, y, color='k')
#
# plt.show()
model = gmm(train_data, q=7, diagonal=True)  # diagonal = True for diagonal covariance matrix

x_list = np.linspace(min_x, max_x, 100)
y_list = np.linspace(min_y, max_y, 100)

k = 0
for class_ in model:
    z = np.zeros((len(x_list), len(x_list)))
    X = []
    Y = []
    for i in range(len(x_list)):
        temp_x = []
        temp_y = []
        for j in range(len(y_list)):
            temp_x.append(x_list[i])
            temp_y.append(y_list[j])
            point = np.array([x_list[i], y_list[j]])
            z[i][j] = get_probabilty(point, model.get(class_).get('mu_q'), model.get(class_).get('c_q'), model.get(class_).get('w_q'))
        X.append(temp_x)
        Y.append(temp_y)

    ax.contour(X, Y, z, 38, colors=colors[k])
    k += 1

for class_ in class_wise_data:
    x = []
    y = []
    class_data = class_wise_data.get(class_)
    for point in class_data:
        x.append(point[0])
        y.append(point[1])
    ax.scatter(x, y, color='k')

plt.show()
