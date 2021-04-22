import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


ROOT_DIR = 'data/Dataset 2A'


def get_class_wise_data():
    train_ = {}
    val_ = {}
    classes = os.listdir(ROOT_DIR)
    for class_ in classes:
        files = os.listdir(ROOT_DIR + '/' + class_)
        for file in files:
            data = pd.read_csv(ROOT_DIR + '/' + class_ + '/' + file)
            data = data.to_numpy()
            data = data[:, 1:]
            data = data.astype('float64')
            # data = normalize_data(data)
            if file == 'dev.csv':
                val_[class_] = data
            else:
                train_[class_] = data
    return train_, val_


def normalize_data(data):
    i = 0
    for point in data:
        point_norm = np.linalg.norm(point)
        point /= point_norm
        data[i] = point
        i += 1
    return data


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


def calculate_log_likelihood(data, mus, cqs, wqs):
    likelihood = 0
    q = wqs.shape[0]
    d = data.shape[1]
    eps = 0.01
    for i in range(data.shape[0]):
        curr = 0
        for j in range(q):
            mean_subtracted = data[i] - mus[j]
            cov_det = np.linalg.det(cqs[j])
            if cov_det < 0.0001:
                cqs[j] = cqs[j] + 0.0001 * np.identity(d)
                cov_det = np.linalg.det(cqs[j])
            cov_inv = np.linalg.pinv(cqs[j])
            # cov_det = np.linalg.det(cqs[j]) + eps
            prod = np.dot(mean_subtracted, np.dot(cov_inv, mean_subtracted.T))
            prod = -0.5 * prod + eps
            p = np.exp(prod)
            p /= (2 * np.pi) ** (d / 2)
            p /= np.sqrt(cov_det)
            curr += wqs[j] * p
        likelihood += np.log(curr)
    return likelihood


def calculate_responsibilty_terms(data, kmeans, covmatrices, wqs):
    Q = kmeans.shape[0]
    gammas = np.zeros((data.shape[0], Q))
    d = len(data[0])
    eps = 0.01
    for i in range(data.shape[0]):
        den = 0
        for j in range(Q):
            mean_subtracted = data[i] - kmeans[j]
            cov_det = np.linalg.det(covmatrices[j])
            if cov_det < 0.0001:
                covmatrices[j] = covmatrices[j] + 0.0001 * np.identity(d)
                cov_det = np.linalg.det(covmatrices[j])
            cov_inv = np.linalg.pinv(covmatrices[j])
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


def gmm(train_, q=4, diagonal=False):
    print('Making model...')
    tol = 0.1
    res = {}
    for class_ in train_:
        k_means, cov_matrices, w_qs = get_kmeans(train_.get(class_), q, diagonal=diagonal)
        curr_likelihood = calculate_log_likelihood(train_.get(class_), k_means, cov_matrices, w_qs)
        err = 999
        while err > tol:
            print(class_, err)
            gammas_ = calculate_responsibilty_terms(train_.get(class_), k_means, cov_matrices, w_qs)
            k_means, cov_matrices, w_qs = maximization_step(train_.get(class_), gammas_, diagonal=diagonal)
            new_likelihood = calculate_log_likelihood(train_.get(class_), k_means, cov_matrices, w_qs)
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
    cov_inv = np.linalg.pinv(cov_matrix)
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
    pred = 'coast'
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
    correct = 0
    total = 0
    for class_ in data:
        class_data = data.get(class_)
        for point in class_data:
            total += 1
            pred_class = predict(point, model)
            if pred_class == class_:
                correct += 1
    acc = correct / total
    return acc


train_data, val_data = get_class_wise_data()
model = gmm(train_data, q=6, diagonal=True)
train_acc = get_accuracy(train_data, model)
val_acc = get_accuracy(val_data, model)
print('Train Accuracy:', train_acc)
print('Val Accuracy:', val_acc)
# print(predict(train_data.get('coast')[0], model))
