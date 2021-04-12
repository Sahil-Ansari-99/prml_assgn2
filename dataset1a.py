import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


train_data = pd.read_csv('data/Dataset 1A/train.csv')
train_data = train_data.to_numpy()

dev_data = pd.read_csv('data/Dataset 1A/dev.csv')
dev_data = dev_data.to_numpy()


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
        for point in val_:
            pred = get_knns(k, point, train_)
            if pred == point[len(point)-1]:
                correct += 1
            total += 1
        acc = float(correct / total)
        accuracies[k] = acc
    return accuracies


accs = get_knn_accuracy(train_data, dev_data, k_vals=[1, 7, 15])
for key in accs:
    print('K:', key, 'Accuracy:', accs.get(key))
