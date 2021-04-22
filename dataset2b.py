import numpy as np
import pandas as pd
import os
import GMM
import pickle


ROOT_DIR = 'data/Dataset 2B'


def get_class_wise_data():
    train_ = {}
    val_ = {}
    classes = os.listdir(ROOT_DIR)
    for class_ in classes:
        files = os.listdir(ROOT_DIR + '/' + class_)
        for file_dir in files:
            file_data = []
            file_list = os.listdir(ROOT_DIR + '/' + class_ + '/' + file_dir)
            dir_ = ROOT_DIR + '/' + class_ + '/' + file_dir
            for file in file_list:
                img_data = np.loadtxt(dir_ + '/' + file)
                for row in img_data:
                    file_data.append(row)
            file_data = np.array(file_data)
            if file_dir == 'train':
                train_[class_] = file_data
            else:
                val_[class_] = file_data
    return train_, val_


train_data, val_data = get_class_wise_data()
model = GMM.gmm(train_data, q=20, diagonal=False)
try:
    model_file = open('dataset2b_model', 'wb')
    pickle.dump(model, model_file)
    model_file.close()
except Exception as e:
    print('Some error', e)
# train_acc = GMM.get_accuracy(train_data, model)
# val_acc = GMM.get_accuracy(val_data, model)
# print('Train Accuracy:', train_acc)
# print('Val Accuracy:', val_acc)
