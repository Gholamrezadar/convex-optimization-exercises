import scipy.io as sio
from numpy.random import permutation
import numpy as np


def to_onehot(label):
    m = len(set(label))
    n = len(label)
    onehot_matrix = np.zeros([n, m])
    for i in range(n):
        onehot_matrix[i, label[i]] = 1
    return onehot_matrix


def load_USPS_data_instace(path, sample_per_class):
    data = sio.loadmat(path)
    img = data['data']
    img_data = []
    label = []
    for i in range(img.shape[-1]):
        temp_sample_list = []
        for j in range(img.shape[1]):
            temp = np.reshape(img[:, j, i], [16, 16])
            temp_sample_list.append(temp)

        idx = permutation(len(temp_sample_list))
        idx = idx[:sample_per_class]
        selected_samples = [temp_sample_list[x] for x in idx]
        selected_labels = [i for _ in idx]
        img_data.extend(selected_samples)
        label.extend(selected_labels)

    img_data = np.array(img_data)
    img_data = img_data.astype('float')
    label = np.array(label[:])
    label = to_onehot(label)
    return img_data, label


img, label = load_USPS_data_instace("USPSdata/usps_all.mat", 10)
img = np.reshape(img, [img.shape[0], -1])