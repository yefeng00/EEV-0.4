import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import numpy as np
import torch
import os
import shutil


def scatter(x, colors, file_name, class_num):
    f = plt.figure(figsize=(226 / 15, 212 / 15))
    ax = plt.subplot(aspect='equal')
    color_pen = ['black', 'r']
    my_legend = ['Delta_C', 'Delta_M']
    label_set = []
    label_set.append(colors[0])
    for i in range(1, len(colors)):
        flag = 1
        for j in range(len(label_set)):
            if label_set[j] == colors[i]:
                flag = 0
                break
        if flag:
            label_set.append(colors[i])
    for i in range(class_num):
        ax.scatter(x[colors == label_set[i], 0], x[colors == label_set[i], 1], lw=0, s=70, c=color_pen[i],
                   label=str(my_legend[i]))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('tight')
    ax.legend(loc='upper right')
    f.savefig(file_name + ".png", bbox_inches='tight')
    print(file_name + ' save finished')


def plot_TSNE(data, label, path, title, class_num):
    colors = label
    all_features_np = data
    tsne_features = TSNE(random_state=20190129).fit_transform(all_features_np)
    scatter(tsne_features, colors, os.path.join(path, title), class_num)


def visualization(length, layers, c, m, path, elements=10):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for t in range(length - 1):
        for i in range(layers):
            data = []
            label = []
            for j in range(c[layers * t + i].shape[0]):
                for k in range(c[layers * t + i].shape[1]):
                    value1, index1 = torch.topk(c[layers * t + i, j, k], elements)
                    value2, index2 = torch.topk(m[layers * t + i, j, k], elements)
                    c_key = F.normalize(torch.cat([value1, c[layers * t + i, j, k, index2]], dim=0),
                                        dim=0).detach().cpu().numpy().tolist()
                    data.append(c_key)
                    label.append(0)
                    m_key = F.normalize(torch.cat([m[layers * t + i, j, k, index1], value2], dim=0),
                                        dim=0).detach().cpu().numpy().tolist()
                    data.append(m_key)
                    label.append(1)
                plot_TSNE(np.array(data), np.array(label), path, 'case_' + str(j) + '_tsne_' + str(i) + '_' + str(t), 2)
