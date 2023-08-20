import torch
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


def plot_tsne(filename='src', perplexity=50.0, early_exaggeration=12.0):
    feature_bank = torch.load('saved_features/{}_features.pt'.format(filename))
    true_labels = torch.load('saved_features/{}_labels.pt'.format(filename))
    softmax_scores = torch.load('saved_features/{}_softmax.pt'.format(filename))

    feature_bank = feature_bank.detach().clone().cpu()
    pred_labels = torch.argmax(softmax_scores, dim=1)
    pred_labels = pred_labels.detach().clone().cpu()

    feature_bank_arr = feature_bank.numpy()
    true_labels_arr = true_labels.numpy()
    pred_labels_arr = pred_labels.numpy()

    feature_bank_embed = TSNE(perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate="auto", init='pca', metric='cosine')\
        .fit_transform(feature_bank_arr)

    data_1 = np.concatenate((feature_bank_embed, true_labels_arr.reshape(-1, 1)), axis=1)
    data_2 = np.concatenate((feature_bank_embed, pred_labels_arr.reshape(-1, 1)), axis=1)

    fig_1 = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(2, 1, 1)
    p1 = sns.scatterplot(
        data=data_1, x=data_1[:, 0], y=data_1[:, 1], hue=data_1[:, 2], legend='full', palette=["b", "r"], s=70
    )
    p1.legend_.remove()
    plt.savefig('plots/{}_true_labels.png'.format(filename))

    fig_2 = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(2, 1, 2)
    p2 = sns.scatterplot(
        data=data_2, x=data_2[:, 0], y=data_2[:, 1], hue=data_2[:, 2], legend='full', palette=["b", "r"], s=70
    )
    p2.legend_.remove()
    plt.savefig('plots/{}_predicted_labels.png'.format(filename))
