import torch
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


def plot_tsne(filename='tar', perplexity=50.0, early_exaggeration=12.0):
    feature_bank = torch.load('saved_features/{}_features.pt'.format(filename))
    true_labels = torch.load('saved_features/{}_labels.pt'.format(filename))

    feature_bank = feature_bank.detach().clone().cpu()

    feature_bank_arr = feature_bank.numpy()
    true_labels_arr = true_labels.numpy()
    
    feature_bank_embed = TSNE(perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate="auto", init='pca', metric='cosine')\
        .fit_transform(feature_bank_arr)

    data_1 = np.concatenate((feature_bank_embed, true_labels_arr.reshape(-1, 1)), axis=1)

    fig_1 = plt.figure(figsize=(18, 18))
    p1 = sns.scatterplot(
        data=data_1, x=data_1[:, 0], y=data_1[:, 1], hue=data_1[:, 2], legend='full', palette='nipy_spectral', s=10
    )
    p1.legend_.remove()
    plt.savefig('plots/{}_true_labels.png'.format(filename))
    