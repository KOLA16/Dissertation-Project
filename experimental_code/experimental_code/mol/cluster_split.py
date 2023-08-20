from sklearn.metrics import pairwise_distances
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pandas as pd
import numpy as np
import sys


def get_ecfp_encoding(smiles, radius=2, nBits=1024):
    ecfp_lst = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            print(f"Unable to compile SMILES: {smile}")
            #sys.exit()
            continue
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        ecfp_lst.append(features)
    ecfp_lst = np.array(ecfp_lst)
    return ecfp_lst


def hi_cluster_split(df, mol_threshold=0.5):
    smile_lst = df['smiles'].unique().tolist()
    mol_feature_lst = get_ecfp_encoding(smile_lst)

    # molecule cluster
    smile_cluster_dict = {}
    distance_matrix = pairwise_distances(X=mol_feature_lst, metric="jaccard")
    cond_distance_matrix = squareform(distance_matrix)
    Z = linkage(cond_distance_matrix, method="single")
    cluster_labels = fcluster(Z, t=mol_threshold, criterion="distance")
    for smile, cluster_ in zip(smile_lst, cluster_labels):
        smile_cluster_dict[smile] = cluster_
    df["mol_cluster"] = df["smiles"].map(smile_cluster_dict)

    return df


def unseen_cluster_pair_test_split(df, test_size=0.4):
    mol_clusters = df.mol_cluster.unique()
    test_num_mol_clusters = int(len(mol_clusters) * test_size)

    test_mol_clusters = np.random.choice(mol_clusters, test_num_mol_clusters, replace=False)
    target_df = df[df['mol_cluster'].isin(test_mol_clusters)]
    source_df = df[~(df['mol_cluster'].isin(test_mol_clusters))]

    print(f"Source Training size: {len(source_df)}")
    print(f"Target Training size: {len(target_df)}")
    return source_df, target_df


if __name__ == "__main__":
    np.random.seed(2137)

    df = pd.read_csv("raw_datasets/bbbp/full.csv")
    df = hi_cluster_split(df, mol_threshold=0.5)
    source_train_df, target_train_df = unseen_cluster_pair_test_split(df, test_size=0.4)

    # save all columns
    source_train_df.to_csv("raw_datasets/bbbp/cluster/train_src.csv", index=False)
    target_train_df.to_csv("raw_datasets/bbbp/cluster/train_tar.csv", index=False)
    target_train_df.to_csv("raw_datasets/bbbp/cluster/test_tar.csv", index=False)

    source_train_df.reset_index(inplace=True)
    target_train_df.reset_index(inplace=True)
    target_train_df.reset_index(inplace=True)

    # save indices only to match format required by ogb
    source_train_df.to_csv(
        "dataset/ogbg_molbbbp/split/cluster/train_src.csv.gz", columns=['index'], index=False, header=False
    )
    target_train_df.to_csv(
        "dataset/ogbg_molbbbp/split/cluster/train_tar.csv.gz", columns=['index'], index=False, header=False
    )
    target_train_df.to_csv(
        "dataset/ogbg_molbbbp/split/cluster/test_tar.csv.gz", columns=['index'], index=False, header=False
    )
