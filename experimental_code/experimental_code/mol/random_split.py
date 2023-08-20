import pandas as pd
from sklearn.model_selection import train_test_split

full_df = pd.read_csv('raw_datasets/bbbp/full.csv')
X = full_df['smiles']
y = full_df['p_np']

# split dataset into train, val, test sets randomly
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.5, random_state=200, shuffle=True)

train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
val_df = pd.merge(X_val, y_val, left_index=True, right_index=True)
test_df = pd.merge(X_val, y_val, left_index=True, right_index=True)

# save all columns
train_df.to_csv('raw_datasets/bbbp/random/train.csv', index=False)
val_df.to_csv('raw_datasets/bbbp/random/valid.csv', index=False)
test_df.to_csv('raw_datasets/bbbp/random/test.csv', index=False)

train_df.reset_index(inplace=True)
val_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)

# save indices only to match ogb format
train_df.to_csv('dataset/ogbg_molbbbp/split/random/train.csv.gz', index=False, header=False, columns=['index'])
val_df.to_csv('dataset/ogbg_molbbbp/split/random/valid.csv.gz', index=False, header=False, columns=['index'])
test_df.to_csv('dataset/ogbg_molbbbp/split/random/test.csv.gz', index=False, header=False, columns=['index'])
