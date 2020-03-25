import numpy as np

respath = "res.npy"

sequence_len = 700
amino_acid_residues = 38
num_classes = 8


def get_train_dataset():
    ret = np.load(respath)
    return ret


def get_data_labels(D):
    X = D[:, :, 0:amino_acid_residues]
    Y = D[:, :, amino_acid_residues:amino_acid_residues + num_classes]
    return X, Y


def split_with_shuffle(Dataset, seed=None):
    np.random.seed(seed)
    np.random.shuffle(Dataset)
    train_split = int(Dataset.shape[0] * 0.95)
    Train = Dataset[0:train_split, :, :]
    Test = Dataset[train_split:, :, :]
    Validation = Dataset[train_split:, :, :]
    return Train, Test, Validation
