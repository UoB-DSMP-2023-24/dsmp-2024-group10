import numpy as np
import pandas as pd
import os
import torch
from sklearn import preprocessing

# Function to encode the seq
def one_hot_encoding(seq, max_len):
    pro_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    encoded_seq = []
    for pro in seq:
        vector = [0]*20
        vector[pro_dict[pro]] = 1
        encoded_seq.append(vector)
    for i in range(max_len-len(seq)):
        encoded_seq.append([0]*20)
    return np.array(encoded_seq)

# Read csv into Python
path = os.path.abspath(".")
df = pd.read_csv(path + "/Data/vdjdb.txt", sep="\t")

a = "CASSYLPGQGDHYSNQPQHF"
print(one_hot_encoding(a, 38))

# # Encode mhc.b
# label_encoder = preprocessing.LabelEncoder()
# encoded_labels = label_encoder.fit_transform(df["mhc.b"])

# # Encode mhc.a
# label_encoder1 = preprocessing.LabelEncoder()
# encoded_labels1 = label_encoder1.fit_transform(df["mhc.a"])

# # Encode all the cdr3 seq in csv
# max_len = df["cdr3"].str.len().max()
# encoded_normalized_cdr3 = [one_hot_encoding(seq, max_len) for seq in df['cdr3']]

# data_label_pairs = list(zip(encoded_normalized_cdr3, encoded_labels))
# data_label_pairs1 = list(zip(encoded_normalized_cdr3, encoded_labels1))

# features = torch.tensor(np.array([pair[0] for pair in data_label_pairs]), dtype=torch.float)
# labels = torch.tensor([pair[1] for pair in data_label_pairs], dtype=torch.long)
# labels1 = torch.tensor([pair[1] for pair in data_label_pairs1], dtype=torch.long)

# torch.save(features, path + "/Data/cdr3_features.pt")
# torch.save(labels, path + "/Data/mhcb_labels.pt")
# torch.save(labels1, path + "/Data/mhca_labels.pt")