import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder

# labels = np.load("./data/10_split_labels/labels0.npy")

# np.random.shuffle(labels)

# np.save("./data/labels0_shuffled.npy", labels)

# create label lookup tool
str_labels = []
with open("./data/labels.csv", "r") as f:
    for label in f.readlines():
        str_labels.append(label.strip())

le = LabelEncoder()
le.fit(str_labels)

shuffled_labels = np.load("./data/labels0_shuffled.npy")
row, col = shuffled_labels.shape
shuffled_labels_list = []
for i in range(row):
    shuffled_labels_list.append(np.argmax(shuffled_labels[[i], :]))
shuffled_pos_labels = le.inverse_transform(shuffled_labels_list)

with open("./data/labels0_shuffled.csv", "w") as file:
    wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE)
    wr.writerow(shuffled_pos_labels)