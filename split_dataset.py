import numpy as np
import csv

# features = np.load("./data/features.npy")
# labels = np.load("./data/labels.npy")

ngrams = []
with open("./data/ngrams.csv", "r") as f:
    for ngram in f.readlines():
        ngrams.append(ngram.strip())

labels = []
with open("./data/labels.csv", "r") as f:
    for label in f.readlines():
        labels.append(label.strip())

# num_features, _ = features.shape

# print(num_features) # 1008788 which is roughly 1000000

for segment in range(100):
    with open("./data/100_split_ngrams/ngrams" + str(segment) + ".csv", "w") as file:
        wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE)
        wr.writerow(ngrams[segment * 10000 : segment * 10000 + 10000])
    with open("./data/100_split_pos/pos" + str(segment) + ".csv", "w") as file:
        wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE)
        wr.writerow(labels[segment * 10000 : segment * 10000 + 10000])
    # np.save("./data/10_split_features/features" + str(segment) + ".npy", features[segment * 100000 : segment * 100000 + 100000, :])
    # np.save("./data/10_split_labels/labels" + str(segment) + ".npy", labels[segment * 100000 : segment * 100000 + 100000])