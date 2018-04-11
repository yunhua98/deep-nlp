import numpy as np
import csv

features = np.load("./ner/features5.npy")
labels = np.load("./ner/ner_labels.npy")

num_instances, _ = features.shape

np.save("./ner/train_features5.npy", features[:9 * num_instances // 10])
np.save("./ner/train_ner_labels.npy", labels[:9 * num_instances // 10])
np.save("./ner/test_features5.npy", features[9 * num_instances // 10:])
np.save("./ner/test_ner_labels.npy", labels[9 * num_instances // 10:])

# ngrams = []
# with open("./data/ngrams3.csv", "r") as f:
#     for ngram in f.readlines():
#         ngrams.append(ngram.strip())

# labels = []
# with open("./data/labels.csv", "r") as f:
#     for label in f.readlines():
#         labels.append(label.strip())

# print(num_instances) # 1008788 which is roughly 1000000

# for segment in range(100):
#     with open("./data/100_split_ngrams3/ngrams" + str(segment) + ".csv", "w") as file:
#         wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE)
#         wr.writerow(ngrams[segment * 10000 : segment * 10000 + 10000])
#     with open("./data/100_split_pos/pos" + str(segment) + ".csv", "w") as file:
#         wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE)
#         wr.writerow(labels[segment * 10000 : segment * 10000 + 10000])
    # np.save("./data/100_split_features3/features" + str(segment) + ".npy", features[segment * 10000 : segment * 10000 + 10000, :])
    # np.save("./data/10_split_labels/labels" + str(segment) + ".npy", labels[segment * 100000 : segment * 100000 + 100000])