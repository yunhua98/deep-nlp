import matplotlib.pyplot as plt
from collections import OrderedDict

labels = OrderedDict()

# with open("./chunking/train_labels.csv", "r") as f:
#     for label in f.readlines():
#         if label == "O\n":
#             continue
#         if label not in labels:
#             labels[label] = 0
#         else:
#             labels[label] += 0

# size = 1015821
with open("./ner/ner_labels.csv", "r") as f:
    for label in f.readlines():
        # size -= 1
        # if size > 101583:
        #     continue
        # if label != "O\n":
        #     labels[label] += 1
        if label.strip() == "O":
            continue
        if label.strip() not in labels:
            labels[label.strip()] = 0
        labels[label.strip()] += 1
sortedKeys = sorted(list(labels.keys()), key=lambda x: labels[x], reverse=True)
plt.bar(range(len(labels)), sorted(list(labels.values()), reverse=True), align='center')
plt.xticks(range(len(labels)), sortedKeys, rotation=60)

plt.show()

