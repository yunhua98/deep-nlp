labels = []
with open("./data/100_split_pos/pos10.csv", "r") as f:
    for label in f.readlines():
        labels.append(label.strip())

predictions = []
with open("./results/predictions/pos10_largetrain_smalltest1.csv", "r") as f:
    for label in f.readlines():
        predictions.append(label.strip())

print("Accuracy:", sum(list(map(lambda x, y: 1 if x == y else 0, labels, predictions))) / len(labels))