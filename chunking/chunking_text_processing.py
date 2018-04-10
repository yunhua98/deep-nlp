import csv
from collections import deque

filename = "train.txt"

with open(filename) as f:
    lines = f.readlines()

sentences = []
sentences.append([])
labels = []
for line in lines:
    line_list = line.strip().split(" ")
    if len(line_list) != 3:
        continue
    if line_list[0] == ",":
        continue
    if line_list[-1] == "O":
        sentences.append([])
    else:
        sentences[-1].append(line_list[0] + "/" + line_list[2])  # [2] is chunking labels

ngrams = []
labels = []

n = 5 # 5 grams

for sentence in sentences:
    d = deque()
    for i in range(n // 2):
        d.append("NULL")
        sentence.append("NULL")
    d.append(sentence[0].split("/")[0])
    sentence.append("NULL")
    for i in range(n // 2):
        d.append(sentence[1 + i].split("/")[0])
    for index, word in enumerate(sentence[:-1 - n // 2]):
        labels.append(word.split("/")[1])
        ngrams.append(" ".join(d))
        d.popleft()
        d.append(sentence[index + 1 + n // 2].split("/")[0])

with open("./train_ngrams" + str(n) + ".csv", "w") as file:
    wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE)
    wr.writerow(ngrams)

with open("./train_labels.csv", "w") as file:
    wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
    wr.writerow(labels)