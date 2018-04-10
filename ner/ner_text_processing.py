import csv
from collections import deque

filename = "ner_dataset.csv"
# Dataset from https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data

sentences = []
pos = []
ner = []
with open(filename, "r", encoding="latin-1") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[1] == ",":
            continue
        if row[0] == "":
            sentences[-1].append(row[1].replace(",", ""))
            pos.append(row[2])
            ner.append(row[3])
        elif "Sentence:" in row[0]:
            sentences.append([])
            sentences[-1].append(row[1].replace(",", ""))
            pos.append(row[2])
            ner.append(row[3])

ngrams = []

n = 5 # 5 grams

for sentence in sentences:
    d = deque()
    for i in range(n // 2):
        d.append("NULL")
        sentence.append("NULL")
    d.append(sentence[0])
    sentence.append("NULL")
    for i in range(n // 2):
        d.append(sentence[1 + i])
    for index, word in enumerate(sentence[:-1 - n // 2]):
        ngrams.append(" ".join(d))
        d.popleft()
        d.append(sentence[index + 1 + n // 2])

with open("./ngrams" + str(n) + ".csv", "w") as file:
    wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
    wr.writerow(ngrams)

with open("./pos_labels.csv", "w") as file:
    wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
    wr.writerow(pos)

with open("./ner_labels.csv", "w") as file:
    wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
    wr.writerow(ner)