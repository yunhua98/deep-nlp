import gensim, csv
from collections import deque

sentences = gensim.models.word2vec.BrownCorpus("./brown")

n = 3 # must be odd

# replace rare words with unknown later?

ngrams = []
# labels = []

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
#         labels.append(word.split("/")[1])
        ngrams.append(" ".join(d))
        d.popleft()
        d.append(sentence[index + 1 + n // 2].split("/")[0])

with open("./data/ngrams" + str(n) + ".csv", "w") as file:
    wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE)
    wr.writerow(ngrams)

# with open("./data/labels.csv", "w") as file:
#     wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE)
#     wr.writerow(labels)