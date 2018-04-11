import gensim
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# model = gensim.models.Word2Vec.load("./features/brown100.model")
# model = gensim.models.Word2Vec.load("./chunking/conll2000.model")
model = gensim.models.Word2Vec.load("./ner/conll2002.model")

ngrams = []
with open("./ner/ngrams5.csv", "r") as f:
    for ngram in f.readlines():
        ngrams.append(ngram)

str_train_labels = []
with open("./ner/ner_labels.csv", "r") as f:
    for label in f.readlines():
        str_train_labels.append(label)

str_labels = []
with open("./ner/ner_labels.csv", "r") as f:
    for label in f.readlines():
        str_labels.append(label)

lb = LabelBinarizer()
lb.fit(str_train_labels)
labels = np.empty((len(str_labels), lb.classes_.size))
for i, label in enumerate(lb.transform(str_labels)):
    for j, el in enumerate(label):
        labels[i, j] = el

features = np.empty((len(ngrams), 500)) # 100 dim vectors

for instance, ngram in enumerate(ngrams):
    for i, word in enumerate(ngram.split(" ")):
        if word == "NULL":
            for j in range(100):
                features[instance, i * 100 + j] = 0
        elif word not in model.wv:
            for j in range(100):
                features[instance, i * 100 + j] = 1
        else:
            for j in range(100):
                features[instance, i * 100 + j] = model.wv[word][j]

np.save("./ner/features5.npy", features)
np.save("./ner/ner_labels.npy", labels)