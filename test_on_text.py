import gensim
import numpy as np
import re, os
from collections import deque
from keras.models import model_from_json
from keras.optimizers import Adamax
from sklearn.preprocessing import LabelEncoder

delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum())

n = 5 # 5-grams

orig_input = input("Paste text body here: ")

sentences = re.split('(?<=[.!?]) +', orig_input)

# Or load from file option

ngrams = []

for s in sentences:
    sentence = "".join(list(filter(lambda c: str.isalnum(c) or c == " ", list(s)))).split(" ")

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

model = gensim.models.Word2Vec.load("./features/brown100.model")

features = np.empty((len(ngrams), 500)) # 5-grams, each word = 100 dim vectors

for instance, ngram in enumerate(ngrams):
    for i, word in enumerate(ngram.split(" ")):
        if word == "NULL":
            for j in range(100):
                features[instance, i * 100 + j] = 0
            continue

        if word not in model.wv:
            word = word.lower()

        if word not in model.wv:
            for j in range(100):
                features[instance, i * 100 + j] = 1
        else:
            for j in range(100):
                features[instance, i * 100 + j] = model.wv[word][j]

# deserialize model
fmods = open(os.path.join("./data", "pos_model_20val.json"), "rb")
model_json = fmods.read().decode()
fmods.close()
model = model_from_json(model_json)
# model.load_weights(os.path.join("./data/checkpoints_small", "small_pos_weights.50-0.65.hdf5"))
# model.load_weights(os.path.join("./data/checkpoints", "large_pos_weights.17-0.39.hdf5"))
# model.load_weights(os.path.join("./data/checkpoints3", "large_pos_weights.49-0.37.hdf5"))
# model.load_weights(os.path.join("./data/checkpoints_hard_sig3", "large_pos_weights.28-1.15.hdf5"))
model.load_weights(os.path.join("./data/checkpoints_large", "large_pos_weights.36-0.28.hdf5"))
adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy", optimizer=adamax)

# create label lookup tool
str_labels = []
with open("./data/labels.csv", "r") as f:
    for label in f.readlines():
        str_labels.append(label.strip())

le = LabelEncoder()
le.fit(str_labels)

one_hot_predictions = model.predict(features)
predictions = []
for prediction in one_hot_predictions:
    predictions.append(np.argmax(prediction))

predicted_labels = le.inverse_transform(predictions)

label = 0
for sentence in sentences:
    sentlist = sentence.split(" ")
    for i, word in enumerate(sentlist):
        sentlist[i] = word + "(" + predicted_labels[label] + ")"
        label += 1
    print(" ".join(sentlist))
