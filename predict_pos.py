from keras.models import model_from_json
from keras.optimizers import Adamax
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os, csv

# deserialize model
fmods = open(os.path.join("./data", "small_pos_model.json"), "rb")
model_json = fmods.read().decode()
fmods.close()
model = model_from_json(model_json)
# model.load_weights(os.path.join("./data/checkpoints_small", "small_pos_weights.50-0.65.hdf5"))
model.load_weights(os.path.join("./data/checkpoints", "large_pos_weights.17-0.39.hdf5"))
adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy", optimizer=adamax)

# create label lookup tool
str_labels = []
with open("./data/labels.csv", "r") as f:
    for label in f.readlines():
        str_labels.append(label.strip())

le = LabelEncoder()
le.fit(str_labels)

# predict
test_data = np.load("./data/100_split_features/features10.npy")

one_hot_predictions = model.predict(test_data)
predictions = []
for prediction in one_hot_predictions:
    predictions.append(np.argmax(prediction))

predicted_labels = le.inverse_transform(predictions)

with open("./results/predictions/pos10_largetrain_smalltest1.csv", "w") as file:
    wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE)
    wr.writerow(predicted_labels)