from keras.models import model_from_json
from keras.optimizers import Adamax
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os, csv

# deserialize model
# fmods = open(os.path.join("./data", "pos_model_20val.json"), "rb")
# fmods = open(os.path.join("./data", "pos_model_25val.json"), "rb")
# fmods = open(os.path.join("./chunking", "chunking_model.json"), "rb")
fmods = open(os.path.join("./ner", "ner_model.json"), "rb")
model_json = fmods.read().decode()
fmods.close()
model = model_from_json(model_json)
# model.load_weights(os.path.join("./data/checkpoints_small", "small_pos_weights.50-0.65.hdf5"))
# model.load_weights(os.path.join("./data/checkpoints", "large_pos_weights.17-0.39.hdf5"))
# model.load_weights(os.path.join("./data/checkpoints3", "large_pos_weights.49-0.37.hdf5"))
# model.load_weights(os.path.join("./data/checkpoints_hard_sig3", "large_pos_weights.28-1.15.hdf5"))
# model.load_weights(os.path.join("./data/checkpoints_large", "large_pos_weights.36-0.28.hdf5"))
# model.load_weights(os.path.join("./chunking/checkpoints", "chunking_weights.19-0.34.hdf5"))
model.load_weights(os.path.join("./ner/checkpoints", "ner_weights.44-0.17.hdf5"))
adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy", optimizer=adamax)

# create label lookup tool
str_labels = []
with open("./ner/ner_labels.csv", "r") as f:
    for label in f.readlines():
        str_labels.append(label.strip())

le = LabelEncoder()
le.fit(str_labels)

# predict
# test_data = np.load("./data/100_split_features/features0.npy")
test_data = np.load("./ner/features5.npy")

one_hot_predictions = model.predict(test_data)
predictions = []
for prediction in one_hot_predictions:
    predictions.append(np.argmax(prediction))

predicted_labels = le.inverse_transform(predictions)

with open("./results/predictions/ner_train_predictions_5gram.csv", "w") as file:
    wr = csv.writer(file, delimiter="\n", quoting=csv.QUOTE_NONE)
    wr.writerow(predicted_labels)