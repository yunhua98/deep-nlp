from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adamax
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import os

# features = np.load("./data/100_split_features/features0.npy")
# labels = np.load("./data/100_split_labels/labels0.npy")
# features = np.load("./data/10_split_features3/features0.npy")
# labels = np.load("./data/10_split_labels/labels0.npy")
# features = np.load("./data/features.npy")
# labels = np.load("./data/labels.npy")
# features = np.load("./data/10_split_features/features0.npy")
# labels = np.load("./data/labels0_shuffled.npy")
features = np.load("./chunking/features5.npy")
labels = np.load("./chunking/labels.npy")

# set up model
model = Sequential([
    # input layer
    Dense(128, input_shape=(500,), W_regularizer=l2(0.001)),
    Activation("relu"),
    # Activation("hard_sigmoid"),
    Dropout(0.2),
    # hidden layer
    Dense(128, W_regularizer=l2(0.001)),
    Activation("relu"),
    # Activation("hard_sigmoid"),
    Dropout(0.2),
    # output layer
    # Dense(34), # POS
    Dense(21), # Chunking
    Activation("softmax")
    # Activation("relu"),
    # Activation("hard_sigmoid")
])

adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy", optimizer=adamax)

# save model structure
model_struct = model.to_json()
fmod_struct = open(os.path.join("./chunking", "chunking_model.json"), "wb")
fmod_struct.write(model_struct.encode())
fmod_struct.close()

# train model
checkpoint = ModelCheckpoint(os.path.join("./chunking", "checkpoints",
    "chunking_weights.{epoch:02d}-{val_loss:.2f}.hdf5"), 
    monitor="val_loss", save_best_only=True, mode="min")
# checkpoint = ModelCheckpoint(os.path.join("./data", "checkpoints_large",
#     "large_pos_weights.{epoch:02d}-{val_loss:.2f}.hdf5"), 
#     monitor="val_loss", save_best_only=True, mode="min")
hist = model.fit(features, labels, batch_size=128, nb_epoch=50, shuffle=True,
                 validation_split=0.25, callbacks=[checkpoint])

# plot losses
train_loss = hist.history["loss"]
val_loss = hist.history["val_loss"]
plt.plot(range(len(train_loss)), train_loss, color="red", label="Train Loss")
plt.plot(range(len(train_loss)), val_loss, color="blue", label="Val Loss")          
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc="best")
plt.show()