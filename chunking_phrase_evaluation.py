labels = []
with open("./chunking/train_labels.csv", "r") as f:
    for label in f.readlines():
        labels.append(label.strip())

predictions = []
with open("./results/predictions/chunking_train_predictions_5gram.csv", "r") as f:
    for label in f.readlines():
        predictions.append(label.strip())

num_phrases = 0
is_phrase_correct = True
num_correct_phrases = 0
for label, pred in zip(labels, predictions):
    if label != pred:
        is_phrase_correct = False
    if label[0] != "I":
        num_phrases += 1
        if is_phrase_correct:
            num_correct_phrases += 1
        is_phrase_correct = True

print("Correct:", num_correct_phrases)
print("Total:", num_phrases)
print("Accuracy:", num_correct_phrases / num_phrases)