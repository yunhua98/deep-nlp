labels = []
with open("./ner/ner_labels.csv", "r") as f:
# with open("./data/100_split_pos/pos10.csv", "r") as f:
# with open("./data/100_split_pos/pos1.csv", "r") as f:
    for label in f.readlines():
        labels.append(label.strip())

# predictions0 = []
# # with open("./results/predictions/ner_test_split_predictions_5gram.csv", "r") as f:
# with open("./results/predictions/pos10_largetrain_smalltest_3ngram0.csv", "r") as f:
#     for label in f.readlines():
#         predictions0.append(label.strip())

# predictions1 = []
# with open("./results/predictions/pos10_largetrain_smalltest0.csv", "r") as f:
#     for label in f.readlines():
#         predictions1.append(label.strip())

# predictions2 = []
# with open("./results/predictions/pos10_largetrain_smalltest1.csv", "r") as f:
#     for label in f.readlines():
#         predictions2.append(label.strip())

predictions = []
with open("./results/predictions/ner_test_augmented_predictions_5gram.csv", "r") as f:
    for label in f.readlines():
        predictions.append(label.strip())


# for p0, p1, p2 in zip(predictions0, predictions1, predictions2):
#     if p0 == p1 and p1 == p2:
#         predictions.append(p0)
#     elif p0 == p1:
#         predictions.append(p0)
#     elif p1 == p2:
#         predictions.append(p1)
#     elif p2 == p0:
#         predictions.append(p2)
#     else:
#         predictions.append(p0)

print("Accuracy:", sum(list(map(lambda x, y: 1 if x == y else 0, labels[-1 * len(predictions):], predictions))) / len(predictions))

# NER evaluation, ignores O's
actual_named_entities = 0
correct_named_entities = 0
incorrect_named_entities = 0
false_positives = 0
false_negatives = 0
for label, prediction in zip(labels[-1 * len(predictions):], predictions):
    if label != "O":
        actual_named_entities += 1
        if prediction == "O":
            false_negatives += 1
        elif prediction == label:
            correct_named_entities += 1
        else:
            incorrect_named_entities += 1
    else:
        if prediction != "O":
            false_positives += 1

print("Named entities correctly labeled:", correct_named_entities / actual_named_entities)
print("Named entities incorrectly labeled:", incorrect_named_entities / actual_named_entities)
print("Named entities missed:", false_negatives / actual_named_entities)
print("Percentage false positives:", false_positives / (len(predictions) - actual_named_entities))