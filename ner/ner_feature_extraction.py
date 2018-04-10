import gensim, logging, os, csv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

filename = "ner_dataset.csv"

sentences = []
with open(filename, "r", encoding="latin-1") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[1] == ",":
            continue
        if row[0] == "":
            sentences[-1].append(row[1].replace(",", ""))
        elif "Sentence:" in row[0]:
            sentences.append([])
            sentences[-1].append(row[1].replace(",", ""))

model = gensim.models.Word2Vec(sentences, min_count=1, iter=500) # note 500 iterations

print(model.similar_by_word("costs"))

model.save("./conll2002.model")