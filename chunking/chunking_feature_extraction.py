import gensim, logging, os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

filename = "train.txt"

with open(filename) as f:
    lines = f.readlines()

sentences = []
sentences.append([])
labels = []
for line in lines:
    line_list = line.strip().split(" ")
    if len(line_list) != 3:
        continue
    if line_list[0] == ",":
        continue
    if line_list[-1] == "O":
        sentences.append([])
    else:
        sentences[-1].append(line_list[0])  # [2] is chunking labels

model = gensim.models.Word2Vec(sentences, min_count=1, iter=500) # note 500 iterations

print(model.similar_by_word("costs"))

model.save("./conll2000.model")