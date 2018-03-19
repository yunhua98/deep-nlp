import gensim, logging, os, nltk

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# sentences = gensim.models.word2vec.BrownCorpus("./brown")
# nltk.download()
sentences = nltk.corpus.brown.sents()

# sentences = [["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"], ["how", "much", "wood", "would", "a", "woodchuck", "chuck", "if", "a", "woodchuck", "could", "chuck", "wood"]]
model = gensim.models.Word2Vec(sentences, min_count=1)#, iter=100)
# model = gensim.models.Word2Vec(sentences, alpha=0.1, cbow_mean=1, size=200, workers=12, min_count=5, sg=0, window=8, iter=20, sample=1e-4, negative=25)

model.accuracy("./word2vec_tests/questions-words.txt")
print(model.similar_by_word("house"))

model.save("./features/brown.model")