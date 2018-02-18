from nltk.corpus import brown
from sklearn import metrics
import math
import pickle
import os
import sys
#sys.path.insert(0, './LessConfusingMatrix/')
import confusion_matrix
import stats

def getStats(cmat):
    print("Accuracy :", cmat.accuracy(1))
    print("Precision:", cmat.precision(1))
    print("Recall   :", cmat.recall(1))
    print("F1-Score :", cmat.f1_score(1))

stanford_tags = pickle.load(open("stanford_brown_20_tags_all.pd", "rb"))

crf_tags = pickle.load(open("crf_brown_20_tags.pd", "rb"))

tnt_tags = pickle.load(open("tnt_brown_20_tags.pd", "rb"))

brill_trigram_tags = pickle.load(open("brill_trigram_brown_20_tags.pd", "rb"))

brill_backoff_tags = pickle.load(open("brill_backoff_brown_20_tags.pd", "rb"))

sentences = [sentence for sentence in brown.tagged_sents(tagset='universal')]
split_idx = math.floor(len(sentences) * 0.20)
testing_sentences = sentences[0:split_idx]
training_sentences = sentences[split_idx:]

# original tags of sentences in the brown corpus
ground_tags = [[tag for word, tag in testing_sentences[sentence_idx]] for
        sentence_idx in range(split_idx)]

confmat = confusion_matrix.ConfusionMatrix()

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Stanford wrt Ground")
confmat.reset()
confmat.add(ground_tags, stanford_tags)
confmat.pprint()
cme = stats.Evaluator(confmat)
getStats(cme)


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("CRF wrt Ground")
confmat.reset()
confmat.add(ground_tags, crf_tags)
confmat.pprint()
cme = stats.Evaluator(confmat)
getStats(cme)


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("TnT wrt Ground")
confmat.reset()
confmat.add(ground_tags, tnt_tags)
confmat.pprint()
cme = stats.Evaluator(confmat)
getStats(cme)

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Brill on Trigram wrt Ground")
confmat.reset()
confmat.add(ground_tags, brill_trigram_tags)
confmat.pprint()
cme = stats.Evaluator(confmat)
getStats(cme)
