# pos-taggers
Evaluation of part-of-speech taggers

This code uses NLTK to calculate accuracy, precision, recall and f-score of the following taggers

Stanford POS tagger (arch: left3)
Brill Tagger (base tagger: trigram)
TnT Tagger
CRF Tagger

The Stanford POS tagger is trained on 80% (~45,000) sentences of the Brown Corpus (Universal Tagset).

The predicted tags for 20% sentences are saved using pickle.
