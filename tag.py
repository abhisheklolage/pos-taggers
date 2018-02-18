import math
import nltk
import os
import pickle
from nltk.corpus import brown

# import the trained taggers

from nltk.tag import StanfordPOSTagger
from nltk.tag import CRFTagger
from nltk.tag import RegexpTagger
from nltk.tag import BrillTaggerTrainer
from nltk.tag import tnt

sentences = [sentence for sentence in brown.tagged_sents(tagset='universal')]
split_idx = math.floor(len(sentences) * 0.20)
testing_sentences = sentences[0:split_idx]
training_sentences = sentences[split_idx:]

# original tags of sentences in the brown corpus
ground_tags = [[tag for word, tag in testing_sentences[sentence_idx]] for
        sentence_idx in range(split_idx)]
testing_tokens = [[word for word, tag in testing_sentences[sentence_idx]] for
        sentence_idx in range(split_idx)]

if (True):
    print ("#######")
    # get trained stanford model
    stanford_model = StanfordPOSTagger(os.environ.get('STANFORD_BROWN_MODEL'))

    # stanford_tokens_tags = [stanford_model.tag(token_list) for token_list in testing_tokens]
    stanford_tokens_tags = []
    stanford_token_tags = stanford_model.tag_sents(testing_tokens)

    stanford_tags = [[tag for word, tag in stanford_token_tags[sentence_idx]] for sentence_idx in range(split_idx)]
    # save computed tags
    pickle.dump(stanford_tags, open("stanford_brown_20_tags_all.pd", "wb"))
    print (len(stanford_tags))
    print (stanford_tags[0])
    print (stanford_tags[1])

if (True):
    print ("#######")
    print ("Training CRF tagger...")
    crf_tagger = CRFTagger()
    crf_tagger.train(training_sentences, '/tmp/crf_tagger_80.model')
    #crf_tagger.set_model_file('./crf_new')
    print ("Done training CRF tagger...")
    crf_tokens_tags = []

    crf_token_tags = crf_tagger.tag_sents(testing_tokens)
    crf_tags = [[tag for word, tag in crf_token_tags[sentence_idx]] for sentence_idx in range(split_idx)]
    # save computed tags
    pickle.dump(crf_tags, open("crf_brown_20_tags.pd", "wb"))
    print (len(crf_tags))
    print (crf_tags[0])
    print (crf_tags[1])

if (True):
    print ("#######")

    print ("Training TnT tagger...")
    tnt_tagger = tnt.TnT()
    tnt_tagger.train(training_sentences)
    print ("Done training TnT tagger...")
    tnt_tokens_tags = []
    tnt_token_tags = tnt_tagger.tag_sents(testing_tokens)
    tnt_tags = [[tag for word, tag in tnt_token_tags[sentence_idx]] for sentence_idx in range(split_idx)]
    pickle.dump(crf_tags, open("tnt_brown_20_tags.pd", "wb"))
    print (len(tnt_tags))
    print (tnt_tags[0])
    print (tnt_tags[1])

if (True):
    default_tagger=nltk.DefaultTagger('NOUN')
    unigram_tagger=nltk.UnigramTagger(training_sentences,backoff=default_tagger)
    bigram_tagger=nltk.BigramTagger(training_sentences,backoff= unigram_tagger)
    trigram_tagger=nltk.TrigramTagger(training_sentences,backoff=bigram_tagger)
    baseline = trigram_tagger
    Template._cleartemplates() #clear any templates created in earlier tests
    templates = [Template(Pos([-1])), Template(Pos([-1]), Word([0]))]

    #construct a BrillTaggerTrainer
    tt = BrillTaggerTrainer(baseline, templates, trace=3)

    tagger1 = tt.train(training_data, max_rules=10)
    # tagger2 = tt.train(training_data, max_rules=10, min_acc=0.99)

    print (testing_sentences[TEST_EXAMPLE_SENTENCE_INDEX])

    test_example = [word for word,_ in testing_sentences[TEST_EXAMPLE_SENTENCE_INDEX]]

    bttrigram = tagger1.tag_sents(testing_tokens)
    brill_tags_trigram = [[tag for word, tag in bttrigram[sentence_idx]] for sentence_idx in range(split_idx)]

    pickle.dump(brill_tags_trigram, open("brill_trigram_brown_20_tags.pd", "wb"))
