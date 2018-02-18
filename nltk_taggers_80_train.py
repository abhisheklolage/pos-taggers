import nltk, math
import pickle

tagged_sentences = [sentence for sentence in nltk.corpus.brown.tagged_sents(tagset='universal')]
# hold out 20% for testing, get index for 20% split
split_idx = math.floor(len(tagged_sentences)*0.20)
testing_sentences = tagged_sentences[0:split_idx]
training_sentences = tagged_sentences[split_idx:]

# split_idx = 10000
training_sentences = tagged_sentences[0:split_idx]
TEST_EXAMPLE_SENTENCE_INDEX = 676 # nice short example in testing sentences

if(True):
    print ("Training TnT tagger...")
    tnt_tagger = nltk.tag.tnt.TnT()
    tnt_tagger.train(training_sentences)
    print ("Done training TnT tagger...")
    pickle.dump(tnt_tagger, open( "TnT_Brown_80.model", "wb"))
    # tnt_tagger_preds = tnt_tagger.tag_sents([[word for word,_ in test_sent] for test_sent in testing_sentences])
    # print(tnt_tagger_preds[TEST_EXAMPLE_SENTENCE_INDEX])

    print (tnt_tagger.tag(testing_sentences[TEST_EXAMPLE_SENTENCE_INDEX]))

    print ("Training CRF tagger...")
    crf_tagger = nltk.tag.CRFTagger()
    crf_tagger.train(training_sentences, '/tmp/crf_tagger_80.model')
    print ("Done training CRF tagger...")
    # crf_tagger_preds = crf_tagger.tag_sents([[word for word,_ in test_sent] for test_sent in testing_sentences])
    # print(crf_tagger_preds[TEST_EXAMPLE_SENTENCE_INDEX])

    test_example = [word for word,_ in testing_sentences[TEST_EXAMPLE_SENTENCE_INDEX]]
    print (crf_tagger.tag(test_example))
