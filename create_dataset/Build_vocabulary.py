# -*- coding: utf-8 -*-

import codecs
import json

import numpy as np
from collections import OrderedDict
import datetime

from deepmoji.word_generator import WordGenerator
from deepmoji.create_vocab import  VocabBuilder

start_time = datetime.datetime.now()
all_start =  start_time
all_sentences = []
i = 0
tweet_data = codecs.open('tweet_Data_final.txt', 'r', 'utf-8')
for line in tweet_data.readlines():
    l_= line.rstrip().split(',')[1]
    # print(l_)
    all_sentences.append(l_)
    i +=1
    # if i >= 100 :
    #     break
    if i % 10000 == 0 :
        print('have read {} tweets'.format(str(i)))
print('finished read {} tweets'.format(str(i)))
end_time = datetime.datetime.now()
print('read file times : ')
print ( end_time - start_time )
tweet_data.close()

print("Create WordGenerator ...")

wg =  WordGenerator(all_sentences)

print('Create Vocabulary Builder ...')
vb =  VocabBuilder(wg)

print('counting words ....')
i = 0
start_time = datetime.datetime.now()
for line in wg.stream:
    valid, words, info = wg.extract_valid_sentence_words(line)

    # Words may be filtered away due to unidecode etc.
    # In that case the words should not be passed on.
    if valid and len(words):
        wg.stats['valid'] += 1
        vb.count_words_in_sentence(words)
        i +=1
        if i % 10000 == 0 :
            end_time = datetime.datetime.now()
            print('have counted {} sentences and time consumed {}'.format(str(i),
                                                                          str(end_time - start_time)))
            start_time = datetime.datetime.now()
    wg.stats['total'] += 1


print ('finished counting {} words .. '.format(str(i)))
dics = vb.word_counts
for it in dics.items():
    print(it)
dtype = ([('word', '|S{}'.format(30)), ('count', 'int')])

ll_ = ["CUSTOM_MASK",
    "CUSTOM_UNKNOWN",
    "CUSTOM_AT",
    "CUSTOM_URL",
    "CUSTOM_NUMBER",
    "CUSTOM_BREAK",
    "CUSTOM_BLANK_6",
    "CUSTOM_BLANK_7",
    "CUSTOM_BLANK_8",
    "CUSTOM_BLANK_9"]
ll_pair = [("CUSTOM_MASK", 0),
           ("CUSTOM_UNKNOWN", 1),
           ("CUSTOM_AT", 2),
           ("CUSTOM_URL", 3),
           ("CUSTOM_NUMBER", 4),
           ("CUSTOM_BREAK", 5),
           ("CUSTOM_BLANK_6", 6),
           ("CUSTOM_BLANK_7", 7),
           ("CUSTOM_BLANK_8",8),
           ("CUSTOM_BLANK_9", 9)]
pairs = [x for x in vb.word_counts.items() if x[0] not in ll_]
np_dict = np.array(pairs, dtype=dtype)
print ("the tokens : ", len(pairs))

print ('sort the tokens ...')
start_time = datetime.datetime.now()
# sort from highest to lowest frequency
data = sorted(np_dict.tolist(), key = lambda  x: - x[1])[:49989]
end_time = datetime.datetime.now()
print ('finished sort, consumed time {}'.format(str(end_time - start_time)))


for j in range(10, 10 + len(data)):
    ll_pair.append((data[j-10][0], j))


# print(ll_pair)
dic = OrderedDict()
for  key_, item_ in ll_pair:
    dic[key_] = item_
print('start writing file ...')
fw = open('myx_vocabulary_final.json', 'w')
json.dump(dic, fw, indent=4, separators=(',', ':'))
fw.close()
print ('finished {}'.format(str(datetime.datetime.now() - all_start)))
