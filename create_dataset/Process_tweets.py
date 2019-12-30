# -*- coding: utf-8 -*-

import codecs
import unicodecsv

from deepmoji.tokenizer import  tokenize

emoji_list = []
emoji_wanted =[]
emoji_list_file = codecs.open("emoji_uni.txt", 'r')
emoji_wanted_file = codecs.open("mini-train/emoji_pro.txt", 'r')
for line in emoji_list_file.readlines():
    l_ = line.rstrip().split('\t')[0]
    em = unicode(l_, 'unicode-escape')
    emoji_list.append(em)
emoji_list_file.close()

for line in emoji_wanted_file.readlines():
    l_ = line.rstrip().split('\t')[0]
    em = unicode(l_, 'unicode-escape')
    emoji_wanted.append(em)
emoji_wanted_file.close()

#
# print(emoji_list)


def separate_emojis_and_text(text):
    emoji_chars = []
    non_emoji_chars = []
    for c in text:
        if c in emoji_list:
            if c not in emoji_chars:
                emoji_chars.append(c)
        else:
            non_emoji_chars.append(c)
    return emoji_chars, ''.join(non_emoji_chars)





tweet_file = codecs.open('playplay/tweet_Data.txt', 'r')

tweet_dataset_with_label = codecs.open('playplay/tweet_Data_pp.txt', 'w', 'utf-8')
j = 0
for line in tweet_file.readlines():
    l_ = unicode(line.rstrip().decode('utf-8')) #.decode('unicode-escape')
    emojis, words =  separate_emojis_and_text(l_)
    length_ = len(emojis)
    # print (type(emojis[0]))
    # words = words.encode('utf-8')
    # em = emojis[0].decode.encode('utf-8')

    for i in range(length_):
        em = emojis[i]
        if em in emoji_wanted :
            tweet_dataset_with_label.write(em)
            tweet_dataset_with_label.write(',')
            tweet_dataset_with_label.write(words)
            tweet_dataset_with_label.write('\n')
    if j % 100 ==0 :
        print ('have processed {} tweets'.format(j))
    j+=1

    if j > 4700000:
        print('finished')
        break
tweet_file.close()
tweet_dataset_with_label.close()
# import emoji
#
# print (type(emoji.UNICODE_EMOJI.keys()[0]))