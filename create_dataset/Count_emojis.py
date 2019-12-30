# -*- coding: utf-8 -*-

import codecs
import unicodecsv

from deepmoji.tokenizer import  tokenize

emoji_list = []
emoji_list_file = codecs.open("emoji_uni.txt", 'r')
for line in emoji_list_file.readlines():
    l_ = line.rstrip().split('\t')[0]
    em = unicode(l_, 'unicode-escape')
    emoji_list.append(em)
emoji_list_file.close()
emoji_dics = {}
tweet_dataset_with_label = codecs.open('mini-train/tweet_Data_pp.txt', 'r', encoding='utf-8')
i = 0
for line in tweet_dataset_with_label.readlines():
    emoji = line.rstrip().split(',', 1)[0]

    try :
        emoji_dics[emoji] +=1
    except KeyError:
        emoji_dics[emoji] = 1
    i+=1
    if i % 100 == 0:
        print("has processed {} tweets".format(str(i)))

tweet_dataset_with_label.close()
emoji_pairs = [x for x in emoji_dics.items()  if x[1] >= 100]
emoji_pairs = sorted(emoji_pairs, key= lambda  x : - x[1])
print(emoji_pairs)
print('the length of emojis have showed no less than 10000 times: ', str(len(emoji_pairs)))

emojis_pro= codecs.open('mini-train/emoji_pro.txt', 'w', encoding='utf-8')
i = 0
for emoji, num_ in emoji_pairs:
    str_ = emoji.encode('unicode-escape').decode('string_escape')
    emojis_pro.write(str_)
    emojis_pro.write('\t')
    emojis_pro.write(str(num_))
    emojis_pro.write('\t')
    emojis_pro.write(emoji)
    emojis_pro.write('\n')
    i +=1
    if i %100 == 0:
        print('has written {} emojis'.format(str(i)))

emojis_pro.close()