# -*- coding: utf-8 -*-

import codecs

emoji_wanted =[]
emoji_wanted_file = codecs.open("emoji_processed.txt", 'r')
for line in emoji_wanted_file.readlines():
    l_ = line.rstrip().split('\t')[0]
    em = unicode(l_, 'unicode-escape')
    emoji_wanted.append(em)
emoji_wanted_file.close()

tweet_dataset_with_label = codecs.open('tweet_Data_all_with_10000.txt', 'r', encoding='utf-8')
tweet_dataset = codecs.open('tweet_Data_final.txt', 'w', 'utf-8')
i = 0
for line in tweet_dataset_with_label.readlines():
    l_ = line.rstrip().split(',', 1)
    if len(l_) >= 2  and (l_[0] in emoji_wanted):
        tweet_dataset.write(line)
    i +=1
    if  i%100 ==0 :
        print("has processed {} tweets".format(str(i)))

tweet_dataset_with_label.close()
tweet_dataset.close()