import json
import pickle

import numpy as np
import pandas as pd
import codecs

import datetime
start_time = datetime.datetime.now()
columns=['label','texts']
dataset = pd.read_table('tweet_Data_final.txt',header=None,sep=',',names=columns)
emojis = codecs.open('emoji_processed_final.txt', 'r' )
# print(dataset)
emoji_dic ={}
label_length = 0
for line in emojis.readlines():
    label_length +=1
    l_ = line.rstrip().split('\t')[0]
    l_ = unicode(l_, 'unicode-escape')
    emoji_dic[l_] = label_length
    # print(l_)
    if label_length >= 64 :
        break
emojis.close()
end_time = datetime.datetime.now()
print("finish read files in {}".format(end_time - start_time))



key_ = emoji_dic.keys()

dataset['label'] = dataset['label'].str.decode('utf8')



print(dataset)
print(type(dataset.iat[0,0]))
start_time = datetime.datetime.now()

for i in range(len(dataset)):
    em = dataset.iat[i,0]
    if em in key_ :
        dataset.iat[i, 0] = emoji_dic[em]
    else :
        dataset.iat[i,0] = 0

    if i% 100000 == 0:
        end_time = datetime.datetime.now()
        print("processed {} consumed {}".format(str(i + 1), end_time - start_time))
        start_time = datetime.datetime.now()
print(dataset)

dataset.to_pickle('data_labeled.pkl')



data = pd.read_pickle('data_labeled.pkl')
# print(data)
print(type(data.iat[0,1]))
data = data[~ data['label'].isin([0])].sort_values(by=['label']).reset_index(drop = True).dropna()
print(data)
start_time = datetime.datetime.now()

data['texts'] = data['texts'].str.decode('ascii', 'ignore')

print(data)
data.to_pickle('data_pro.pkl')

# new_df = pd.read_pickle('data.pkl')
# print(new_df)