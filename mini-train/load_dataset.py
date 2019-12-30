import pandas as pd
import codecs
from sklearn.model_selection import train_test_split

dataset = codecs.open('tweet_Data_pp.txt','r', 'utf-8')
emojis = codecs.open('emoji_pro.txt', 'r' )
emoji_dic ={}
i=1
for line in emojis.readlines():
    l_ = line.rstrip().split('\t')[0]
    l_ = unicode(l_, 'unicode-escape')
    emoji_dic[l_] = i
    i+=1
print(emoji_dic)
dataframe = pd.DataFrame(columns=['label', 'text'])
for line in dataset.readlines():
    emoji, text = line.rstrip().split(',,', 1)
    # print (emoji)
    label = emoji_dic[emoji]
    new = pd.DataFrame({"label": label, "text": text}, index=["0"])
    dataframe = dataframe.append(new, ignore_index = True)
dataset.close()


### split into train test val
all_dataset = {}
for j in range(1, i) :
    df = dataframe[dataframe['label'] == j].reset_index()
    all_dataset[j] =df

# for label, df in all_dataset.items():
#     print('label :', label)
#     print (df)

test_ind = []
val_ind = []

for now_label in range(1,i):
    now_datafame = all_dataset[now_label]
    length = now_datafame.shape[0]
    ind = list(range(length))
    size_of_test = float(36)/length
    size_of_val = 36/(length-36)
    ind_train, ind_test = train_test_split(ind, test_size=size_of_test)
    ind_train, ind_val = train_test_split(ind_train, test_size=size_of_val)
    print ("ind_test:",ind_test)
    new_test_ind=[now_datafame.iat[row,0] for row in ind_test]
    test_ind.extend(new_test_ind)
    new_val=[now_datafame.iat[row,1] for row in ind_val]
    val_ind.extend(new_val)
    break
print (test_ind)


# data_list=dataframe.to_dict(orient='list')
# for pair in data_list.items():
#     print (pair)
