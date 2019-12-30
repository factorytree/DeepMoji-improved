# -*- coding: utf-8 -*-
from __future__ import print_function
import codecs

# Convert emoji_data to a file contains all unicode for emojie
emoji_data = open('emoji-data.txt', 'r')

emoji_uni = codecs.open('emoji_uni.txt', 'w', 'utf-8')

for line in emoji_data.readlines():
    left, right = line.rstrip().split('[')
    right1, right2 = right.split(']')
    l_ = ' '.join([left, right1, right2]).split(' ')
    res = [x for x in l_ if x !='']
    nums_ = int(res[5])
    start_ = res[0].split('.')[0]
    length_ = len(start_)
    start_ = int(start_, 16)
    # print(start_, nums_)
    for j in range(start_ , start_ + nums_):
        uni = hex(j)
        uni = '0' * (length_ - len(uni) + 2) + uni[2:]
        if length_ == 4 :
            uni = '\u' + uni
        if length_ > 4 :
            uni = '\U' + '0' * (8 - length_) + uni
        em = unicode(uni, 'unicode-escape')
        emoji_uni.write(uni)
        emoji_uni.write('\t')
        emoji_uni.write(em)
        emoji_uni.write('\n')

        print(uni, ' : ', em, end= ',')
    print(' ')
emoji_data.close()
emoji_uni.close()
