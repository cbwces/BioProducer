#!/usr/bin/env python
# coding: utf-8

import pickle
from tqdm import tqdm
from nltk.corpus import conll2000
from nltk.chunk import tree2conlltags
from nltk.stem import SnowballStemmer

def extract_tuple(chunked_sent):
    # 根据IOB提取conll2000语料的三元组
    sb_stemmer = SnowballStemmer('english')
    words = []
    tags = []
    for word_tag_pair in chunked_sent:
        stemmed_word = sb_stemmer.stem(word_tag_pair[0])
        words.append(stemmed_word) # 提取stem
        tags.append(word_tag_pair[2])
    
    # 记录三元组三个元素的位置
    first_container = []
    second_container = []
    third_container = []
    record_switch = 0 # 开关，判断是否录入
                       #0: 未录入
                       #0.5:第一元组录入中
                       #1:第一元组录入完成，第二元组未录入
                       #1.5:第二元组录入中
                       #2:第二元组录入完成，第三元组未录入
                       #2.5:第三元组录入中
                       #3:完成录入
    for pos, (w, t) in enumerate(zip(words, tags)):
        if t == "B-NP":
            if record_switch == 0:
                first_container.append(w)
                if pos != (len(words) - 1):
                    if tags[pos+1] == "I-NP":
                        record_switch = 0.5
                    else:
                        record_switch = 1
                else:
                    record_switch = 3
                    
            if record_switch == 2:
                third_container.append(w)
                if pos != (len(words) - 1):
                    if tags[pos+1] == "I-NP":
                        record_switch = 2.5
                    else:
                        record_switch = 3
                else:
                    record_switch = 3
                    
        if t == "I-NP":
            if record_switch == 0.5:
                first_container.append(w)
                if pos != (len(words) - 1):
                    if tags[pos+1] != "I-NP":
                        record_switch = 1
                else:
                    record_switch = 3
                    
            if record_switch == 2.5:
                third_container.append(w)
                if pos != (len(words) - 1):
                    if tags[pos+1] != "I-NP":
                        record_switch = 3
                else:
                    record_switch = 3
                    
        if t == "B-VP":
            if record_switch == 1:
                second_container.append(w)
                if pos != (len(words) - 1):
                    if tags[pos+1] == "I-VP":
                        record_switch = 1.5
                    else:
                        record_switch = 2
                else:
                    record_switch = 3
                    
        if t == "I-VP":
            if record_switch == 1.5:
                second_container.append(w)
                if pos != (len(words) - 1):
                    if tags[pos+1] != "I-VP":
                        record_switch = 2
                else:
                    record_switch = 3
                    
    return (first_container, second_container, third_container)

all_tuple_container = []
for tree in tqdm(conll2000.chunked_sents()):
    all_tuple_container.append(extract_tuple(tree2conlltags(tree)))

# 二进制形式保存
with open('../datas/tuple/conll2000_tuple.pkl', 'wb') as bfile:
    pickle.dump(all_tuple_container, bfile, protocol=4)
