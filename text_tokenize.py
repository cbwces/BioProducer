#!/usr/bin/env python
# coding: utf-8

import re
import copy
import json
from tqdm import tqdm
import pickle
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from transformers import BertTokenizer

def load_file(path):    # 读取文本

    bio_dict = {}
    
    with open(path) as f:
        raw_text = json.load(f)
    json_text = json.loads(raw_text)
    
    json_text = list(json_text.values())[0]
    for page_num in range(len(json_text)):
        species = list(json_text[page_num].keys())[0]
        text = list(json_text[page_num].values())[0]['text']
        bio_dict[species] = text
        
    return bio_dict

def file_preprocess(text):    # 文本预处理
    
    regex_ws=re.compile("\s+")
    enter = re.compile('\n+')
    reference = re.compile('\[\d+\]')
    html = re.compile(r'<.*?>')
    url = re.compile("(https?:\/\/(?:www\.|(?!www)|(?:xmlns\.))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})")
    email = re.compile('@[A-z0-9_]+')
    num = re.compile(r"(\s\d+)")

    text = text.replace("&amp;","&").replace("&lt;","<").replace("&gt;",">").replace("%20", " ")
    text = enter.sub(" ", text)
    text = regex_ws.sub(" ", text)
    text = reference.sub(" ", text)
    text = regex_ws.sub(" ", text)
    text = html.sub(" ", text)
    text = regex_ws.sub(" ", text)
    text = url.sub(" ", text)
    text = regex_ws.sub(" ", text)
    text = email.sub(" ", text)
    text = regex_ws.sub(" ", text)
    text = num.sub(" ", text)
    text = regex_ws.sub(" ", text)

    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"We're", "We are", text)
    text = re.sub(r"That's", "That is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"Can't", "Cannot", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"don\x89Ûªt", "do not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"What's", "What is", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"There's", "There is", text)
    text = re.sub(r"He's", "He is", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"You're", "You are", text)
    text = re.sub(r"I'M", "I am", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"i'm", "I am", text)
    text = re.sub(r"I\x89Ûªm", "I am", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"Isn't", "is not", text)
    text = re.sub(r"Here's", "Here is", text)
    text = re.sub(r"you've", "you have", text)
    text = re.sub(r"you\x89Ûªve", "you have", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"it\x89Ûªs", "it is", text)
    text = re.sub(r"doesn\x89Ûªt", "does not", text)
    text = re.sub(r"It\x89Ûªs", "It is", text)
    text = re.sub(r"Here\x89Ûªs", "Here is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"I\x89Ûªve", "I have", text)
    text = re.sub(r"y'all", "you all", text)
    text = re.sub(r"can\x89Ûªt", "cannot", text)
    text = re.sub(r"would've", "would have", text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"wouldn\x89Ûªt", "would not", text)
    text = re.sub(r"We've", "We have", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"Y'all", "You all", text)
    text = re.sub(r"Weren't", "Were not", text)
    text = re.sub(r"Didn't", "Did not", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"DON'T", "DO NOT", text)
    text = re.sub(r"That\x89Ûªs", "That is", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"should've", "should have", text)
    text = re.sub(r"You\x89Ûªre", "You are", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"Don\x89Ûªt", "Do not", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"They're", "They are", text)
    text = re.sub(r"Can\x89Ûªt", "Cannot", text)
    text = re.sub(r"you\x89Ûªll", "you will", text)
    text = re.sub(r"I\x89Ûªd", "I would", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"i've", "I have", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"I've", "I have", text)
    text = re.sub(r"Don't", "do not", text)
    text = re.sub(r"I'll", "I will", text)
    text = re.sub(r"I'd", "I would", text)
    text = re.sub(r"Let's", "Let us", text)
    text = re.sub(r"you'd", "You would", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"Ain't", "am not", text)
    text = re.sub(r"Haven't", "Have not", text)
    text = re.sub(r"Could've", "Could have", text)
    text = re.sub(r"youve", "you have", text)  
    text = re.sub(r"donå«t", "do not", text) 
    
    return text

animal_text = load_file("../datas/raw/wikidatas_animals.json")
plant_text = load_file("../datas/raw/0wikidatas_plants.json")

# 词标记化（转化为LSTM+CRF结构输入格式）
animal_text_word_token = copy.deepcopy(animal_text)
plant_text_word_token = copy.deepcopy(plant_text)
sb_stemmer = SnowballStemmer('english')

for text_dict in [animal_text_word_token, plant_text_word_token]:
    for i in tqdm(text_dict.keys()):
        text_container = []
        full_doc = file_preprocess(text_dict[i])
        doc_split_by_sent = sent_tokenize(full_doc)
        for sent in doc_split_by_sent:
            word_in_sent = word_tokenize(sent)
            wd = [sb_stemmer.stem(word) for word in word_in_sent]
            tg = pos_tag(word_in_sent)
            text_container.append((wd, tg))
        text_dict[i] = text_container

# 二进制形式保存
with open('../datas/animal_text_word_token.pkl', 'wb') as bfile:
    pickle.dump(animal_text_word_token, bfile, protocol=4)

with open('../datas/plant_text_word_token.pkl', 'wb') as bfile:
    pickle.dump(plant_text_word_token, bfile, protocol=4)

# 词标记化（转化为BERT结构输入格式）
animal_text_bert_token = copy.deepcopy(animal_text)
plant_text_bert_token = copy.deepcopy(plant_text)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for text_dict in [animal_text_bert_token, plant_text_bert_token]:
    for i in tqdm(text_dict.keys()):
        text_container = []
        full_doc = file_preprocess(text_dict[i])
        doc_split_by_sent = sent_tokenize(full_doc)
        for sent in doc_split_by_sent:
            bert_tokens = tokenizer.tokenize("[CLS] " + sent + " [SEP]")
            tg = pos_tag(word_tokenize(sent))
            text_container.append((bert_tokens, tg))
        text_dict[i] = text_container

# 二进制形式保存
with open('../datas/token/animal_text_bert_token.pkl', 'wb') as bfile:
    pickle.dump(animal_text_bert_token, bfile, protocol=4)

with open('../datas/token/plant_text_bert_token.pkl', 'wb') as bfile:
    pickle.dump(plant_text_bert_token, bfile, protocol=4)
