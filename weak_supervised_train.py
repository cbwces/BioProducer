#!/usr/bin/env python
# coding: utf-8

# In[7]:


import gc
from tqdm import tqdm
import pickle
import multiprocessing
from collections import defaultdict
from nltk.corpus import conll2000
from nltk.chunk import tree2conlltags
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF


# In[8]:


def str_encoder(raw_list, max_pad=99999):
    # 将字符进行编码， raw_list:经过标记的语料
    max_sent_lenth = 0
    all_tag_list = []
    for raw_in_sent in raw_list:
        all_tag_list.extend(raw_in_sent)
        if len(raw_in_sent) > max_sent_lenth:
            max_sent_lenth = len(raw_in_sent)
    
    if max_pad > max_sent_lenth:
        max_pad = max_sent_lenth

    label_encoder = LabelEncoder()
    label_encoder.fit(all_tag_list)
    encoded_list = np.zeros((len(raw_list), max_sent_lenth))
    for pos, raw_in_sent in enumerate(raw_list):
        encoded_list[pos] = text_padding(label_encoder.transform(raw_in_sent)+1, max_pad)
        
    return encoded_list, label_encoder

def text_padding(real_value_sequence, max_pad):
    # 使所有句子的长度相同（不足的补0）
    if len(real_value_sequence) <= max_pad:
        remain_space = max_pad - len(real_value_sequence)
        return np.concatenate((real_value_sequence, np.zeros(remain_space)))
    
    else:
        return real_value_sequence[:max_pad]


# In[43]:


class ConllDataset(Dataset):
    # 构建输入神经网路的数据集
    def __init__(self, part_speech, iob):
        self.part_speech = part_speech
        self.iob = iob
    
    def __len__(self):
        return len(self.part_speech)
    
    def __getitem__(self, idx):
        return (self.part_speech[idx], self.iob[idx])
    
class TagNet(nn.Module):
    # LSTM+CRF
    def __init__(self, n_words, n_tags, embedding_dim=100, lstm_dim=64):
        super(TagNet, self).__init__()
        
        self.embedding_layer = nn.Embedding(n_words, embedding_dim=embedding_dim)
        self.lstm_layer = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, 
                                  dropout=0.2, bidirectional=True, batch_first=True)
        self.linear_layer = nn.Linear(2*lstm_dim, n_tags)
        self.crf_layer = CRF(n_tags, batch_first=True)
        
    def forward(self, x):
        
        x = self.embedding_layer(x)
        x, state = self.lstm_layer(x)
        x = F.relu(x)
        output = self.linear_layer(x)
        
        return output


# In[44]:


# 提取coll2000语料的词性和IOB标记，逐句存放
train_part_speech_list = []
train_iob_list = []
for tree in conll2000.chunked_sents():
    train_part_speech_in_sent = []
    train_iob_in_sent = []
    tags = tree2conlltags(tree)
    for tg in tags:
        train_part_speech_in_sent.append(tg[1])
        train_iob_in_sent.append(tg[2])
    train_part_speech_list.append(train_part_speech_in_sent)
    train_iob_list.append(train_iob_in_sent)


# In[45]:


# 分别对词性和IOB进行编码
encoded_part_speech, part_speech_encoder = str_encoder(train_part_speech_list)
encoded_iob, iob_encoder = str_encoder(train_iob_list)

# 完成数据输入准备
conll_corpus_dataset = ConllDataset(encoded_part_speech, encoded_iob)
conll_loader = DataLoader(conll_corpus_dataset, batch_size=32, shuffle=True, num_workers=multiprocessing.cpu_count())


# In[46]:


# 统计词性和iob的种类个数
sample_num = encoded_part_speech.shape[0]
n_words = np.unique(encoded_part_speech).shape[0]
n_tags = np.unique(encoded_iob).shape[0]

# 网络结构、优化器、损失函数初始化
net = TagNet(n_words, n_tags).to('cuda')
optimizer = torch.optim.Adam(net.parameters())
# loss = nn.CrossEntropyLoss()

# 训练
EPOCHS = 7
for epk in range(EPOCHS):
    
    for pos, (x, y) in tqdm(enumerate(conll_loader)):
        x = x.long().to('cuda')
        y = y.long().to('cuda')
        mask = torch.where(y > 0, y, torch.zeros(1).long().to('cuda')).bool()
        
        optimizer.zero_grad()
        output = net(x)
        
        loss = net.crf_layer(output, y, mask=mask)
        loss.backward()
        optimizer.step()
        
    torch.cuda.empty_cache()


# In[47]:


def wikidata_dataset(dic_datas, encoder):
    # 对维基数据进行编码
    data_dict = defaultdict(list)
    for key, value in tqdm(dic_datas.items()):
        for sentence in value:
            sent_tag_container = []
            for word_tag_pair in sentence[1]:
                sent_tag_container.append(word_tag_pair[1])
            try:
                data_dict[key].append(encoder.transform(sent_tag_container))
            except ValueError:
                continue
            
    return data_dict


# In[48]:


gc.collect()

# 读取词标注文本
with open("../datas/token/animal_text_word_token.pkl", 'rb') as f:
    animal_tokens = pickle.load(f)
with open("../datas/token/plant_text_word_token.pkl", 'rb') as f:
    plant_tokens = pickle.load(f)


# In[49]:


animal_tokens_dict = wikidata_dataset(animal_tokens, part_speech_encoder)
plant_tokens_dict = wikidata_dataset(plant_tokens, part_speech_encoder)


# In[ ]:


example = np.zeros((len(animal_tokens_dict['Ecology']), len(encoded_part_speech)))
for p, i in enumerate(animal_tokens_dict['Ecology']):
    example[p] = text_padding(i, len(encoded_part_speech))
    
with torch.no_grad():
    out = net(torch.from_numpy(example).long().to('cuda'))

with open('../datas/tags/animal_tager.pkl, 'wb') as bfile:
    pickle.dump(all_tuple_container, bfile, protocol=4)


# In[88]:


example = np.zeros((len(plant_tokens_dict['Ecology']), len(encoded_part_speech)))
for p, i in enumerate(plant_tokens_dict['Ecology']):
    example[p] = text_padding(i, len(encoded_part_speech))
    
with torch.no_grad():
    out = net(torch.from_numpy(example).long().to('cuda'))

with open('../datas/tags/plant_tager.pkl, 'wb') as bfile:
    pickle.dump(all_tuple_container, bfile, protocol=4)

