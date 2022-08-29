import numpy as np
import pandas as pd
import os
import ast
import csv
import re
import keras
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.utils.vis_utils import plot_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
with open('usual_train.txt') as train_file:
    train_document = train_file.read()
with open('usual_test_labeled.txt') as train_file:
    test_document = train_file.read()

train_list = ast.literal_eval(train_document)
test_list = ast.literal_eval(test_document)

def clean_data(word_list):
    for item in word_list:
        # transfer label
        if(item['label']=='neutral'):
            item['label'] = 0
        elif(item['label']=='happy'):
            item['label'] = 1
        elif(item['label']=='angry'):
            item['label'] = 2
        elif(item['label']=='sad'):
            item['label'] = 3
        elif(item['label']=='fear'):
            item['label'] = 4
        elif(item['label']=='surprise'):
            item['label'] = 5

clean_data(train_list)
clean_data(test_list)

# transform to csv 
headers = ['id', 'label', 'content']

with open('train.csv', 'w') as fp:
    writer = csv.DictWriter(fp, fieldnames=headers)
    writer.writeheader()
    for elem in train_list:
        writer.writerow(elem)
fp.close()

with open('test.csv', 'w') as fp:
    writer = csv.DictWriter(fp, fieldnames=headers)
    writer.writeheader()
    for elem in test_list:
        writer.writerow(elem)
fp.close()

train_data = pd.read_csv('train.csv')
train_data = train_data[['label', 'content']]

test_data = pd.read_csv('test.csv')
test_data = test_data[['label', 'content']]

#定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line
 

def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords 


# 加载停用词
stopwords = stopwordslist("stopwords.txt")

# 数据清理（2）
train_data['clean_content'] = train_data['content'].apply(remove_punctuation)
train_data['cut_content'] = train_data['clean_content'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))
test_data['clean_content'] = test_data['content'].apply(remove_punctuation)
test_data['cut_content'] = test_data['clean_content'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))
# print(train_data.head())

# 设置最频繁使用的6000个词
MAX_NB_WORDS = 6000
# 每条cut_review最大的长度
max_len = 60

word_list = np.append(train_data['cut_content'].values, test_data['cut_content'].values)
print(type(word_list))