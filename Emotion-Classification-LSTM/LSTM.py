import re
import ast
import zhconv
import jieba
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, LSTM
from keras.callbacks import EarlyStopping
import csv

with open('Datasets/train/usual_train.txt') as train_file:
    train_document = train_file.read()
with open('Datasets/test/real/usual_test_labeled.txt') as test_file:
    test_document = test_file.read()

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

# transform to csv 
headers = ['id', 'label', 'content']

with open('train.csv', 'w') as fp:
    writer = csv.DictWriter(fp, fieldnames=headers)
    writer.writeheader()
    for elem in train_list:
        writer.writerow(elem)
fp.close()

train_data = pd.read_csv('train.csv')
train_data = train_data[['label', 'content']]

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
 
#加载停用词
stopwords = stopwordslist("stopwords.txt")
train_data['clean_content'] = train_data['content'].apply(remove_punctuation)
train_data['cut_content'] = train_data['clean_content'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))
# print(train_data.head())

# 设置最频繁使用的50000个词
MAX_NB_WORDS = 50000
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 250
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 100
 
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(train_data['cut_content'].values)
word_index = tokenizer.word_index
# print('共有 %s 个不相同的词语.' % len(word_index))

X = tokenizer.texts_to_sequences(train_data['cut_content'].values)
#填充X,让X的各个列的长度统一
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
 
#多类标签的onehot展开
Y = pd.get_dummies(train_data['label']).values
 
# print(X.shape)
# print(Y.shape)

#拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
# print(X_train.shape,Y_train.shape)
# print(X_test.shape,Y_test.shape)

#定义模型
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

epochs = 5
batch_size = 64
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
