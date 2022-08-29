import re
import time
import ast
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.corpora.dictionary import Dictionary
from torch import Tensor
from sklearn.metrics import accuracy_score,classification_report
import jieba
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch
import numpy as np

def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords

stopwords = stopwordslist("./stopwords.txt")


with open('usual_train.txt') as train_file:
    train_document = train_file.read()
with open('usual_test_labeled.txt') as test_file:
    test_document = test_file.read()

train_list = ast.literal_eval(train_document)
test_list = ast.literal_eval(test_document)

def clean_data(word_list):
    for item in word_list:
        # clean data
        item['content'] = re.sub(r'\/\/\@.*?(\：|\:)', "", item['content']) # 清除@用户信息
        item['content'] = re.sub(r'\#.*?\#', "", item['content']) # 清除话题信息
        item['content'] = re.sub(r'\【.*?\】', "", item['content']) # 清除话题信息
        item['content'] = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', "", item['content'], flags=re.MULTILINE) # 清除链接信息
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

def splite_content(data):    
    string_splite = ''
    pure_data = []
    for temp in range(len(data)):
        content = data[temp]['content']
        new_content = jieba.cut(content,cut_all=False)
        str_out = ' '.join(new_content)
        cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9^\s]") # 去除非中英文字、数字的所有字符
        str_out = cop.sub('', str_out)

        for i in range(6):
            str_out = str_out.replace('  ',' ') # 去除多余空格
        str_out = str_out.strip() # 去除两边空格

        data[temp]['content'] = str_out.split(' ')
        pure_data.append([data[temp]['content'],data[temp]['label']])
        str_out += '\r\n' 
        string_splite += str_out
    return pure_data, string_splite

splite_word_all = '' # 分词总文本(包括训练文本和测试文本)
data_seg_train, out = splite_content(train_list)
splite_word_all += out 
data_seg_test, out = splite_content(test_list)
splite_word_all += out 

# 保存分好词的文本
f = open('splite_word_all.txt', 'w', encoding='utf-8')
f.write(splite_word_all)
f.close()


#模型训练，生成词向量
model_file_name = 'w2v.model'
sentences = LineSentence('splite_word_all.txt')
model = Word2Vec(sentences, vector_size=60, window=20, min_count=5, workers=4) # 参数含义：数据源，生成词向量长度，时间窗大小，最小词频数，线程数
model.save(model_file_name)

# 使用训练好的模型
model = Word2Vec.load(model_file_name)

# 创建词语字典
def create_dictionaries(p_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(p_model.wv.index_to_key, allow_update=True)
    w2indx = {v: k  for k, v in gensim_dict.items()}  # 词语的索引，从0开始编号
    id2vec = {w2indx.get(word): model.wv.__getitem__(word) for word in w2indx.keys()}  # 词语的词向量
    return w2indx, id2vec

word_id_dic, id_vect_dic= create_dictionaries(model) # 两个词典的功能：word-> id , id -> vector
# print('失望对应的id为：',word_id_dic['失望'])
# print('失望对应的词向量vector为：',id_vect_dic[word_id_dic['失望']])

# token化数据，word->id
def get_tokenized_weibo(data):
    """
    data: list of [list of word , label]
    
    """
    for word_list, label in data:
        temp = []
        for word in word_list:
            if(word in word_id_dic.keys()):
                temp.append(int(word_id_dic[word]))
            else:
                temp.append(0)
        yield [temp,label]


# 对数据进行 截断 和 填充
def preprocess_weibo(data):
    max_l = 60  # 将每条微博通过截断或者补1，使得长度变成30

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [1] * (max_l - len(x))

    features = torch.tensor([pad(content[0]) for content in data])
    labels = torch.tensor([score for _, score in data])
    return features, labels

data_train = preprocess_weibo(list(get_tokenized_weibo(data_seg_train)))
data_test = preprocess_weibo(list(get_tokenized_weibo(data_seg_test)))

# 加载数据到迭代器，并规定batch 大小
batch_size = 64
train_set = Data.TensorDataset(*data_train)  # *表示接受元组类型数组
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_set = Data.TensorDataset(*data_test)  # *表示接受元组类型数组
test_iter = Data.DataLoader(test_set, batch_size, shuffle=True)

# 定义一维互相关运算
def corr1d(X, K):
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y


# 定义全局最大池化层（pytorch没有）
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
         # x shape: (batch_size, channel, seq_len)
         # return shape: (batch_size, channel, 1)
        return F.max_pool1d(x, kernel_size=x.shape[2])


class TextCNN(nn.Module):
    def __init__(self, vocab_num, embed_size, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_num, embed_size)
        # 不参与训练的嵌入层
        self.constant_embedding = nn.Embedding(vocab_num, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 6)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = 2*embed_size, 
                                        out_channels = c, 
                                        kernel_size = k))

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = torch.cat((
            self.embedding(inputs), 
            self.constant_embedding(inputs)), dim=2) # (batch, seq_len, 2*embed_size)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维(即词向量那一维)，变换到前一维
        embeddings = embeddings.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


vocab_num = len(model.wv.index_to_key)
embed_size, kernel_sizes, nums_channels = 60, [3, 4, 5], [100, 100, 100]
net = TextCNN(vocab_num, embed_size, kernel_sizes, nums_channels)

# 将训练好的词向量输入embedding
id_vect = torch.Tensor(np.array(list(id_vect_dic.values())))
net.embedding.weight.data.copy_(id_vect)
net.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它

# 设置训练参数
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()

evaluate_loss = []
train_loss = []

# 训练函数
def train(train_iter, model, loss, optimizer):
    all_label = torch.tensor([])
    all_pre = torch.tensor([])
    epoch_loss = 0.0
    batch_count = 0
    model.train()  # 设置模型为训练模式
    for step, batch in enumerate(train_iter):
        X_train, y_train = batch[0], batch[1]
        out = model(X_train)
        pre_lab = torch.argmax(out, 1)  # 预测的标签
        all_label = torch.cat((all_label, y_train), dim=0)
        all_pre = torch.cat((all_pre, pre_lab), dim=0)

        # 计算损失函数值
        l = loss(out, y_train)  

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        l.backward()

        # 梯度调整
        optimizer.step()

        epoch_loss += l.item()
        batch_count += 1

    train_loss.append(epoch_loss)
    print('train')
    print(classification_report(Tensor.cpu(all_label), Tensor.cpu(all_pre), target_names=['0', '1', '2', '3', '4', '5'], digits=4))


# 评价函数
def evaluate(data_iter, model, loss):
    model.eval()
    all_label = torch.tensor([]) # 目标标签
    all_pre = torch.tensor([]) # 预测标签
    test_loss = 0.0
    for step, batch in enumerate(data_iter):
        X_test, y_test = batch[0], batch[1]
        out = model(X_test)
        l = loss(out, y_test)
        test_loss += l.item() * len(y_test)
        pre_lab = torch.argmax(out, 1)
        all_label = torch.cat((all_label, y_test), dim=0)
        all_pre = torch.cat((all_pre, pre_lab), dim=0)
    evaluate_loss.append(test_loss)
    print('test')
    print(classification_report(Tensor.cpu(all_label), Tensor.cpu(all_pre), target_names=['0', '1', '2', '3', '4', '5'], digits=4))

# 训练轮数
print("training...")
for i in range(num_epochs):
    print('epoch:', i + 1)
    train(train_iter, net, loss, optimizer)
    evaluate(test_iter, net, loss)

