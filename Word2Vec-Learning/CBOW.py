import jieba #分词
import re
import torch #pytorch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm, trange

# 初始化矩阵
torch.manual_seed(1)


# 加载停用词表
def load_stopwords():
    with open('data/stopwords.txt', 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines


# 加载文本,切词
def cut_words():
    stop_words = load_stopwords()
    with open('data/zh.txt', encoding='utf8') as f:
        allData = f.readlines()
    result = []
    for words in allData:
        c_words = jieba.lcut(words)
        result.append([word for word in c_words if word not in stop_words])
    return result


# 调用切词方法,并且去除非中英文字、数字的所有字符
data = cut_words()
# print(type(data))

# 用一个集合存储所有的词
wordList = []
for words in data:
    for word in words:
        if word not in wordList:
            wordList.append(word)
# print("wordList=", wordList)
 
raw_text = wordList
# print("raw_text=", raw_text)
 
# 超参数
learning_rate = 0.001
# 放cuda或者cpu里
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 上下文信息，即涉及文本的前n个和后n个
context_size = 2
# 词嵌入的维度，即一个单词用多少个浮点数表示比如 the=[10.2323,12.132133,4.1219774]...
embedding_dim = 100
# 训练次数
epoch = 10


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


# 把所有词集合转成dict
vocab = set(raw_text)
vocab_size = len(vocab)

# 两套字典：word2id & id2word 
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}


# cbow那个词表，即{[w1,w2,w4,w5],"label"}这样形式
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
 
# print(data[:5])

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, 128)
        self.output = nn.Linear(128, vocab_size)
 
    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        nll_prob = F.log_softmax(out, dim=-1)
        return nll_prob
 
 
# 模型在cuda训练
model = CBOW(vocab_size, embedding_dim).to(device)
# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.001)
# 存储损失的集合
losses = []
"""
    负对数似然损失函数，用于处理多分类问题，输入是对数化的概率值。
    对于包含N NN个样本的batch数据 D ( x , y ) D(x, y)D(x,y)，x xx 是神经网络的输出，
    进行了归一化和对数化处理。y yy是样本对应的类别标签，每个样本可能是C种类别中的一个。
"""
loss_function = nn.NLLLoss()
 
for epoch in trange(epoch):
    total_loss = 0
    for context, target in tqdm(data):
        # 把训练集的上下文和标签都放到GPU中
        context_vector = make_context_vector(context, word_to_idx).to(device)
        target = torch.tensor([word_to_idx[target]]).cuda()
        # print("context_vector=", context_vector)
        # 梯度清零
        model.zero_grad()
        # 开始前向传播
        train_predict = model(context_vector).cuda()  # 这里要从cuda里取出，不然报设备不一致错误
        loss = loss_function(train_predict, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
print("losses-=", losses)
