import torch
import pandas as pd
import os
import ast
import csv
import re
import logging
from transformers import logging

logging.set_verbosity_warning()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


with open('Datasets/train/usual_train.txt') as train_file:
    train_document = train_file.read()
with open('Datasets/test/real/usual_test_labeled.txt') as train_file:
    test_document = train_file.read()

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

'''
moods = {
    0: 'neutral',
    1: 'happy',
    2: 'angry', 
    3: 'sad', 
    4: 'fear', 
    5: 'surprise'
}

for label, mood in moods.items():
    print('文本数量（{}）：{}'.format(mood,  train_data[train_data.label==label].shape[0]))
'''

train_contents = train_data.content.values
train_labels = train_data.label.values

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

'''
print(' 原句: ', train_contents[0])
print('Tokenizen 后的句子: ', tokenizer.tokenize(train_contents[0]))
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_contents[0])))
'''

input_ids = []
attention_masks = []

for content in train_contents:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        str(content), # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 256,           # Pad & truncate all sentences.
                        padding = 'max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # 把编码的句子加入list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # 加上 attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])
    

# 把lists 转为 tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(train_labels)

'''
print('原句: ', train_contents[0])
print('Token IDs:', input_ids[0])
print('attention_masks:', attention_masks[0])
'''

from torch.utils.data import TensorDataset, random_split

# 把input 放入 TensorDataset。
dataset = TensorDataset(input_ids, attention_masks, labels)

# 计算 train_size 和 val_size 的长度.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# 90% 的dataset 为train_dataset, 10% 的的dataset 为val_dataset.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# 推荐batch_size 为 16 或者 32
batch_size = 32

# 为训练数据集和验证数据集设计DataLoaders. 
train_dataloader = DataLoader(
            train_dataset,  # 训练数据.
            sampler = RandomSampler(train_dataset), # 打乱顺序
            batch_size = batch_size 
        )

validation_dataloader = DataLoader(
            val_dataset, # 验证数据.
            sampler = RandomSampler(val_dataset), # 打乱顺序
            batch_size = batch_size 
        )


from transformers import BertForSequenceClassification, BertConfig
from torch.optim import AdamW

model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese", # 使用 12-layer 的 BERT 模型.
    num_labels = 6, # 多分类任务的输出标签为 4个.                     
    output_attentions = False, # 不返回 attentions weights.
    output_hidden_states = False, # 不返回 all hidden-states.
)

# 选择优化器
# AdamW 是一个 huggingface library 的类，'W' 是'Weight Decay fix"的意思。
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - 默认是 5e-5
                  eps = 1e-8 # args.adam_epsilon  - 默认是 1e-8， 是为了防止衰减率分母除到0
                )


from transformers import get_linear_schedule_with_warmup

# bert 推荐 epochs 在2到4之间为好。
epochs = 4

# training steps 的数量: [number of batches] x [number of epochs]. 
total_steps = len(train_dataloader) * epochs

# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

import numpy as np
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime
def format_time(elapsed):    
    elapsed_rounded = int(round((elapsed)))    
    # 返回 hh:mm:ss 形式的时间
    return str(datetime.timedelta(seconds=elapsed_rounded))


import os
import random
import numpy as np
from transformers import WEIGHTS_NAME, CONFIG_NAME

output_dir = "./models/"
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
# 代码参考 https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# 设置随机种子.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 记录training ,validation loss ,validation accuracy and timings.
training_stats = []

# 设置总时间.
total_t0 = time.time()
best_val_accuracy = 0

for epoch_i in range(0, epochs):      
    print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))  

    # 记录每个 epoch 所用的时间
    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()
  
    for step, batch in enumerate(train_dataloader):

        # 每隔40个batch 输出一下所用时间.
        if step % 40 == 0 and not step == 0:            
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # `batch` 包括3个 tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # 清空梯度
        model.zero_grad()        

        # forward        
        # 参考 https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
       
        total_train_loss += loss.item()

        # backward 更新 gradients.
        loss.backward()

        # 减去大于1 的梯度，将其设为 1.0, 以防梯度爆炸.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新模型参数 
        optimizer.step()
       
        # 更新 learning rate.
        scheduler.step()        
             
        logit = logits.detach().cpu().numpy()
        label_id = b_labels.to('cpu').numpy()
        # 计算training 句子的准确度.
        total_train_accuracy += flat_accuracy(logit, label_id)    
     
    # 计算batches的平均损失.
    avg_train_loss = total_train_loss / len(train_dataloader)      
    # 计算训练时间.
    training_time = format_time(time.time() - t0)
    
    # 训练集的准确率.
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    print("  训练准确率: {0:.2f}".format(avg_train_accuracy))
    print("  平均训练损失 loss: {0:.2f}".format(avg_train_loss))
    print("  训练时间: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================

    t0 = time.time()

    # 设置 model 为valuation 状态，在valuation状态 dropout layers 的dropout rate会不同
    model.eval()

    # 设置参数
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:        

        # `batch` 包括3个 tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)        

        # 在valuation 状态，不更新权值，不改变计算图
        with torch.no_grad():        

            # 参考 https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        # 计算 validation loss.
        total_eval_loss += loss.item()        
        logit = logits.detach().cpu().numpy()
        label_id = b_labels.to('cpu').numpy()

        # 计算 validation 句子的准确度.
        total_eval_accuracy += flat_accuracy(logit, label_id)
        
    # 计算 validation 的准确率.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("")
    print("  测试准确率: {0:.2f}".format(avg_val_accuracy))
    
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(),output_model_file)
        model.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_dir)
         

    # 计算batches的平均损失.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # 计算validation 时间.
    validation_time = format_time(time.time() - t0)
    
    print("  平均测试损失 Loss: {0:.2f}".format(avg_val_loss))
    print("  测试时间: {:}".format(validation_time))

    # 记录模型参数
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("训练一共用了 {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

