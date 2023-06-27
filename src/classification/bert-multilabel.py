"""






This file is deprecated!!!
This file is deprecated!!!
This file is deprecated!!!
This file is deprecated!!!
This file is deprecated!!!
This file is deprecated!!!
This file is deprecated!!!







"""



import time
import datetime

import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (BertForSequenceClassification,
                          BertForTokenClassification, AdamW, BertConfig)
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from src.loadfiles import load_pickle
from src.evaluate import Evaluator
from src.classification.multilabel import convert_to_one_hot, convert_to_label


num_labels = 18
name = "guanzhi_sep_vec_each_1350_select"
df_train = load_pickle(fr'C:\Users\10507\OneDrive\桌面\202301 危险源挖掘论文\不安全事件分类'
                       fr'\code\out\datasets\{name}-train.pkl')
df_test = load_pickle(fr'C:\Users\10507\OneDrive\桌面\202301 危险源挖掘论文\不安全事件分类'
                      fr'\code\out\datasets\{name}-test.pkl')

sentences_train = df_train['后果'].to_list()
sentences_test = df_test['后果'].to_list()

# labels_train = (df_train['label'] - 1).tolist()
# labels_test = (df_test['label'] - 1).tolist()
labels_train = df_train[['label', 'label2', 'label3', 'label4', 'label5']].fillna(0).to_numpy().astype(int)
labels_train = convert_to_one_hot(labels_train)
labels_test = convert_to_one_hot(df_test[['label', 'label2', 'label3', 'label4', 'label5']].fillna(0).to_numpy().astype(int))


# 加载 BERT 分词器
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
# model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')

# 或加载保存的分词器
# tokenizer = BertTokenizer.from_pretrained("model_save")
# model = BertModel.from_pretrained("model_save")


# 确定 MAX_LEN 这个参数
max_len = 0
for sent in sentences_train:
    # 将文本分词，并添加 `[CLS]` 和 `[SEP]` 符号
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))
for sent in sentences_test:
    # 将文本分词，并添加 `[CLS]` 和 `[SEP]` 符号
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)


# 将数据集分完词后存储到列表中
input_ids_train = []
attention_masks_train = []

# 分词
for sent in sentences_train:
    encoded_dict = tokenizer.encode_plus(
        sent,  # 输入文本
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        max_length=64,  # 填充 & 截断长度
        pad_to_max_length=True,
        return_attention_mask=True,  # 返回 attn. masks.
        return_tensors='pt',  # 返回 pytorch tensors 格式的数据
    )

    # 将编码后的文本加入到列表
    input_ids_train.append(encoded_dict['input_ids'])

    # 将文本的 attention mask 也加入到 attention_masks 列表
    attention_masks_train.append(encoded_dict['attention_mask'])

# 将列表转换为 tensor
input_ids_train = torch.cat(input_ids_train, dim=0)
attention_masks_train = torch.cat(attention_masks_train, dim=0)
labels_train = torch.tensor(labels_train)

# 输出第 1 行文本的原始和编码后的信息
print('Original: ', sentences_train[0])
print('Token IDs:', input_ids_train[0])


# 拆分训练集和验证集
# 将输入数据合并为 TensorDataset 对象
dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)

# 计算训练集和验证集大小
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# 按照数据大小随机拆分训练集和测试集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))


# 我们使用 DataLoader 类来读取数据集，相对于一般的 for 循环来说，这种方法在训练期间会比较节省内存：
# 在 fine-tune 的训练中，BERT 作者建议小批量大小设为 16 或 32
batch_size = 16

# 为训练和验证集创建 Dataloader，对训练样本随机洗牌
train_dataloader = DataLoader(
            train_dataset,  # 训练样本
            sampler=RandomSampler(train_dataset),  # 随机小批量
            batch_size=batch_size  # 以小批量进行训练
        )

# 验证集不需要随机化，这里顺序读取就好
validation_dataloader = DataLoader(
            val_dataset,  # 验证样本
            sampler=SequentialSampler(val_dataset),  # 顺序选取小批量
            batch_size=batch_size
        )


# 加载 BertForSequenceClassification, 预训练 BERT 模型 + 顶层的线性分类层
model = BertForSequenceClassification.from_pretrained(
    "hfl/chinese-roberta-wwm-ext", # 小写的 12 层预训练模型
    num_labels=num_labels,  # 分类数 --2 表示二分类
                    # 你可以改变这个数字，用于多分类任务
    problem_type="multi_label_classification",  # 问题类型为多标签分类
    output_attentions=False,  # 模型是否返回 attentions weights.
    output_hidden_states=False,  # 模型是否返回所有隐层状态.
    return_dict=False
)


# 优化器 & 学习率调度器
# 我认为 'W' 代表 '权重衰减修复"
optimizer = AdamW(model.parameters(),
                  lr=2e-5, # args.learning_rate - default is 5e-5
                  eps=1e-8 # args.adam_epsilon  - default is 1e-8
                )

from transformers import get_linear_schedule_with_warmup

# 训练 epochs。 BERT 作者建议在 2 和 4 之间，设大了容易过拟合
epochs = 2

# 总的训练样本数
total_steps = len(train_dataloader) * epochs

# 创建学习率调度器
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)


# 4.3. 训练循环
# 根据预测结果和标签数据来计算准确率
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))

    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# 全部训练代码
# 以下训练代码是基于 `run_glue.py` 脚本:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# 设定随机种子值，以确保输出是确定的
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)

# 存储训练和评估的 loss、准确率、训练时长等统计指标,
training_stats = []

# 统计整个训练时长
total_t0 = time.time()

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # 统计单次 epoch 的训练时间
    t0 = time.time()

    # 重置每次 epoch 的训练总 loss
    total_train_loss = 0

    # 将模型设置为训练模式。这里并不是调用训练接口的意思
    # dropout、batchnorm 层在训练和测试模式下的表现是不同的 (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # 训练集小批量迭代
    for step, batch in enumerate(train_dataloader):

        # 每经过40次迭代，就输出进度信息
        if step % 1 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 准备输入数据，并将其拷贝到 gpu 中
        # b_input_ids = batch[0].to(device)
        # b_input_mask = batch[1].to(device)
        # b_labels = batch[2].to(device)
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2].type(torch.FloatTensor)

        # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
        model.zero_grad()

        # 官网改的
        # inputs = {
        #     'input_ids': b_input_ids,
        #     'attention_mask': b_input_mask,
        #     'labels': b_labels
        # }
        # with torch.no_grad():
        #     logits = model(**inputs)[0]
        #     # logits = model(**inputs).logits
        # predicted_class_ids = torch.arange(0, logits.shape[-1]).repeat(batch_size, 1)[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
        #
        # labels = torch.sum(
        #     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
        # ).to(torch.float)
        # loss = model(**inputs, labels=labels).loss


        # 前向传播
        # 文档参见:
        # https://huggingface.co/docs/transformers/model_doc/bert
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # 该函数会根据不同的参数，会返回不同的值。 本例中, 会返回 loss 和 logits -- 模型的预测结果
        loss, logits = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

        # 累加 loss
        total_train_loss += loss.item()

        # 反向传播
        loss.backward()

        # 梯度裁剪，避免出现梯度爆炸情况
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新参数
        optimizer.step()

        # 更新学习率
        scheduler.step()

    # 平均训练误差
    avg_train_loss = total_train_loss / len(train_dataloader)

    # 单次 epoch 的训练时长
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # 完成一次 epoch 训练后，就对该模型的性能进行验证

    print("")
    print("Running Validation...")

    t0 = time.time()

    # 设置模型为评估模式
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # 将输入数据加载到 gpu 中
        # b_input_ids = batch[0].to(device)
        # b_input_mask = batch[1].to(device)
        # b_labels = batch[2].to(device)
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2].type(torch.FloatTensor)

        # 评估的时候不需要更新参数、计算梯度
        with torch.no_grad():
            (loss, logits) = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

        # 累加 loss
        total_eval_loss += loss.item()

        # 将预测结果和 labels 加载到 cpu 中计算
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 计算准确率
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # 打印本次 epoch 的准确率
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # 统计本次 epoch 的 loss
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # 统计本次评估的时长
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # 记录本次 epoch 的所有统计信息
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

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


# 看一下整个训练的概要：
# 保留 2 位小数
pd.set_option('precision', 2)
# 加载训练统计到 DataFrame 中
df_stats = pd.DataFrame(training_stats)
# 使用 epoch 值作为每行的索引
# df_stats = df_stats.set_index('epoch')
# 展示表格数据
print(df_stats)


# 存储模型
import os

# 模型存储到的路径
output_dir = './model_save/bert-1350'

# 目录不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# 使用 `save_pretrained()` 来保存已训练的模型，模型配置和分词器
# 它们后续可以通过 `from_pretrained()` 加载
model_to_save = model.module if hasattr(model, 'module') else model  # 考虑到分布式/并行（distributed/parallel）训练
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))


# 评估结果
# 预测测试集
# 创建测试集
input_ids_test = []
attention_masks_test = []

# 分词
for sent in sentences_test:
    encoded_dict = tokenizer.encode_plus(
        sent,  # 输入文本
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        max_length=64,  # 填充 & 截断长度
        pad_to_max_length=True,
        return_attention_mask=True,  # 返回 attn. masks.
        return_tensors='pt',  # 返回 pytorch tensors 格式的数据
    )

    # 将编码后的文本加入到列表
    input_ids_test.append(encoded_dict['input_ids'])

    # 将文本的 attention mask 也加入到 attention_masks 列表
    attention_masks_test.append(encoded_dict['attention_mask'])

# 将列表转换为 tensor
input_ids_test = torch.cat(input_ids_test, dim=0)
attention_masks_test = torch.cat(attention_masks_test, dim=0)
labels_test = torch.tensor(labels_test).type(torch.FloatTensor)

# 输出第 1 行文本的原始和编码后的信息
print('Original: ', sentences_test[0])
print('Token IDs:', input_ids_test[0])

# 将输入数据合并为 TensorDataset 对象
test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)

test_dataloader = DataLoader(
            test_dataset,  # 验证样本
            sampler=SequentialSampler(test_dataset),  # 顺序选取小批量
            batch_size=batch_size
        )
print('Predicting labels for {:,} test sentences...'.format(len(input_ids_test)))
prediction_dataloader = test_dataloader
# 依然是评估模式
model.eval()

# Tracking variables
predictions, true_labels = [], []

# 预测
for batch in prediction_dataloader:
    # 将数据加载到 gpu 中
    # batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    # 不需要计算梯度
    with torch.no_grad():
        # 前向传播，获取预测结果
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]

    # 将结果加载到 cpu 中
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # 存储预测结果和 labels
    predictions.append(logits)
    true_labels.append(label_ids)

print('    DONE.')
# print('Positive samples: %d of %d (%.2f%%)' % (df.label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0)))


# # 最终评测结果会基于全量的测试数据，不过我们可以统计每个小批量各自的分数，以查看批量之间的变化。
# from sklearn.metrics import matthews_corrcoef
#
# matthews_set = []
#
# # 计算每个 batch 的 MCC
# print('Calculating Matthews Corr. Coef. for each batch...')
#
# # For each input batch...
# for i in range(len(true_labels)):
#     pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
#
#     # 计算该 batch 的 MCC
#     matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
#     matthews_set.append(matthews)
#
# # 创建柱状图来显示每个 batch 的 MCC 分数
# ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)
#
# plt.title('MCC Score per Batch')
# plt.ylabel('MCC Score (-1 to +1)')
# plt.xlabel('Batch #')
#
# plt.show()
#
#
# # 我们将所有批量的结果合并，来计算最终的 MCC 分：
# # 合并所有 batch 的预测结果
# flat_predictions = np.concatenate(predictions, axis=0)  # 可用于计算auc
#
# # 取每个样本的最大值作为预测值
# flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
#
# # 合并所有的 labels
# flat_true_labels = np.concatenate(true_labels, axis=0)
#
# # 计算 MCC
# mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
#
# print('Total MCC: %.3f' % mcc)


