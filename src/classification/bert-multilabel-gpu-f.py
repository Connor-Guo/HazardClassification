import time
import datetime
import os

import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torcheval.metrics.functional.classification.accuracy import topk_multilabel_accuracy
from transformers import (BertForSequenceClassification,
                          BertForTokenClassification, AdamW, BertConfig)
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from src.loadfiles import load_pickle
from src.evaluate import MultiLabelProbEvaluator
from src.classification.multilabel import convert_to_one_hot, convert_to_label


def _get_max_len(
    tokenizer,
    sentences_train,
    sentences_test,
) -> int:
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
    return max_len


def _encode_dataset(
    tokenizer,
    sentences_train: list,
    labels_train,
):
    # 将数据集分完词后存储到列表中
    input_ids_train = []
    attention_masks_train = []

    # 分词
    for sent in sentences_train:
        encoded_dict = tokenizer.encode_plus(
            sent,  # 输入文本
            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
            max_length=128,  # 填充 & 截断长度
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
    # print('Original: ', sentences_train[0])
    # print('Token IDs:', input_ids_train[0])
    return input_ids_train, attention_masks_train, labels_train


def _train_valid_split(
    dataset,
):
    # 拆分训练集和验证集
    # 计算训练集和验证集大小
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # 按照数据大小随机拆分训练集和测试集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    return train_dataset, val_dataset


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


def find_best_threshold(probs: torch.Tensor, label_ids: torch.Tensor):
    """计算最优阈值"""
    thresholds = np.arange(0.05, 0.25, 0.01)  # 设置不同的阈值
    best_threshold = 0.5  # 初始化最优阈值
    best_f1 = 0.0  # 初始化最优F1-score

    for threshold in thresholds:
        pred = (probs > threshold).int()
        f1 = f1_score(label_ids, pred, average='samples')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def get_prediction_with_threshold(probs: torch.Tensor, threshold: float):
    preds = (probs > threshold).int().detach().numpy()
    return preds


def train_bertC(
    model,
    train_dataloader,
    validation_dataloader,
    epochs=3,
    seed_val=42,
):
    # 全部训练代码
    # 以下训练代码是基于 `run_glue.py` 脚本:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # 设定随机种子值，以确保输出是确定的

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

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
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].type(torch.FloatTensor).to(device)
            # b_input_ids = batch[0]
            # b_input_mask = batch[1]
            # b_labels = batch[2].type(torch.FloatTensor)

            # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
            model.zero_grad()

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

        # 模型验证
        # 设置模型为评估模式
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        lst_logits = []
        lst_label_ids = []

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # 将输入数据加载到 gpu 中
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].type(torch.FloatTensor).to(device)
            # b_input_ids = batch[0]
            # b_input_mask = batch[1]
            # b_labels = batch[2].type(torch.FloatTensor)

            # 评估的时候不需要更新参数、计算梯度
            with torch.no_grad():
                loss, logits = model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)

            # 累加 loss
            total_eval_loss += loss.item()

            # 将预测结果和 labels 加载到 cpu 中计算
            logits = logits.detach().cpu()
            label_ids = b_labels.to('cpu')

            # 记录每个batch的logits和label_ids
            lst_logits.append(logits)
            lst_label_ids.append(label_ids)

            # 计算准确率
            # total_eval_accuracy += flat_accuracy(logits, label_ids)

        # 合并当前epoch的数据
        epoch_logits = torch.concat(lst_logits)
        epoch_label_ids = torch.concat(lst_label_ids)

        # 计算最优阈值和精度
        epoch_probs = torch.softmax(epoch_logits, dim=1)
        best_threshold = find_best_threshold(epoch_probs, epoch_label_ids)
        epoch_preds = get_prediction_with_threshold(epoch_probs, best_threshold)
        val_accuracy = accuracy_score(epoch_label_ids, epoch_preds)

        # 打印本次 epoch 的准确率
        # avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(val_accuracy))

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
                'Valid. Accur.': val_accuracy,
                'Best T': best_threshold,
                # 'Training Time': training_time,
                # 'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    return model, training_stats


def plot_training_stats(training_stats) -> None:
    # 看一下整个训练的概要：
    # 保留 2 位小数
    # pd.set_option('precision', 2)
    # 加载训练统计到 DataFrame 中
    df_stats = pd.DataFrame(training_stats)
    # 使用 epoch 值作为每行的索引
    df_stats = df_stats.set_index('epoch')
    # 展示表格数据
    print(df_stats)


def predict_bertC(
    model,
    test_dataloader,
):
    # 依然是评估模式
    model.eval()

    # Tracking variables
    probabillities, true_labels = [], []

    # 预测
    for batch in test_dataloader:
        # 将数据加载到 gpu 中
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # 不需要计算梯度
        with torch.no_grad():
            # 前向传播，获取预测结果
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # 将结果加载到 cpu 中
        logits = logits.detach().cpu()
        label_ids = b_labels.to('cpu')

        # 计算预测概率
        probs = torch.softmax(logits, dim=1)

        # 存储预测结果和 labels
        probabillities.append(probs)
        true_labels.append(label_ids)

    print('    DONE.')
    return probabillities, true_labels


# 最终评估
# def evaluate(predictions, true_labels, top_k=5):
#     # 如有需要调整k的值，可以照此思路
#     # 先根据true_labels将结果按标签个数分组
#     # 设标签个数为n_labels，那么top-k accuracy的k可以取n_labels + 1
#     # 输入的是一个包含多个batch数据的列表，这里将其合并为一个矩阵
#
#     # flat_predictions = torch.softmax(flat_predictions, dim=1)
#     # 计算模型性能
#     perf = {
#         "contain": topk_multilabel_accuracy(predictions, true_labels, criteria='contain', k=top_k),
#         "overlap": topk_multilabel_accuracy(predictions, true_labels, criteria='overlap', k=top_k),
#         "hamming": topk_multilabel_accuracy(predictions, true_labels, criteria='hamming', k=top_k),
#     }
#     return perf


if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练参数
    # 在 fine-tune 的训练中，BERT 作者建议小批量大小设为 16 或 32
    batch_size = 16
    # 训练 epochs。 BERT 作者建议在 2 和 4 之间，设大了容易过拟合
    epochs = 4

    # 读取数据集
    num_labels = 21
    DATASETS_PATH = r"C:\Users\envss\gzy\HazardClassification\code\out\datasets"
    EXP_NAME = "guanzhi_fix_bert_abbr_2941_GPT"
    dataset_path = os.path.join(DATASETS_PATH, EXP_NAME)
    # df_train = load_pickle(os.path.join(dataset_path, f"{EXP_NAME}-train.pkl"))
    # df_test = load_pickle(os.path.join(dataset_path, f"{EXP_NAME}-test.pkl"))
    df_train = pd.read_excel(os.path.join(dataset_path, f"{EXP_NAME}-train.xlsx"))
    df_test = pd.read_excel(os.path.join(dataset_path, f"{EXP_NAME}-test.xlsx"))
    # 输出路径
    model_name = EXP_NAME + "_BertC"
    output_dir = model_name  # 模型输出路径

    # 解析数据集
    sentences_train = df_train['后果'].to_list()
    sentences_test = df_test['后果'].to_list()
    # labels_train = (df_train['label'] - 1).tolist()
    # labels_test = (df_test['label'] - 1).tolist()
    labels_train = convert_to_one_hot(
        df_train[['label', 'label2', 'label3', 'label4', 'label5']].fillna(0).to_numpy().astype(int))
    labels_test = convert_to_one_hot(
        df_test[['label', 'label2', 'label3', 'label4', 'label5']].fillna(0).to_numpy().astype(int))

    # 加载 BERT 分词器
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    # model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')

    # 或加载保存的分词器
    # tokenizer = BertTokenizer.from_pretrained("model_save")
    # model = BertModel.from_pretrained("model_save")

    # 计算最长的句子长度
    max_len = _get_max_len(tokenizer, sentences_train, sentences_test)
    input_ids_train, attention_masks_train, labels_train = _encode_dataset(tokenizer, sentences_train, labels_train)
    # 将输入数据合并为 TensorDataset 对象
    dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    train_dataset, val_dataset = _train_valid_split(dataset)

    # 我们使用 DataLoader 类来读取数据集，相对于一般的 for 循环来说，这种方法在训练期间会比较节省内存：

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
        "hfl/chinese-roberta-wwm-ext",  # 小写的 12 层预训练模型
        num_labels=num_labels,  # 分类数 --2 表示二分类
        # 你可以改变这个数字，用于多分类任务
        problem_type="multi_label_classification",  # 问题类型为多标签分类
        output_attentions=False,  # 模型是否返回 attentions weights.
        output_hidden_states=False,  # 模型是否返回所有隐层状态.
        return_dict=False
    )

    # 在 gpu 中运行该模型
    model.cuda()
    # 优化器 & 学习率调度器
    # 我认为 'W' 代表 '权重衰减修复"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8
                      )

    # 总的训练样本数
    total_steps = len(train_dataloader) * epochs
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # 训练分类器
    model, training_stats = train_bertC(model, train_dataloader, validation_dataloader, epochs=epochs, seed_val=42)

    # 查看训练与验证的loss曲线
    plot_training_stats(training_stats)

    # 存储模型
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

    # 模型测试
    input_ids_test, attention_masks_test, labels_test = _encode_dataset(tokenizer, sentences_test, labels_test)
    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    test_dataloader = DataLoader(
        test_dataset,  # 验证样本
        sampler=SequentialSampler(test_dataset),  # 顺序选取小批量
        batch_size=batch_size,
    )
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids_test)))

    # 获取结果
    probs, true_labels = predict_bertC(model, test_dataloader)
    flat_probs = torch.tensor(np.concatenate(probs, axis=0))
    # flat_predictions = torch.softmax(flat_predictions, dim=1)  # 得到整个测试集的预测矩阵（是Prob）
    flat_true_labels = torch.tensor(np.concatenate(true_labels, axis=0))
    threshold = find_best_threshold(flat_probs, flat_true_labels)

    eva = MultiLabelProbEvaluator(flat_true_labels, flat_probs, threshold=threshold)
    acc = eva.accuracy()
    top_k_acc = eva.top_k_accuracy(k=5)
    print("Best threshold: {:.2f}".format(threshold))
    print("Model accuracy: {}".format(acc))
    print("Model top-k accuracy: {}".format(top_k_acc))

    # 得到预测结果
    top_p, top_class = torch.topk(flat_probs, 5, dim=1)
    top_class = top_class + 1
    pred = get_prediction_with_threshold(flat_probs, threshold)
    np.savetxt('./pred_class.csv', pred, fmt="%d", delimiter=",")
    np.savetxt('./top_class.csv', top_class.detach().numpy(), fmt="%d", delimiter=",")
    np.savetxt('./top_p.csv', top_p.detach().numpy(), fmt="%.4f", delimiter=",")

