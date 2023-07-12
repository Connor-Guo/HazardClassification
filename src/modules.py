"""
存放一些独立的功能模块

除loadfiles外，不要从src引入其他模块。

模块1：统计不安全事件统计表中有各标签的数量。
"""
import warnings

import pandas as pd
import numpy as np
import jieba
from src.loadfiles import load_pickle, save_pickle


def init_tokenizer(*args):
    """args第一个是自定义词典，第二个是缩写词典"""
    tk = jieba.Tokenizer()
    tk.load_userdict(r'./nlp/atc_terms.txt')  # 读取自定义词典


def count_labels(df: pd.DataFrame, save=False):
    """统计（多标签）不安全事件统计表中各标签的数量。"""
    # df = load_pickle("out/dump/guanzhi_bert_2941.pkl")
    df_v = df.value_counts('label')
    for i in range(2, 6):
        t = df.value_counts(f'label{i}')
        df_v = df_v.add(t, fill_value=0).astype(int)
    if save:
        df_v.to_csv('./out/dump/样本统计.csv')

    return df_v


def _1col_300col(df):
    """把原本1列的doc_vec转成300列，输出一个300列的df"""
    data = df.loc[:, 'doc_vec'].to_list()
    d = np.array(data)
    c = list(range(300))  # column names
    df_300 = pd.DataFrame(data=d, columns=c)
    return df_300


def _df_to_ag(df):
    data = df.loc[:, 'doc_vec'].to_list()
    d = np.array(data)
    to_insert = df.loc[:, 'label'].to_numpy().reshape(-1, 1)  # labels
    out = np.insert(d, 300, to_insert, axis=1)
    c = list(range(300))  # column names
    c.append('label')
    df_new = pd.DataFrame(data=out, columns=c)
    df_new.to_csv('./out/230207_td_ag_300d_1.xlsx.csv', index=False)


def _conda_ag():
    from autogluon.tabular import TabularDataset, TabularPredictor
    train_data = TabularDataset(r'C:\Users\10507\OneDrive\桌面\202301 危险源挖掘论文\不安全事件分类\code\out\for_ag.csv')
    # test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    predictor = TabularPredictor(label='label').fit(train_data=train_data)
    # score = predictor.evaluate(test_data)


def regularize_data():
    """
    *Run in console*

    * 仅用于a=1情况

    把数据改写成一一对应的形式，输出到新的df里。
    只保留（描述文本的项数）/（不安全事件的类数）均为1的数据
    再剔除“其他类型的数据”

    * 输出的标签范围为 1-18(19)
    """
    import os
    import pandas as pd
    import numpy as np
    import jieba
    import src.preprocessing as pre
    from src.loadfiles import (Loader,
                               load_data,
                               load_pretrained,
                               )


    df = load_data("./data/1-管制专业不安全事件匹配结果.xlsx")  # 读取数据库
    # 创建停用词列表、乱码词列表
    stopwords = []
    gibberish = Loader().load_gibberish('./nlp/gibberish.txt')
    # 读取word2vec词典
    keys = Loader().load_keys("./nlp/word2vec/keys.txt")
    # 读取缩写词典
    abbr = {}  # dict
    # 初始化分词器
    tk = jieba.Tokenizer()
    tk.load_userdict(r'./nlp/atc_terms.txt')  # 读取自定义词典
    df = pre.preprocess_df(df,
                           tk=tk,
                           stopwords=stopwords,
                           gibberish=gibberish,
                           word2vec_keys=keys,
                           abbr_dict=abbr,
                           keep_english=True,
                           )

    # 填写到新的df里
    df_new = pd.DataFrame(columns=["危险源编号", "后果原文", "后果", "不安全事件", "不安全事件2", "不安全事件3",
                                   "不安全事件4", "不安全事件5", "label", "label2", "label3", "label4", "label5"])

    # 读取不安全事件标签
    label_dict = {}
    for l in [line.strip() for line in open("./data/label-guanzhi.txt", encoding='UTF-8').readlines()]:
        v, k = l.split('\t')  # 1	跑道侵入
        label_dict[k] = int(v)

    new_idx = 0
    repeat_end = 0
    for i in range(len(df)):  # 对每行进行遍历
        if i < repeat_end:  # 跳过重复项
            continue
        a = 1  # a表示后果文本的点数，b表示后果文本对应的标签数

        # 计算b
        b = 5 - sum(df.iloc[i, 2:7].isna())
        # if b != 1:
        #     continue

        for ii in range(i + 1, len(df)):
            # 判断a是否大于1
            if df.iloc[i, 1] == df.iloc[ii, 1]:
                a += 1  # 如果这条重复了，a += 1
                continue
            else:
                # 如果这条没重复，两种情况，a=1 or a>1
                if a == 1:
                    line = df.loc[i, ["危险源编号", "后果"]].tolist()

                    t = ''
                    for _ in df.loc[i, "text"]:
                        t += _
                    line.append(t)

                    # 添加不安全事件文本和标签
                    # line.append(df.loc[i, "不安全事件"])
                    # line.append(label_dict[df.loc[i, "不安全事件"]])
                    for col_name in df.columns[2:7]:  # "不安全事件，不安全事件2，……不安全事件5
                        line.append(df.loc[i, col_name])
                    for col_name in df.columns[2:7]:
                        if not pd.isna(df.loc[i, col_name]):
                            line.append(label_dict[df.loc[i, col_name]])
                        else:
                            line.append(np.nan)

                    df_new.loc[new_idx] = line  # 写入新的df
                    new_idx += 1
                else:  # a>1
                    repeat_end = ii
                break
    # 输出
    df_new.drop(df_new.index[(df_new["label"] == 19) |
                             (df_new["label2"] == 19) |
                             (df_new["label3"] == 19) |
                             (df_new["label4"] == 19) |
                             (df_new["label5"] == 19)]).to_excel("./data/管制-一比多-已剔除其他标签.xlsx", index=False)
    # df_new.to_excel("./data/管制-一比多-未剔除其他标签.xlsx", index=False)


def regularize_data_alt():
    """
    *Run in console*

    * 仅用于a>1情况
    * 多比一：多条后果文本对应一个标签的数据。需要人工核查。a>1，b=1

    把数据改写成一一对应的形式，输出到新的df里。
    只保留（描述文本的项数）/（不安全事件的类数）均为1的数据
    再剔除“其他类型的数据”

    * 输出的标签范围为 1-18(19)
    """
    import os
    import pandas as pd
    import numpy as np
    import jieba
    import src.preprocessing as pre
    from src.loadfiles import (Loader,
                               load_data,
                               load_pretrained,
                               )

    df = load_data("../data/1-管制专业不安全事件匹配结果.xlsx")  # 读取数据库
    # 创建停用词列表、乱码词列表
    stopwords = []
    gibberish = Loader().load_gibberish('../nlp/gibberish.txt')
    # 读取word2vec词典
    keys = Loader().load_keys("../nlp/word2vec/keys.txt")
    # 读取缩写词典
    abbr = {}  # dict
    # 初始化分词器
    tk = jieba.Tokenizer()
    tk.load_userdict(r'../nlp/atc_terms.txt')  # 读取自定义词典
    df = pre.preprocess_df(df,
                           tk=tk,
                           stopwords=stopwords,
                           gibberish=gibberish,
                           word2vec_keys=keys,
                           abbr_dict=abbr,
                           keep_english=True,
                           )

    # 填写到新的df里
    df_new = pd.DataFrame(columns=["危险源编号", "后果原文", "后果", "不安全事件", "不安全事件2", "不安全事件3",
                                   "不安全事件4", "不安全事件5", "label", "label2", "label3", "label4", "label5"])

    # 读取不安全事件标签
    label_dict = {}
    for l in [line.strip() for line in open("../data/label-guanzhi.txt", encoding='UTF-8').readlines()]:
        v, k = l.split('\t')  # 1	跑道侵入
        label_dict[k] = int(v)

    new_idx = 0
    for i in range(len(df)):  # 对每行进行遍历

        a = 1  # a表示后果文本的点数，b表示后果文本对应的标签数

        # 计算b
        b = 5 - sum(df.iloc[i, 2:7].isna())
        if b != 1:
            continue

        # b=1时，无论a为何值，均写入数据，后面再删除a=1的数据
        line = df.loc[i, ["危险源编号", "后果"]].tolist()
        t = ''
        for _ in df.loc[i, "text"]:
            t += _
        line.append(t)
        # 添加不安全事件文本和标签
        # line.append(df.loc[i, "不安全事件"])
        # line.append(label_dict[df.loc[i, "不安全事件"]])
        for col_name in df.columns[2:7]:  # "不安全事件，不安全事件2，……不安全事件5
            line.append(df.loc[i, col_name])
        for col_name in df.columns[2:7]:
            if not pd.isna(df.loc[i, col_name]):
                line.append(label_dict[df.loc[i, col_name]])
            else:
                line.append(np.nan)
        df_new.loc[new_idx] = line  # 写入新的df
        new_idx += 1

    # 删除a=1的数据（危险源编号没重复的）
    unique_mask = (1 - df_new.duplicated(subset="危险源编号", keep=False)).astype('bool')
    df_new = df_new.drop(index=df_new.index[unique_mask])

    # 输出
    df_new.drop(df_new.index[df_new["label"] == 19]).to_excel("../data/管制-多比一-已剔除其他标签.xlsx", index=False)
    df_new.to_excel("../data/管制-多比一-未剔除其他标签.xlsx", index=False)


def split_df_on_label(df, train_size=.8, test_size=None, shuffle=True, random_state=42):
    """
    Split the dataframe in proportion to label.

    This is to make sure that all labels
    present in both train/val set and test set.

    Notes
    -----
    Label "Other events" are thrown.
    """
    if shuffle:
        df = df.sample(frac=1, random_state=random_state)

    l_label = list(range(1, 19))
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for lbl in l_label:
        dft = df.loc[df['label'] == lbl, :]
        df_1 = dft.iloc[:int(train_size * len(dft)), :]
        df_2 = dft.iloc[int(train_size * len(dft)):, :]
        if len(df_1) == 0:
            warnings.warn(f"Training set missing label {lbl}.")
        if len(df_2) == 0:
            warnings.warn(f"Test set missing label {lbl}.")
        df_train = pd.concat((df_train, df_1))
        df_test = pd.concat((df_test, df_2))

    print("Split complete.")
    print(f"Training set size: {len(df_train)}.")
    print(f"Test set size: {len(df_test)}.")
    return df_train, df_test


if __name__ == "__main__":
    # 数据集筛选
    # regularize_data_alt()

    # 数据集划分
    name = 'guanzhi_sep_vec_each_1350_select'
    df = load_pickle(f'out/dump/{name}.pkl')
    a, b = split_df_on_label(df)

    a.to_excel(f"./out/datasets/{name}-train.xlsx", index=False)
    b.to_excel(f"./out/datasets/{name}-test.xlsx", index=False)
    save_pickle(a, f"./out/datasets/{name}-train.pkl")
    save_pickle(b, f"./out/datasets/{name}-test.pkl")

    pass

