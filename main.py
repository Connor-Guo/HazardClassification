# -*- encoding utf-8 -*-
"""
程序示例

分为三个步骤：
    1. 数据预处理
        具体信息见preprocessing.py
    2. 特征提取
        具体信息见vectorize.py
    3. 分类（+评估）
        目前为止，尝试过三种分类方法：
        1) 对得到的词向量**求平均/根据tfidf加权求平均**后，直接分类（效果很差）
        2) 计算文本与术语词典中词语的向量相似度（效果一般）
        3) 设置一些专业关键词，忽略其他的词。仅计算这些词的向量，取平均作为句子向量，再利用分类器分类。
            具体信息见./classification/catb.py
            230505：目前使用了Catboost，准确率达到0.8533，不足之处在于BALANCED ACCURACY仅有0.4991
                    18类的P, R, F1的加权平均分别达到0.84, 0.85, 0.83.


***表示费时间的语句
"""

import os
import sys
import pickle
from collections import defaultdict

import pandas as pd
import numpy as np
import jieba
from sklearn.model_selection import train_test_split

import src.preprocessing as pre
import src.tokenize as vectorize
import src.modules as modules

from src.loadfiles import (Loader,
                           load_data,
                           load_washed_data,
                           load_pretrained,
                           load_pickle,
                           save_pickle,
                           )
from src.tfidf import tfidf
from src.symdict import (load_terms,
                         term2vec,
                         term2list)
from src.similarity import (sim_m2m,
                            sim_v2m,
                            top_sims_name,
                            top_class,
                            sim_word2term
                            )


EXP_NAME = "guanzhi_sep_vec_each_1350_select"  # 实验名称
__SAVE__ = False  # 是否保存此次计算的df和术语特征
__LOAD__ = not __SAVE__  # 是否读取预计算的df和术语特征
__PROCESS__ = __SAVE__  # 是否从头计算df和术语特征（速度较慢）


if __name__ == '__main__':
    # df = load_data("./data/1-管制专业不安全事件匹配结果.xlsx")  # 读取数据库
    df = load_washed_data("./data/管制-一比多-已剔除其他标签.xlsx")
    # 创建停用词列表、乱码词列表
    loader = Loader()
    stopwords = [line.strip() for line in open('./nlp/hit_stopwords.txt', encoding='UTF-8').readlines()]
    gibberish = loader.load_gibberish('./nlp/gibberish.txt')
    # 读取word2vec词典
    keys = loader.load_keys("./nlp/word2vec/keys.txt")
    # 读取缩写词典
    abbr = loader.load_abbr("./data/abbr/管制.csv")  # dict
    # 读取术语 -> {'跑道侵入': 1, '跑道入侵': 1, ...}
    dict_terms = load_terms('./data/sym_dict/specialist_管制.txt')
    # 初始化分词器
    tk = jieba.Tokenizer()
    tk.load_userdict(r'./nlp/atc_terms.txt')  # 读取自定义词典
    # 创建关键术语，分词只保留在这个列表中的词语
    specialist_words = term2list(dict_terms, stopwords=stopwords,
                                 gibberish=gibberish,
                                 word2vec_keys=keys,
                                 tk=tk)
    # 预处理
    if __PROCESS__:
        df = pre.preprocess_df(df,
                               tk=tk,
                               stopwords=stopwords,
                               gibberish=gibberish,
                               word2vec_keys=keys,
                               abbr_dict=abbr,
                               select_on=specialist_words
                               )
        # （可省略）生成语料库，将预处理结束的句子以文本形式输出到<fff>，用于tfidf训练
        # pre.to_corpus(df, './out/corpus_guanzhi.txt')

    # 词向量计算
    if __PROCESS__:
        # （***）读取词向量模型
        model = load_pretrained("./nlp/models/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2")
        # （需要模型）计算句子特征向量
        # dict_tfidf = tfidf('./out/corpus_未预训练词拆为单字.txt')  # 用tfidf对词语加权
        df = vectorize.cal_doc_vec_df(df, model,
                                      how='each',
                                      # dict_tfidf=dict_tfidf
                                      )
        # df = vectorize.add_label_df(df)  # 向量列叫doc_vec，标签列叫label

    # 思路2：术语匹配
    # （需要模型）计算术语特征向量
    if __PROCESS__:
        term_vec = term2vec(dict_terms, model, how='each',
                            stopwords=stopwords,
                            gibberish=gibberish,
                            word2vec_keys=keys,
                            # dict_tfidf=dict_tfidf,
                            tk=tk
                            )
    # 保存/读取df和术语特征
    if __LOAD__:
        print("Loading data...")
        df = load_pickle(f'./out/dump/{EXP_NAME}.pkl')
        term_vec = load_pickle('./out/dump/term_vec.pkl')
        print("Data loaded.")
    if __SAVE__:
        print("Saving data...")
        # 暂时保存df和术语特征以省略读取模型步骤
        df.to_excel(f'./out/dump/{EXP_NAME}.xlsx', index=False)  # 仅供阅读
        save_pickle(df, f'./out/dump/{EXP_NAME}.pkl')
        save_pickle(term_vec, './out/dump/term_vec.pkl')
        print("Data saved to /out/dump/.")

    # 生成训练集和测试集
    # 见modules.py

    # -----------------------------------------------------------------
    # 特征提取结束，开始进行相似度计算
    # -----------------------------------------------------------------

    # -----------------------方法1--------------------------
    # 此段代码针对how='each'的特征提取结果，得到分类结果label记录在"pred_label"列中
    # 设数据为Di，其中i为索引，对Di的第j个词向量Dij，计算其与term_vec中每个词语的相似度
    # df = sim_word2term(df, dict_terms, term_vec)
    # df.to_excel(f'./out/result/result_Sim_{EXP_NAME}.xlsx', index=False)

    # 方法2见``networks.py``

    # -----------------------方法3--------------------------
    # 对输出的向量序列求平均，作为句子向量，用句子向量训练一个Catboost分类器，见``./classification/catb.py``
    # x_data = np.array(df['doc_vec'].apply(lambda x: np.mean(x, axis=0)).tolist())
    # y_data = df['label'].to_numpy().astype(int)
    # X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=33)

    # -----------------------方法4--------------------------
    # 微调bert分类器，见``./classification/bert.py``

    # -----------------------------------------------------------------
    # 模型评估（evaluate模块）
    # -----------------------------------------------------------------
    from src.evaluate import Evaluator
    # eva = Evaluator(y_test, y_pred)
    # eva.evaluate()

    pass
