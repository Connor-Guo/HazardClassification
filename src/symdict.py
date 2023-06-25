# -*- encoding utf-8 -*-
"""专业词典有关的操作

子术语和术语的定义：
    例如
    飞行取消、返航、迫降、改航或备降/机场关闭/返航/备降/飞行取消/迫降/改航/复飞
    主术语为“飞行取消、返航、迫降、改航或备降”
    子术语为“飞行取消、返航、迫降、改航或备降/机场关闭/返航/备降/飞行取消/迫降/改航/复飞”

期待的输出：
1. 专业字典与其对应的向量：
    {子术语: vec}
2. 专业术语的包含关系：
    {子术语: 主术语}
"""
from typing import List
from collections import defaultdict

from src.loadfiles import load_pretrained, Loader
from src.tokenize import cal_doc_vec
from src.preprocessing import process_text
from src.tfidf import tfidf


def load_terms(fp: str):
    """读取专业词典。

    专业词典的格式见'./data/sym_dict/...'。

    Parameters
    ----------
    fp : str
        专业词典的文件路径

    Returns
    -------
    dict
        返回一个字典，格式为
        {子术语: 事件类型（int）}
        例如{'飞行取消': 6,
            '机场关闭': 6,
            '返航': 6}
    """
    return Loader().load_terms(fp)


def term2vec(d_term: dict, model, how='mean', **kwargs):
    """将术语字典转化为向量（的列表）

    Parameters
    ----------
    d_term
    model
    how

    Returns
    -------
    dict
        格式为{子术语: vec}
        例如：{'尾流间隔': [vec(尾流), vec(间隔)]}
    """
    dict_vec = {}
    for subterm in d_term.keys():
        # para = PreproPara().kwargs()
        # 逐个计算关键词的向量，并存入相同结构的字典中
        doc = process_text(subterm, **kwargs)[0]  # 先分词，传一堆烂参数
        v = cal_doc_vec(doc, model=model, how=how, **kwargs)  # 转换为句子向量或词向量列表
        dict_vec[subterm] = v
    return dict_vec


def term2list(d_term: dict, **kwargs):
    """
    Create a list containing every single word in the term dictionary.

    **kwargs : Same as **kwargs in func process_text().
    """
    l_words = []
    for subterm in d_term.keys():
        l_subterm_words = process_text(subterm, **kwargs)[0]
        for w in l_subterm_words:
            if w not in l_words:
                l_words.append(w)
    return l_words



if __name__ == '__main__':

    # 创建停用词列表、乱码词列表
    stopwords = [line.strip() for line in open('../nlp/hit_stopwords.txt', encoding='UTF-8').readlines()]
    gibberish = [line.strip() for line in open('../nlp/gibberish.txt', encoding='UTF-8').readlines()]
    # 读取word2vec词典
    loader = Loader()
    keys = loader.load_keys("../nlp/word2vec/keys.txt")
    dict_tfidf = tfidf('../out/corpus_未预训练词拆为单字.txt')  # 用tfidf对词语加权
    dict_terms = load_terms()

    model = load_pretrained()
    term_vec = term2vec(dict_terms,
                        model,
                        how='tfidf',
                        stopwords=stopwords,
                        gibberish=gibberish,
                        word2vec_keys=keys,
                        dict_tfidf=dict_tfidf)
    print(term_vec)
