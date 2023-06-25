"""对文本进行向量化、降维、标签生成。

利用word2vec中文预训练词向量（数据集：百度百科）*获取词向量（300d）
得到词向量序列后，有两种处理方式：
    a) 不做处理
    b) 用词向量的平均值表征句子。得到维度为300的句子特征向量。
通过PCA降维（未完成）。
生成标签。

* 模型来源：
    word2vec：https://github.com/Embedding/Chinese-Word-Vectors
    RoBERTa-wwm-ext, Chinese：https://github.com/ymcui/Chinese-BERT-wwm

"""
import warnings

from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel

from src.tfidf import tfidf
from src.modules import _1col_300col
from src.loadfiles import load_pretrained


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        # corpus_path = datapath('lee_background.cor')
        corpus_path = "../out/corpus.txt"
        for line in open(corpus_path, encoding='utf-8'):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


class Tokenizer:
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.model = None

    def load_model(self, model):
        self.model = model

    def tokenize(self, s: str):
        """将传入的任意字符串转化为向量并返回"""
        pass

    def _tokenize_sentence(self, sentence):
        """将句子转化为向量并返回"""
        pass

    def _tokenize_word(self, word):
        """将词语转化为向量并返回"""
        pass


class W2vTokenizer(Tokenizer):
    def __init__(self):
        super().__init__('word2vec')
        self.load_model()

    def load_model(self):
        self.model = load_pretrained("./nlp/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2")

    def tokenize(self, s: str):
        """该对象目前仅支持tokenize单个词语"""
        return self._tokenize_word(s)

    def _tokenize_word(self, word):
        return self.model[word]


class BertTokenizerNew(Tokenizer):
    def __init__(self):
        super().__init__('bert')
        self.load_model()

    def load_model(self):
        self.model = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

    def tokenize(self):
        pass


def cal_doc_vec_df(df: pd.DataFrame, model, column_in='text', column_out='doc_vec', how='mean', **kwargs):
    """对一个Dataframe的指定列计算特征向量。

    Parameters
    ----------
    df : pd.DataFrame
        待处理的数据表。
    model : gensim.models.keyedvectors.KeyedVectors
        词向量模型。
    column_in : str
        需要处理的列。Default column "text".
    column_out : str
        输出df中，句子向量的列名。Default "doc_vec".
        在新版本中此项被废弃。输出的向量会以300列的形式接在后面。(how='each'时除外，由于输出是列表，因此仍然是原本的格式)
        这一参数仍然被保留，但是不推荐使用。
    how : str
        处理方式，见_cal_doc_vec(). Default "mean".

    **kwargs
        dict_tfidf : dict
            格式：{'word': tf-idf value}，仅在how='tfidf'时生效。

    Returns
    -------
    处理过后的数据表。词向量在变量column_out列中。
    """
    df[column_out] = df[column_in].apply(cal_doc_vec, model=model, how=how, **kwargs)
    if how != 'each':
        df = pd.concat((df, _1col_300col(df)), axis=1)
    return df


def add_label_df(df: pd.DataFrame, column_in='不安全事件', column_out='label'):
    """将标签添加至df中
    """
    # 管制专业
    df[column_out] = df[column_in].apply(_get_single_label)
    return df


def cal_doc_vec(doc: list, model=None, how='mean', req_vec=True, **kwargs):
    """计算单条文本的特征向量。

    Parameters
    ----------
    model : gensim.models.keyedvectors.KeyedVectors
        词向量模型。
    doc : 待计算的文本，作为分词后的列表传入。
        例如：['通信', '信道', '无法', '使用']
    how : 计算的方法，默认是mean。
        'mean'，对每个词的词向量取平均值，作为该句子的向量。
        'tfidf'，向量的tfidf加权平均。
        'each'，返回每个词向量构成的列表。

    Returns
    -------
    np.ndarray
        当how参数为'mean'或'tfidf'时，返回句子的特征向量。
        当how参数为'each'时，返回每个词向量依序构成的列表。
    """
    if req_vec and (model is None):
        raise RuntimeError("缺少参数model")
    #     model = load_pretrained()

    vec = None

    if how == 'mean':
        if req_vec:
            try:
                l_vec = [model[x] for x in doc]
            except KeyError as e:
                print(e)
                vec = np.array([0]*300)
            vec = np.mean(l_vec, axis=0)

    elif how == 'tfidf':
        dict_tfidf = kwargs.get('dict_tfidf')
        # dict_tfidf = tfidf()
        if dict_tfidf is None:
            raise RuntimeError(f"方法{how}需要传入dict_tfidf对象")
        d_weight = {}
        for x in doc:
            if not (x in dict_tfidf):
                warnings.warn(f"未在tfidf词典中找到词语{x}，因此将该词权重置为0。")
                d_weight[x] = 0.000001
            else:
                d_weight[x] = dict_tfidf[x]
        w_sum = sum(d_weight.values())  # 权重之和，用于归一化
        for key in d_weight:
            d_weight[key] = d_weight[key] / w_sum  # 归一化权重

        if req_vec:
            vec = sum([model[x]*d_weight[x] for x in doc])

    elif how == 'each':
        try:
            l_vec = [model[x] for x in doc]
        except KeyError as e:
            print(e)
            l_vec = [np.array([0] * 300)]
        vec = l_vec

    else:
        raise ValueError(f"Unknown method \"{how}\" for calculating document vector.")
    return vec


def _get_single_label(label):
    """把包含两项标签的数据转换成一项标签。

    目前采用的方法是取第一个标签，例如“1/4”取1。
    后续可以改为取样本数少的标签。
    最好的情况是人工核对每个标签。
    或许可以复制样本，然后打上两个标签。最后通过置信度阈值来尝试从一个样本中预测出多个标签分类。
    """
    # return int(str(label).split('/')[0])
    return str(label).split('/')[0]





if __name__ == "__main__":
    dict_tfidf = tfidf('../out/corpus_未预训练词拆为单字.txt')
    cal_doc_vec(['信号', '干扰'], how='tfidf', dict_tfidf=dict_tfidf)
