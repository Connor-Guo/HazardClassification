"""读取文件操作

不要引入src的其他模块
"""
import pandas as pd
from typing import Optional
from collections import defaultdict
import pickle

from gensim.test.utils import datapath
from gensim import utils
import gensim.models


class Loader(object):
    def __init__(self) -> None:
        self.data = {}

    def _l(self, fp: str):
        with open(fp, encoding='UTF-8') as f:
            for line in f.readlines():
                if line.startswith('#'):  # 忽略注释行
                    continue
                else:
                    yield line.strip()

    def load(self, fp: str, name=None) -> list:
        l = list(self._l(fp))
        if name:
            self.data[name] = l
        return l

    def load_abbr(self, fp: str) -> dict:
        """读取缩写词的词典，返回{缩写词 -> 对应完整中文}"""
        df = pd.read_csv(fp, header=None).fillna('')
        keys = df.iloc[:, 0]
        values = df.iloc[:, 1]
        d = {k: v for k, v in zip(keys, values)}
        self.data["abbr_dict"] = d
        return d

    def load_gibberish(self, fp: str) -> list:
        l = self.load(fp)
        l.extend(['\t', '\uf06c'])
        self.data["gibberish"] = l
        return l

    def load_keys(self, fp: str) -> list:
        """读取预训练模型的字典"""
        # keys = [line.strip() for line in open(fp, encoding='utf8').readlines()]
        l = self.load(fp)
        self.data["word2vec_keys"] = l
        return l

    def load_label_dict(self, fp: str) -> dict:
        """{"跑道入侵" -> 1}"""
        l = self.load(fp)
        d = {}
        for line in l:
            v, k = line.split("\t")
            d[k] = int(v)
        self.data["label_dict"] = d
        return d

    def load_terms(self, fp):
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
        lines = list(self._l(fp))
        sym_dict = {}
        for line in lines:
            if line:
                if line[0].isdigit():
                    term_no = int(line.split(' ')[0])
                    print(f"Symloader: {line} loaded.")
                else:
                    list_terms = line.split('/')
                    for subterm in list_terms:
                        sym_dict[subterm] = term_no
        self.data["dict_terms"] = sym_dict
        return sym_dict


def load_data(fp: str) -> pd.DataFrame:
    df = pd.read_excel(fp, sheet_name="不安全事件匹配结果", header=1)
    # df.drop_duplicates(inplace=True)
    # df.drop_duplicates(subset=["危险源编号"], inplace=True)  # 删除重复行和重复危险源编号（删除重复行后还有38条）
    return df


def load_washed_data(fp: str):
    df = pd.read_excel(fp)
    return df


def load_pretrained(fp: str = None):
    """读取预训练的模型"""
    # 训练自己的模型
    # sentences = MyCorpus()
    # model = gensim.models.Word2Vec(sentences=sentences, vector_size=200, window=3, min_count=3)

    # 或者读取模型
    if fp is None:
        fp = "../nlp/models/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2"
    print(f"Loading model from \"{fp}\" ...")
    model = gensim.models.KeyedVectors.load_word2vec_format(fp, binary=False)
    print("Model loaded.")
    return model
    # 测试模型是否有效
    # pairs = [
    #     ('运行', '工作'),   # a minivan is a kind of car
    #     ('运行', '使用'),   # still a wheeled vehicle
    #     ('运行', '传输'),  # ok, no wheels, but still a vehicle
    #     ('运行', '故障'),    # ... and so on
    #     ('故障', '中断'),
    # ]
    # for w1, w2 in pairs:
    #     print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))
    #
    # print(wv.most_similar(positive=['人员'], topn=20))


def save_pickle(obj, fp, *args):
    with open(fp, 'wb') as tf:
        pickle.dump(obj, tf)


def load_pickle(fp, *args):
    with open(fp, 'rb') as tf:
        obj = pickle.load(tf)
    return obj


if __name__ == '__main__':
    loader = Loader()
    label_dict = loader.load_label_dict("../data/label-guanzhi.txt")
    pass

