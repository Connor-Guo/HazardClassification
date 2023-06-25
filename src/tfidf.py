"""tf-idf的功能函数"""
from collections import defaultdict
import math
import operator


def get_tfidf_dict(corpus_fp: str):
    """

    Parameters
    ----------
    corpus_fp : file path for corpus
        语料库必须包含所有分词中可能遇到的词、

    Returns
    -------
    dict
        格式：{字/词: 归一化的tf-idf值}

    """


def load_corpus(fp: str):
    c = [line.strip() for line in open(fp, encoding='utf8').readlines()]
    corpus = []
    for line in c:
        l = line.split(sep=' ')
        corpus.append(l)
    return corpus  # [['w1', 'w2'...], ['', ''...]...]


def cal_tfidf(list_words):
    # 获取所有关键词的词典
    doc_frequency = defaultdict(int)  # freq字典存储了所有词的词频
    for i in list_words:  # i是一个文档，['w1', 'w2'...]
        for j in i:  # j是一个词，'w1'
            doc_frequency[j] += 1

    # 计算每个词的TF值
    word_tf = {}  # 存储词的tf值
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())

    # 计算每个词的IDF值
    doc_num = len(list_words)
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i] += 1
    for i in doc_frequency:
        word_idf[i] = math.log(doc_num / (word_doc[i] + 1))

    # 计算每个词的TF*IDF的值
    word_tf_idf = {}
    for i in doc_frequency:
        word_tf_idf[i] = word_tf[i] * word_idf[i]

    return word_tf_idf

    # 对字典按值由大到小排序
    # dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    # return dict_feature_select


def tfidf(corpus_fp: str = None):
    if corpus_fp is None:
        corpus_fp = './out/corpus_未预训练词拆为单字.txt'
    return cal_tfidf(load_corpus(corpus_fp))


if __name__ == '__main__':
    # corpus = load_corpus('../out/corpus_未预训练词拆为单字.txt')
    # a = cal_tfidf(corpus)
    a = tfidf('../out/corpus_未预训练词拆为单字.txt')
    print(a)
