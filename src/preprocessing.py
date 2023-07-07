"""文本预处理.

目的：Regular the structure of text data & remove implicit noise for better feature extraction
0.	Text written in points
    有些文本存在分点阐述的结构，在这样的结构下，一个点一般包含了一个独立的不安全事件。
    因此，需要将这些文本按照其编号拆解为多段文本，以保证每段文本仅对应一个不安全事件。
    例如：
        "1、造成管制员道面状况代码通报错误。
        2、跑道代码与跑道表面评估状况不符，易发生航空器冲偏出跑道不安全事件。"
            处理后变为：
        {“造成管制员道面状况代码通报错误。”,
        “跑道代码与跑道表面评估状况不符，易发生航空器冲偏出跑道不安全事件。”}
1.	分词
    将中文句子拆分成多个词语，例如……
2.	停止词Stop words
    停止词是指在执行分类算法时不具有重要意义的词，例如：“或”、“从而”、“导致”等词语。
    需要在预处理阶段去除停止词。构建了停止词库，当分词结果中的词语存在于停止词库时，删除这些词。
3.	缩写Abbreviations
    数据集中的一些词语是以英文或中文缩写的形式表述的，但是这些表述并不统一，
    这可能造成一个实义词被表达为中文原词、中文缩写、英文缩写三种形式。
    例如：
        “导致航空器低于MVA”和“导致航空器低于最低安全高度”
        实际上表示的是同样的意思。
    对于这样的情况，需要对缩写进行统一。为此，建立了缩写词库。
4.	Noise Removal（放在第0步之前做）
    数据集中的噪声主要为标点符号。标点符号对人们理解句意至关重要，但对于计算机来说则是噪声。
    在预处理中，对标点符号予以去除。


操作方法：对于一个完整的句子s : str
    调用process_text()函数，传入三个参数即可


目前bug：
    1.分词后存在连续空字符：['','','']  # 已解决，这些不是空字符，而是英文空格。
    2.存在空句子和仅包含空字符的空句子：[['内容'], [], ['内容']]
                                        [['']]
    # 以上问题，已通过在process_text()中添加第4步解决。

    3.删不掉的神秘逗号句号（df2091行）
    # 可以通过与上文相同方法解决，但是没有添加对应代码，因为影响程序运行速度。
    # 很奇怪的是，停用词库里有相同的逗号和句号，通过应用停用词库可以去除数据中心大部分标点符号，只有几条数据的留了下来。
    # 这是什么原因呢？
"""
import re
from typing import Optional

import numpy as np
import pandas as pd
import jieba

from src.loadfiles import load_data


class Preprocessor(object):
    def __init__(self, data: pd.DataFrame, **params):
        self.data = data
        self.params = params

    def __repr__(self):
        return f"Preprocessor({self.params.keys()})"

    def exec(self):
        self.data = preprocess_df(self.data, **self.params)
        return self.data


def preprocess_df(df: pd.DataFrame, *args, **kwargs):
    """在数据库层面对文本进行预处理。返回一个处理后的df。

    处理步骤：
    1.首先将文本分点，
    2.然后去除乱码（噪声）和无意义空格，
    3.再进行分词，包括：
        r) 正则替换缩写词
        a) 采用jieba进行分词
        b) 去除停用词
        c) 进一步拆分不在字典中的词
    4.进行善后处理，包括以下几步：
        a) 去除列表中的空字符串[..., '']
        b) 去除空列表[[], ...]
        c) （未实装）（对_process_text()函数）
            如果结果为空（[[]]）则返回None，None需要在本函数中被处理。
            删除所有结果为None的行。

    5.把df拆分：每一行只能包含一个点。确保df中不出现嵌套的list。
        这一步是为了便于后续分类。

    1-4步由_process_text()完成，第5步在本函数中完成。
    默认每个换行符会分割，每个分隔符处也会分割。
    """

    df["text"] = df["后果"].apply(process_text, axis=1, result_type='reduce',
                                **kwargs)
    # 把df拆分：每一行只能包含一个点。确保df中不出现嵌套的list。
    # 需要创建一个全新的df，然后对旧的df逐行遍历，填充到新df中。
    df_unique = pd.DataFrame(columns=df.columns)
    c_text_id = np.argwhere(df.columns == 'text').ravel()[0]  # 获取text列的列索引
    for i in range(len(df)):
        text = df.iloc[i, c_text_id]
        for p in text:
            tmp_df = df.iloc[[i], :].copy()
            tmp_df['text'] = tmp_df['text'].astype('object')  # 为了将list对象赋值给df的cell，需要将df对应列的数据类型改为object
            idx = tmp_df.index[0]
            # tmp_df.loc[idx, 'text'] = p  # 此处原本会出现SettingWithCopyWarning，因为tmp_df实际上是df的一个copy（与copy相对的是view）
            tmp_df.at[idx, 'text'] = p
            # 实际上我们就是想copy一个df的副本，在副本上进行操作，这样不会改变df的值
            # 因此不妨把代码改为tmp_df = df.iloc[[i], :].copy()
            # 然后用推荐的df.loc[] = ... 修改值
            # 成功辽
            df_unique = pd.concat((df_unique, tmp_df))
    df_unique.index = [_ for _ in range(len(df_unique))]
    return df_unique


def preprocess_bert_df(df: pd.DataFrame, *args, **kwargs):
    """对bert使用的数据进行预处理并打上标签。

    目前只有去除gibberish的功能。
    与process_df输出格式不一样，需要注意。

    kwargs :
        label_dict : dict
            形如 {跑道侵入 -> 1} 的字典，用于转换标签
        gibberish : list
            乱码列表。
    """
    gibberish = kwargs.get("gibberish")
    abbr_dict = kwargs.get("abbr_dict")
    label_dict = kwargs.get("label_dict")

    df_new = pd.DataFrame(columns=["危险源编号", "后果原文", "后果", "不安全事件", "不安全事件2", "不安全事件3",
                                   "不安全事件4", "不安全事件5", "label", "label2", "label3", "label4", "label5"])
    df_new["危险源编号"] = df["危险源编号"]
    df_new["后果原文"] = df["后果"]
    df_new["后果"] = df["后果"]
    # 上标签
    if label_dict:
        labels_cols = ['label', 'label2', 'label3', 'label4', 'label5']
        chinese_labels_cols = ['不安全事件', '不安全事件2', '不安全事件3', '不安全事件4', '不安全事件5']

        def _gl(s):  # get label from string
            if pd.isna(s):
                return np.nan
            else:
                return label_dict[s]

        for lbl_col, cn_col in zip(labels_cols, chinese_labels_cols):
            df_new[cn_col] = df[cn_col]
            df_new[lbl_col] = df[cn_col].apply(_gl)

        # 检查标签是否递增
        labels = df_new.loc[:, labels_cols].to_numpy()
        _labels_ascending_check(labels)

    # 预处理过程
    df_new.drop_duplicates(inplace=True)
    print("[Preprocess] {} duplicated samples are dropped.".format(len(df) - len(df_new)))
    if gibberish:
        df_new["后果"] = df_new["后果"].apply(_denoise, args=(gibberish, ))
        print("[Preprocess] Gibberish removed.")
    if abbr_dict:
        df_new["后果"] = df_new["后果"].apply(_replace_k_with_v, args=(abbr_dict, ))
        print("[Preprocess] Abbreviations replaced.")
    return df_new


def _labels_ascending_check(labels: np.ndarray) -> None:
    """检查每行的标签是否满足要求，空标签用np.nan表示"""
    # labels = np.array([[1, 2, 3], [1, 2, np.nan], [1, np.nan, np.nan]])
    # labels = np.array([[1, 2, 3], [np.nan, np.nan, np.nan], [1, np.nan, 3]])
    # labels = np.array([[1, 2, 3], [1, 2, np.nan], [1, np.nan, 3]])
    # labels = np.array([[1, 3, 2], [1, 1, np.nan], [1, 2, 3]])
    error_flag = False
    # 检查是否都有标签
    if any(np.isnan(labels[:, 0])):
        empty_lines = np.argwhere(np.isnan(labels[:, 0]))
        print("[Preprocess] The following lines lacks a label: {}".format(empty_lines.ravel() + 1))
        error_flag = True
    # 检查标签是否连续，即两个标签之间不包含nan
    continuous_arr = 1 - np.isnan(labels).astype(int)  # 将labels矩阵的所有nan变为0，数字变为1
    continuous_arr = np.diff(continuous_arr)  # 逐列作差，如果有nan夹在两个数字之间，则该矩阵会出现数字1
    fault_coords = np.argwhere(continuous_arr == 1)
    if len(fault_coords) > 0:
        print("[Preprocess] The following lines contain nan between two labels: {}".format(fault_coords[:, 0] + 1))
        error_flag = True
    # 检查标签是否单调增
    # 后一列减前一列，如果是nan，则这列通过检查（已经到了最后一列标签）
    # 如果大于0，通过，如果小于等于0，记录
    ascending_arr = np.diff(labels)
    descending_coords = np.argwhere(ascending_arr <= 0)
    if len(descending_coords) > 0:
        print("[Preprocess] The following lines do not ascend: {}".format(descending_coords[:, 0] + 1))
        error_flag = True

    if error_flag:
        raise ValueError("The labels failed to pass labels ascending check. Please revise as suggested.")


def to_corpus(df: pd.DataFrame, fp: str, encoding='utf-8'):
    """将preprocess得到的df输出到txt文件，每点一行。

    删除了重复文本！
    """
    # 删除重复文本
    l = []
    c_text_id = np.argwhere(df.columns == 'text').ravel()[0]  # 获取text列的列索引
    for i in range(len(df)):  # 每一行
        s = ""
        for word in df.iloc[i, c_text_id]:
            s += (word + " ")
        s = s[:-1]
        l.append(s)
    df_cor = pd.Series(l).drop_duplicates()
    df_cor.to_csv(fp, encoding=encoding, index=False, header=False)


def process_text(s: str, **kwargs):  # 注意：传回的是两层列表
    """将文本按点分割，分词后返回，具体步骤如下。

    1.将文本分点，
    2.去除乱码（噪声）和无意义空格，
    3.正则替换
        a) 替换缩写词
        b) 替换剩余的英文字母（取决于keep_english参数，默认替换）
    4.进行分词
        a)
    5.进行善后处理，包括以下几步：
        a) 去除列表中的空字符串[..., '']
        b) 去除空列表[[], ...]

    默认每个换行符会分割，每个分隔符处也会分割。
    假设有n个点，则返回包含n个元素的列表，每个元素
    都是包含了该点关键词的列表。（列表套列表）

    Parameters
    ----------
    s
    kwargs
        gibberish : list
            乱码
        stopwords : list
            停用词
        tk : jieba.Tokenizer
            jieba分词器
        word2vec_keys : list
            word2vec预训练模型字典
        abbr_dict : dict
            缩写词典
        select_on : list
            重要词的列表，结果中仅保留该列表中的词

        keep_english : bool, default False.
            是否保留英文字母，默认不保留。

    Returns
    -------
    out : list[list]
        内层的list是字符串分词后的每个单词，
        外层的list是字符串分点后的每个小点。
    """
    sep = ['；']  # 默认分隔符为换行符、分号
    gibberish = kwargs.get("gibberish")
    stopwords = kwargs.get("stopwords")
    tk = kwargs.get('tk')
    word2vec_keys = kwargs.get("word2vec_keys")
    abbr = kwargs.get("abbr_dict")
    select_on = kwargs.get("select_on")
    keep_english = kwargs.get("keep_english", False)

    out = []
    points = _cut_para(s, ["；"])  # 1
    for p in points:
        p = _denoise(p, gibberish)  # 2
        p = _sub(p, abbr)  # 3a
        if not keep_english:
            p = _del_english(p)  # 3b
        p = _cut_sentence(p, stopwords, tokenizer=tk, word2vec_keys=word2vec_keys, select_on=select_on)  # 4
        #  5
        # 5a除列表中的空字符串[..., '']
        while ' ' in p:  # 英文空格
            p.remove(' ')
        # 5b删除空列表
        if len(p) == 0:
            continue
        else:
            out.append(p)
    if len(out) == 0:
        out = [['。']]
    return out


def _cut_para(para: str, sep):
    """把一段话按照分隔符进行拆分，以列表形式输出。

    Parameters
    ----------
    para : 段落文本
    sep : list,分隔符

    Returns
    -------
    out : list，拆分后的列表。

    Examples
    --------
    test_s = "运行不稳定，保障能力降低。\n
        \n
        目前区管中心KU站无在用业务承载，且故障较多，已计划下线；\n
        新机场KU站有承载各管局间自动转报高速线备份线路，暂不考虑下线，修复故障设备后保持运行。"
    _cut_para(test_s, ["；"])

    ['运行不稳定，保障能力降低。', '目前区管中心KU站无在用业务承载，且故障较多，已计划下线', '新机场KU站有承载各管局间自动转报高速线备份线路，暂不考虑下线，修复故障设备后保持运行。']
    """
    out = []
    cut_sentences = para.splitlines()  # 首先按行分割
    for s in cut_sentences:
        for d in sep:
            cut = s.split(d)  # 然后按分隔符分割
            out.extend(cut)
    while '' in out:
        out.remove('')  # 删除空项目
    return out


def _sub(s: str, abbr: dict = None) -> str:
    """替换缩写词，之后将所有英文字母删除"""
    if abbr is not None:
        for k, v in abbr.items():
            s = str(re.sub(k, v, s))
    return str(s)


def _del_english(s: str) -> str:
    """将所有英文字母删除"""
    s = re.sub('[a-zA-Z]', '', s)
    return str(s)


def _denoise(
    s: str,
    gibberish: list
) -> str:
    """去除数据库中的标点符号乱码，例如'&ldquo'，以及分点的编号，例如'1.'。"""
    for gib in gibberish:
        s = s.replace(gib, '')
    return s


def _replace_k_with_v(
    s: str,
    d: dict,
) -> str:
    """将字符串s按照字典d进行翻译，将s中的key替换为value。
    字典d的key和value在替换时将被视为字符串"""
    for k, v in d.items():
        s = s.replace(str(k), str(v))
    return s


def _cut_sentence(
    s: str,
    stopwords: list = None,
    tokenizer: jieba.Tokenizer = None,
    keep_stopwords: bool = False,
    *args,
    **kwargs
) -> list:
    """利用结巴分词将句子拆解为词语，再删除停用词和多余空格，最后仅保留重要词语。

    词库为哈工大停用词库。
    对于不在word2vec词典中的词，需要进一步拆分成字。
    Parameters
    ----------
    s : str
        待拆解的句子
    stopwords : list
        停用词的列表
    keep_stopwords : bool
        是否保留停用词，默认为False。
    **kwargs
        有以下几种变量：
        word2vec_keys : list
            word2vec词典，部分分词结果不在此范围中，需要进一步拆分。
        tokenizer : jieba.Tokenizer
            分词器
        select_on : list
            如果传入该参数，则对分词结果进行筛选。
            只有在此列表中的词语才会被保留。

    Returns
    -------
    l_words : 列表形式的词语集合（已删除停止词）
    """
    if stopwords is None:
        stopwords = []
    tk = jieba.Tokenizer() if tokenizer is None else tokenizer
    word2vec_keys = kwargs.get("word2vec_keys")  # 这个参数一般肯定会传，所以就当他是必要参数，后面有必要重构再改
    l_select = kwargs.get("select_on")

    l_words = list(tk.cut(s))
    if not keep_stopwords:
        for idx, word in enumerate(l_words):
            if word in stopwords:
                l_words.remove(word)  # 移除停止词
            elif word == u'\u3000':
                l_words.remove(word)  # 移除unicode英文空格
    # 对词典外的词进行拆分，例如'主用'拆成('主','用')
    if word2vec_keys:
        for idx, word in enumerate(l_words):
            if not (word in word2vec_keys):
                l_words[idx] = (_ for _ in word)
            # l_words = list(itertools.chain(*l_words))
        l_words_ravel = []
        for w in l_words:
            if type(w) is str:
                l_words_ravel.append(w)
            else:
                for(ww) in w:
                    l_words_ravel.append(ww)
    else:
        l_words_ravel = l_words
    # 只保留select_on中的词语
    if l_select:
        l_words_select = [w for w in l_words_ravel if w in l_select]
    else:
        l_words_select = l_words_ravel
    return l_words_select


if __name__ == "__main__":
    df = load_data()  # 读取数据库

    # 创建停用词列表、乱码词列表
    stopwords = [line.strip() for line in open('../nlp/hit_stopwords.txt', encoding='UTF-8').readlines()]
    gibberish = [line.strip() for line in open('../nlp/gibberish.txt', encoding='UTF-8').readlines()]
    tk = jieba.Tokenizer()
    # 预处理
    df = preprocess_df(df, tk=tk, stopwords=stopwords, gibberish=gibberish)
    # 生成语料库，供词向量训练
    # to_corpus(df, '../out/corpus.txt')
    # 测试文本和代码
    # test_s = "核心交换机故障：1800M系统无法对外提供通信服务；"\
    #             "\n"\
    #             "防火墙故障将导致1800M系统与航空公司数据对接失效，川航、国航、机场等1800M用户无法使用APP服务"
    # p = process_text(test_s, sep=['；'], gibberish=gibberish, stopwords=stopwords)
    # print(p)
    pass
