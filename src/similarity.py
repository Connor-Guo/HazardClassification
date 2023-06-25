# -*- coding utf-8 -*-
"""计算向量相似度"""
import numpy as np
from scipy.spatial.distance import cosine


def sim_v2v(v1: np.ndarray, v2: np.ndarray):
    """计算两个向量的余弦相似度

    Parameters
    ----------
    v1 : np.ndarray
        向量1
    v2 : np.ndarray
        向量2

    Returns
    -------
    float
        余弦相似度
    """
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def sim_v2m(v1, v2):
    num = np.dot([v1], np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


def sim_m2m(v1, v2):
    """两个矩阵的行向量之间的余弦相似度"""
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


def top_sims(mat: np.ndarray, top_n: int = 1):
    """

    Parameters
    ----------
    mat : np.ndarray
        相似度矩阵
    top_n
        返回前n个相似项的索引
    Returns
    -------

    """
    idxs = np.argsort(mat)[::-1]
    return idxs[:, :top_n]


def top_sims_name(mat: np.ndarray, terms, top_n: int = 1):
    idx = top_sims(mat, top_n)
    new_mat = np.empty_like(idx, dtype=object)
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            new_mat[i, j] = terms[idx[i, j]]
    return new_mat


def top_class(mat: np.ndarray, terms, sym_dict):
    """获取相似度最高的子术语所在的事件类别"""
    l_subterm = list(top_sims_name(mat, terms).ravel())
    return [sym_dict[_] for _ in l_subterm]


def sim_word2term(df, dict_terms, term_vec):
    # 此段代码针对how='each'的特征提取结果，得到分类结果label记录在"pred_label"列中
    # 设数据为Di，其中i为索引，对Di的第j个词向量Dij，计算其与term_vec中每个词语的相似度
    print("Calculating similarity...")
    for i in df.index:
        sim_subterm = {}  # 记录每个subterm与当前语句的相似度矩阵
        # 先不优化，计算全部相似度试下能不能跑通
        for subterm in list(term_vec.keys()):
            sim_subterm[subterm] = sim_m2m(np.array(term_vec[subterm]), np.array(df.loc[i, 'doc_vec']))
            # {'跑道侵入': array([[0.545661, 0.59203976, 0.54135513, 0.6044051, 0.5647019,
            #                      0.69200623, 0.62126935, 0.6044051, 0.622167, 0.55367935,
            #                      0.58078223, 0.5270904, 0.60612386, 0.58078223, 0.58281356,
            #                      0.5593316, 0.57528275, 0.6044051, 0.5792388],
            #                     [0.5439216, 0.6010323, 0.6072846, 0.639768, 0.6332015,
            #                      0.59144646, 0.5873276, 0.639768, 0.58819556, 0.60785574,
            #                      0.60477865, 0.6082441, 0.5943184, 0.60477865, 0.55161184,
            #                      0.64878935, 0.5611693, 0.639768, 0.5863357]], dtype=float32))}
            # 数组的行数为subterm的词语长度，列数为待分类不安全事件描述的词语个数
            # 用前10个相似度的平均值衡量真实的相似度
            sim_subterm[subterm] = np.mean(np.sort(sim_subterm[subterm])[-3:])
        # 得到分类结果，记录在df中
        most_similar_subterm = list(term_vec.keys())[0]
        max_similarity = -1.0
        for subterm in list(sim_subterm.keys()):
            if sim_subterm[subterm] > max_similarity:
                most_similar_subterm = subterm
                max_similarity = sim_subterm[subterm]
        df.loc[i, 'most_similar_subterm'] = most_similar_subterm
        df.loc[i, 'max_similarity'] = max_similarity
        df.loc[i, 'pred_label'] = int(dict_terms[most_similar_subterm])
    print("Similarity finished.")
    return df

