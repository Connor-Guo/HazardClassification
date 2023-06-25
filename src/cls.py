"""
模型评估，已作废
被``evaluate.py``替代
"""
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def accuracy(y: List[int],
             y_pred: List[int]):
    """判断分类正确率"""
    l = len(y)
    assert l == len(y_pred)
    a = np.array(y) - np.array(y_pred)
    n_neg = np.count_nonzero(a)
    return 1 - n_neg / l


def acc_cls(y, y_pred):
    # 输出: label accuracy num TP FP TN FN
    l = len(y)
    assert l == len(y_pred)
    if type(y) is pd.Series:
        y = y.astype('int32').tolist()
    if type(y_pred) is pd.Series:
        y_pred = y_pred.astype('int32').tolist()
    # df = pd.DataFrame()
    # df_y = pd.DataFrame({'y': y, "y_pred": y_pred})
    # for i, label in enumerate(df_y['y'].drop_duplicates()):
    #     df.loc[i, 'label'] = label
    #     df.loc[i, 'accuracy'] = accuracy(df_y.loc[df_y['y'] == label, 'y'], df_y.loc[df_y['y'] == label, 'y_pred'])
    #     df.loc[i, 'num'] = df_y.loc[df_y['y'] == label, 'y'].count()
    #     df.loc[i, 'TP'] = df_y
    sns.set()
    f, ax = plt.subplots()
    C2 = confusion_matrix(y, y_pred, labels=pd.Series(y).drop_duplicates().sort_values(ascending=True).tolist())
    # 打印 C2
    # print(C2)
    sns.heatmap(C2, annot=True, ax=ax)  # 画热力图
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x 轴
    ax.set_ylabel('true')  # y 轴
    plt.show()
    return C2


def cls_mat():
    """计算分类矩阵"""


if __name__ == "__main__":
    acc_cls([1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 2, 1, 1, 1, 3, 2, 1, 2, 1, 2, 3])
