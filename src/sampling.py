"""
对不平衡数据集的重采样

(UNFINISHED)
"""
# from __future__ import annotations

import pandas as pd
import numpy as np
from typing import TypeVar, Tuple, List
from copy import deepcopy

ResamplerT = TypeVar('ResamplerT', bound='Resampler')


class Resampler:
    def __init__(
        self,
        data: pd.DataFrame,
        labels_cols: List[str],
        chinese_labels_cols: List[str] = None,
    ) -> None:
        """"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected input type pd.DataFrame, but {type(data)} is provided.")
        if chinese_labels_cols is not None:
            assert len(labels_cols) == len(chinese_labels_cols)
        assert len(labels_cols) > 0
        # 检查列

        self.data_original = data
        self.data_resampled = None
        self.labels_cols = labels_cols
        self.chinese_labels_cols = chinese_labels_cols

    def sample(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
    ) -> ResamplerT:
        """对样本重采样，具体方案见函数"""
        assert len(data) == labels.shape[0]  # 输入数据的长度要相等
        self.data_original = (data, labels)
        self.data_resampled = None
        return self

    def del_labels(
        self,
        labels_to_del: List[int]
    ) -> pd.DataFrame:
        if isinstance(labels_to_del, list):
            labels_to_del = np.array(labels_to_del)
        elif isinstance(labels_to_del, np.ndarray):
            pass
        else:
            raise TypeError(f"Type of argument \"labels_to_del\" should be list or np.ndarray, but {type(labels_to_del)} is provided.")
        self.data_resampled = _del_labels(self.data_original, labels_to_del, self.labels_cols, self.chinese_labels_cols, fill_labels=True)
        return self.data_resampled


def _resample(
    data: pd.DataFrame,
    labels_cols: List[str],
    chinese_labels_cols: List[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    对样本重采样。data若需要去除对应中文标签或数字标签，需要声明参数。

    Parameters
    ----------
    data :
        元数据的dataframe。
    labels_cols :
        data中包含标签的列名。
    labels :
        标签构成的array。标签的值从1开始。
        默认从labels_cols中提取数据。
    chinese_labels_cols :
        data中包含中文标签的列名。

    Returns
    -------


    """
    assert len(labels_cols) > 0
    assert chinese_labels_cols is None or len(chinese_labels_cols) == len(labels_cols)

    labels = data[labels_cols].to_numpy(na_value=0).astype(int)

    # 删除所有样本低于50的标签

    # 对样本容量在50 - 100的标签，过采样为150条；
    # 对样本容量在100 - 200的标签，过采样为200条；
    # 对样本容量在200以上的标签，过 / 欠采样为250条。


def _del_labels(
    data: pd.DataFrame,
    labels_to_del: np.ndarray,
    labels_cols: List[str],
    chinese_labels_cols: List[str] = None,
    fill_labels: bool = True,
) -> pd.DataFrame:
    """
    删除指定标签与对应的数据，并将后方的标签编号前移。

    Parameters
    ----------
    data
    labels_to_del
    labels_cols
    chinese_labels_cols
    fill_labels

    Returns
    -------

    Examples
    --------
    >>> data = pd.DataFrame({
    >>> "label1": [1, 2, 1, 3],
    >>> "label2": [2, 3, 3, np.nan],
    >>>  "text1":  ['A', 'B', 'A', 'C'],
    >>>  "text2":  ['B', 'C', 'C', np.nan],
    >>> })
    >>> data
           label1  label2 text1 text2
    0       1     2.0     A     B
    1       2     3.0     B     C
    2       1     3.0     A     C
    3       3     NaN     C   NaN

    >>> labels_to_del = [3]
    >>> labels_cols = ['label1', 'label2']
    >>> chinese_labels_cols = ['text1', 'text2']
    >>> _del_labels(data, labels_to_del, labels_cols, chinese_labels_cols, fill_labels=False)
           label1  label2 text1 text2
    0       1     2.0     A     B
    1       2     0.0     B   NaN
    2       1     0.0     A   NaN
    """
    # 数据类型转换
    if not isinstance(labels_to_del, np.ndarray):
        labels_to_del = np.array(labels_to_del)

    # 删除{待删除标签}中的标签
    data_new = deepcopy(data)
    for label_to_del in labels_to_del.astype(int):
        data_new = _del_single_label(data_new, label_to_del, labels_cols, chinese_labels_cols)

    if fill_labels:
        labels = data_new[labels_cols].to_numpy()
        filled_labels = _fill_labels(labels, labels_to_del)
        data_new.loc[:, labels_cols] = filled_labels
    return data_new


def _del_single_label(
    data: pd.DataFrame,
    label_to_del: int,
    labels_cols: List[str],
    chinese_labels_cols: List[str] = None,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    data
    label_to_del
    labels_cols
    chinese_labels_cols

    Returns
    -------

    Examples
    --------
    >>> data = pd.DataFrame({
    >>>     "label1": [1, 2, 1, 3],
    >>>     "label2": [2, 3, 3, np.nan],
    >>>     "text1":  ['A', 'B', 'A', 'C'],
    >>>     "text2":  ['B', 'C', 'C', np.nan],
    >>> })
    >>> data
           label1  label2 text1 text2
    0       1     2.0     A     B
    1       2     3.0     B     C
    2       1     3.0     A     C
    3       3     NaN     C   NaN

    >>> label_to_del = 3
    >>> labels_cols = ['label1', 'label2']
    >>> chinese_labels_cols = ['text1', 'text2']
    >>> _del_single_label(data, label_to_del, labels_cols, chinese_labels_cols)
       label1  label2 text1 text2
    0       1     2.0     A     B
    1       2     0.0     B   NaN
    2       1     0.0     A   NaN
    """
    data_copy = deepcopy(data)

    for idx, line in data.iterrows():
        # 提取label和中文的部分
        labels = line.loc[labels_cols].to_numpy(na_value=0).astype(int)
        if chinese_labels_cols is not None:
            chinese = line.loc[chinese_labels_cols].to_numpy()

        if label_to_del in labels:  # 1a如果这行存在要删除的标签
            if len(labels) == 1 or np.nan_to_num(labels, nan=0).sum() == label_to_del:  # 2a.如果这行只有这一个标签，那么直接删除这行
                data_copy.drop(index=idx, inplace=True)
            else:  # 2b.如果这行不止一个标签，那么替换标签
                start = np.argwhere(labels == label_to_del).ravel()[0]
                if start == len(labels) - 1:  # 3a如果被替换的标签刚好是最后一个，直接替换为0
                    labels[start] = 0
                    if chinese_labels_cols is not None:
                        chinese[start] = np.nan
                else:  # 3b如果被替换的标签不是最后一个，则标签整体前移
                    labels[start:-1] = labels[start + 1:]
                    labels[-1] = 0
                    if chinese_labels_cols is not None:
                        chinese[start:-1] = chinese[start + 1:]
                        chinese[-1] = np.nan
                # 4将labels的0.0全部换为np.nan
                labels = np.where(labels, labels, np.nan)
                # 执行操作，替换data_copy
                data_copy.loc[idx, labels_cols] = labels
                if chinese_labels_cols is not None:
                    data_copy.loc[idx, chinese_labels_cols] = chinese
        else:  # 1b如果这行不用操作
            pass

    return data_copy


def _fill_labels(
    labels: np.ndarray,
    labels_to_del: np.ndarray,
) -> np.ndarray:
    """
    Move the labels forward to make the labels continuous.

    Examples
    --------
        >>> labels = np.array([[1, 3, 0], [1, 5, 6]])  # labels 2,4 are absent
        >>> labels_to_del = np.array([2, 4])
        >>> _fill_labels(labels, labels_to_del)
        array([[1, 2],
               [1, 3]])  # label 3 becomes label 2, and label 5 becomes label 3

    """
    labels_to_del = deepcopy(labels_to_del)
    filled_labels = labels[:]  # make a copy

    labels_to_del.sort()
    # Consider that each time a label is filled, all label values greater than that label get 1 smaller,
    # It's better to generate a new array of labels_to_del in advance.
    labels_to_del_trans = [_ - i for i, _ in enumerate(labels_to_del)]
    for label in labels_to_del_trans:
        for i in range(filled_labels.shape[0]):
            for j in range(filled_labels.shape[1]):
                if filled_labels[i, j] > label:
                    filled_labels[i, j] = filled_labels[i, j] - 1
    return filled_labels


if __name__ == "__main__":
    data = pd.DataFrame({
        "label1": [1, 2, 1, 3],
        "label2": [2, 3, 3, np.nan],
        "text1": ['A', 'B', 'A', 'C'],
        "text2": ['B', 'C', 'C', np.nan],
    })
    label_to_del = 3
    labels_cols = ['label1', 'label2']
    chinese_labels_cols = ['text1', 'text2']
    _del_single_label(data, label_to_del, labels_cols, chinese_labels_cols)


