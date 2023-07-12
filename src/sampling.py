"""
对不平衡数据集的重采样

(UNFINISHED)
"""
import os.path
# from __future__ import annotations

from typing import TypeVar, Tuple, List
from copy import deepcopy
import json
from time import sleep

import pandas as pd
import numpy as np
import openai

from src.loadfiles import *


ResamplerT = TypeVar('ResamplerT', bound='Resampler')


class Resampler:
    def __init__(
        self,
        data: pd.DataFrame,
        id_col: str,
        text_col: str,
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
        self.data_generated = None
        self.id_col = id_col
        self.text_col = text_col
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

    def generate_with_gpt(
        self,
        label: int,
        prompt_str: str,
        expected_n: int,
        save_folder: str,
        lb_text_len: int = 12,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        """
        采用"gpt-3.5-turbo"对一个标签进行过采样。
        保存采样记录至`{label}-log.xlsx`，缓存至`cache.pkl`，返回采样结果。

        Parameters
        ----------
        label : 待采样的标签
        prompt_str : 提示词的字符串，包含两个参数，分别是`n_paraphrase`和`narrative`
        expected_n : 标签的期待样本个数
            如果该数量小于等于已有样本数量，则不会进行采样。返回一个空DataFrame，且不保存log
            否则会持续采样，直到已有样本数量大于等于期待样本数量。
        save_folder : 输出记录文件的路径
            记录文件为excel文件，
        lb_text_len : 采样需要的最小文本长度。低于该长度的样本不会被传给ChatGPT采样
        shuffle : 采样时是否打乱输入
        random_state : 随机数种子

        Returns
        -------
        pd.DataFrame
            返回修改了危险源编号、paraphrase字符串及其标签的Dataframe，其中危险源编号为"样本编号-para-i"
            返回值中不包含原有样本
        """
        assert lb_text_len > 0

        self.data_generated = _oversample_label_gpt35(self.data_original, self.id_col, self.text_col, self.labels_cols,
                                                      label, prompt_str, expected_n, save_folder, lb_text_len,
                                                      shuffle, random_state)
        return self.data_generated


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
        # labels = data_new[labels_cols].to_numpy()
        labels = _get_labels_from_df(data_new, labels_cols)
        filled_labels = _fill_labels(labels, labels_to_del)
        data_new.loc[:, labels_cols] = filled_labels
    return data_new


def _get_labels_from_df(
    df: pd.DataFrame,
    labels_cols: List[str],
) -> np.ndarray:
    return df[labels_cols].to_numpy()


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


def get_completion(prompt: str, model="gpt-3.5-turbo") -> str:
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
               model=model,
               messages=messages,
               temperature=0,  # this is the degree of randomness of the model ’s output
               )
    return response.choices[0].message["content"]


def _json_2_dict(s: str):
    """input string should be the output of `get_completion`"""
    return json.loads(s)


def _get_response_list(
    prompt: str,
    *,
    tries: int = 0,
) -> List[str]:
    if tries > 2:  # 最大重试次数
        print("Retry failed. This sample is skipped.")
        return []
    try:
        response = get_completion(prompt)
        lst = _json_2_dict(response)["Paraphrase"]
    except openai.error.OpenAIError as e:
        print(e, "Retrying...")
        sleep(12)
        lst = _get_response_list(prompt, tries=tries+1)
    except Exception as ee:
        return []
    return lst


def _oversample_label_gpt35(
    df: pd.DataFrame,
    id_column: str,
    text_column: str,
    labels_cols: List[str],
    label: int,
    prompt_str: str,
    expected_n: int,
    save_folder: str,
    lb_text_len: int,
    shuffle: bool,
    random_state: int,
) -> pd.DataFrame:
    """
    采用"gpt-3.5-turbo"对一个标签进行过采样。
    保存采样记录至`{label}-log.xlsx`，缓存至`cache.pkl`，返回采样结果。

    Parameters
    ----------
    df : 包含所有标签的样本的DataFrame
    id_column : 危险源编号的列名
    text_column : 待采样文本的列名
    labels_cols : 标签的列名
    label : 待采样的标签
    prompt_str : 提示词的字符串，包含两个参数，分别是`n_paraphrase`和`narrative`
    expected_n : 标签的期待样本个数
        如果该数量小于等于已有样本数量，则不会进行采样。返回一个空DataFrame，且不保存log
        否则会持续采样，直到已有样本数量大于等于期待样本数量。
    save_folder : 输出记录文件的路径
        记录文件为excel文件，
    lb_text_len : 采样需要的最小文本长度。低于该长度的样本不会被传给ChatGPT采样
    shuffle : 采样时是否打乱输入
    random_state : 随机数种子

    Returns
    -------
    pd.DataFrame
        返回修改了危险源编号、paraphrase字符串及其标签的Dataframe，其中危险源编号为"样本编号-para-i"
        返回值中不包含原有样本
    """
    # log目录和缓存目录
    log_path = os.path.join(save_folder, f"{label}-log.xlsx")
    cache_path = os.path.join(save_folder, "cache.pkl")

    # 从输入的df里拿到所有包含label的行
    labels = _get_labels_from_df(df, labels_cols)
    contain_idx = np.where(label == labels)[0]
    df_label = deepcopy(df).iloc[contain_idx, :]
    # 删除重复和过短的行
    df_label.drop(index=df_label.index[df_label[text_column].duplicated()], inplace=True)
    llb_idx = df_label[text_column].apply(lambda s: True if len(s) < lb_text_len else False)  # 过短为True
    df_label.drop(index=llb_idx.index[llb_idx], inplace=True)
    # 根据当前已有样本数和采样后预期样本总数，计算每个样本需要被采样几次
    n_samples = len(df_label)
    print(f"[Sampling] Label {label} has {n_samples} samples. It is expected to be resampled to {expected_n}.")
    if expected_n <= n_samples:
        print(f"[Sampling] Label {label} need not to be generated.")
        return pd.DataFrame()
    n_paraphrase = np.ceil(expected_n / n_samples)
    # 根据计算的样本数量，调用函数进行采样
    resample_log_lst = []
    resample_df = pd.DataFrame(columns=df.columns)
    # 打乱样本顺序
    if shuffle:
        df_label.sample(frac=1, random_state=random_state)
    # 采样
    for idx, line in df_label.iterrows():
        hae_id = line[id_column]
        narrative = line[text_column]
        prompt = prompt_str.format(int(n_paraphrase), narrative)
        r_lst = _get_response_list(prompt)
        # 存储log和结果
        for i, para in enumerate(r_lst):
            para_id = f"{hae_id}-para-{i+1}"
            # 写log
            resample_log_lst.append(
                {
                    "hae_id": hae_id,
                    "para_id": para_id,
                    "narrative": narrative,
                    "paraphrase": para,
                }
            )
            # 写结果
            new_line = deepcopy(line)
            new_line[id_column] = para_id
            new_line[text_column] = para
            resample_df.loc[len(resample_df)] = new_line
        # 每次得到数据后缓存
        save_pickle(resample_df, cache_path)
        print(f"[Sampling] {len(resample_df)} samples acquired. {max(0, expected_n - len(resample_df) - len(df_label))} left.")
        # 判断是否结束
        if len(resample_df) + len(df_label) >= expected_n:
            print(f"[Sampling] Label {label} resampled to {len(resample_df) + len(df_label)} samples.")
            break
    # 存log
    pd.DataFrame(resample_log_lst).to_excel(log_path, index=False)
    # 返回修改了危险源编号、paraphrase字符串及其标签的Dataframe，其中危险源编号为"{样本编号}-para-{i}"
    # 返回值中不包含原有样本
    return resample_df


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


