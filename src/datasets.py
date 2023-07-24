"""

This file is deprecated.

"""
from copy import deepcopy
from typing import Union, TypeVar

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.loadfiles import load_pickle

"""
imblearn
https://imbalanced-learn.org/stable/introduction.html
"""
HazardDatasetT = TypeVar('HazardDatasetT', bound='HazardDataset')


class HazardDataset:
    def __init__(
            self,
            data: pd.DataFrame,
            id_col: str,
            text_col: str,
            labels_cols: list[str],
            chinese_labels_cols: list[str] = None,
            ) -> None:
        # label从1开始
        self.df = data
        self.id_col = id_col
        self.text_col = text_col
        self.labels_cols = labels_cols
        self.chinese_labels_cols = chinese_labels_cols

        # 继承df的属性
        self.columns = self.df.columns

        # 计算得到的属性
        self._labels = self.get_labels_numpy()
        self.n_classes = self._labels.max()  # 不能这么用

        self.drop = self.df.drop

    def _create_with_current_attributes(self, df):
        return HazardDataset(df, self.id_col, self.text_col, self.labels_cols, self.chinese_labels_cols)

    def __len__(self):
        return len(self.df)

    def count_labels(self, **kwargs):
        """Count the number of labels in the dataset"""
        return self.df.value_counts(**kwargs)

    def get_labels_numpy(self, *, na_value=0):
        return self.df[self.labels_cols].to_numpy(na_value=na_value).astype(int)

    def get_data_by_label(self, label, *, deep_copy=True):
        # 从输入的df里拿到所有包含label的行
        labels = self.get_labels_numpy()
        contain_idx = np.where(label == labels)[0]
        if deep_copy:
            return self._create_with_current_attributes(
                deepcopy(self.df).iloc[contain_idx, :])
        else:
            return self._create_with_current_attributes(
                self.df.iloc[contain_idx, :])

    def shuffle(self, *, random_state=42):
        self.df = self.df.sample(frac=1, random_state=random_state)

    def train_test_split(
        self,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True
    ):
        # return train_test_split(self.X, self.y.reshape(-1),
        #                         test_size=test_size,
        #                         train_size=train_size,
        #                         random_state=random_state,
        #                         shuffle=shuffle)
        pass

    def _oversampling(self):
        pass

    def _is_imbalanced(self, lc=0.8, uc=1.2):
        """Return a bool indicating whether the dataset is imbalanced."""
        # 假设数据集规模为N，包含m种标签
        # 那么当每种标签的数据量n分布在[N/m*0.8, N/m*1.2]时，认为其是平衡的
        r = self._sample_label_ratio()
        for idx, n in self.count_labels().items():
            if n > r * uc or n < r * lc:
                return True
        return False

    def _sample_label_ratio(self):
        """return the sample-label ratio of the dataset"""
        return len(self) / self.n_classes


if __name__ == "__main__":
    df = load_pickle("../out/dump/guanzhi_sep_vec_each_1000.pkl")

    dataset = HazardDataset()
    print(dataset.count_labels())
    print(dataset._is_imbalanced())


