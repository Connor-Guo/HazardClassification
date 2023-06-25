import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.metrics import classification_report_imbalanced

from src.loadfiles import load_pickle

"""
imblearn
https://imbalanced-learn.org/stable/introduction.html
"""


class UnsafeDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """y must start from 0"""
        assert min(y) == 0
        assert X.shape[0] == len(y)
        self.X = X
        self.y = y.reshape((X.shape[0], -1))
        self.data = np.concatenate((self.X, self.y), axis=1)
        self.df = pd.DataFrame(data=self.data)

        self.n_classes = pd.DataFrame(y).nunique()[0]

    def __len__(self):
        return len(self.y)

    def count_labels(self, **kwargs):
        """Count the number of labels in the dataset"""
        return pd.DataFrame(self.y).value_counts(**kwargs)

    def train_test_split(
        self,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True
    ):
        return train_test_split(self.X, self.y.reshape(-1),
                                test_size=test_size,
                                train_size=train_size,
                                random_state=random_state,
                                shuffle=shuffle)

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
    x_data = np.array(df['doc_vec'].apply(lambda x: np.mean(x, axis=0)).tolist())
    y_data = df['label'].to_numpy().astype(int) - 1
    dataset = UnsafeDataset(x_data, y_data)
    print(dataset.count_labels())
    print(dataset._is_imbalanced())

    X_train, X_test, y_train, y_test = dataset.train_test_split(test_size=.3)
    eec = EasyEnsembleClassifier(n_estimators=100, random_state=42)
    eec.fit(X_train, y_train)
    y_pred = eec.predict(X_test)
    print(classification_report_imbalanced(y_test, y_pred))
