from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import pickle
import numpy as np

from src.evaluate import Evaluator
from src.loadfiles import load_pickle


"""
test_size=0.2
              precision    recall  f1-score   support

           1       1.00      0.50      0.67         4
           2       0.86      1.00      0.92       125
           3       1.00      0.20      0.33         5
           4       0.00      0.00      0.00         1
           5       1.00      0.50      0.67         6
           6       1.00      1.00      1.00         3
           7       1.00      0.75      0.86         4
           8       0.85      0.94      0.89        18
           9       0.00      0.00      0.00         1
          10       1.00      0.69      0.82        13
          11       0.67      0.86      0.75         7
          12       0.00      0.00      0.00         2
          13       0.00      0.00      0.00         1
          14       0.86      0.86      0.86         7
          15       1.00      0.50      0.67         8
          17       0.00      0.00      0.00         1
          18       0.72      0.68      0.70        19

    accuracy                           0.85       225
   macro avg       0.64      0.50      0.54       225
weighted avg       0.84      0.85      0.83       225
"""


def train_catboost(X_train, X_test, y_train, y_test):
    # X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=33)

    params = {
        'iterations': 300,
        'learning_rate': 0.1,
        'depth': 5,
        'loss_function': 'MultiClass'
    }

    clf = CatBoostClassifier(**params)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)

    # 模型评估
    eva = Evaluator(y_test, y_predict)
    print(eva.evaluate())
    eva.save("../evaluation", "Catboost-iter300-lr01-dep5")

    return clf, eva


if __name__ == "__main__":
    df_train = load_pickle(r'C:\Users\10507\OneDrive\桌面\202301 危险源挖掘论文\不安全事件分类'
                     r'\code\out\datasets\guanzhi_sep_vec_each_1000_select-train.pkl')
    df_test = load_pickle(r'C:\Users\10507\OneDrive\桌面\202301 危险源挖掘论文\不安全事件分类'
                     r'\code\out\datasets\guanzhi_sep_vec_each_1000_select-test.pkl')
    x_train = np.array(df_train['doc_vec'].apply(lambda x: np.mean(x, axis=0)).tolist())
    y_train = df_train['label'].to_numpy().astype(int)
    x_test = np.array(df_test['doc_vec'].apply(lambda x: np.mean(x, axis=0)).tolist())
    y_test = df_test['label'].to_numpy().astype(int)
    clf, eva = train_catboost(x_train, x_test, y_train, y_test)


